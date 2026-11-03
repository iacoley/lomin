#!/usr/bin/env python3
"""Tests for multimodal camera & BLE IMA ADPCM ingest (vision_adpcm.py and gateway.py)."""

from __future__ import annotations

import base64
import http.client
import json
import math
import struct
import threading
import uuid

import pytest

import gateway
from vision_adpcm import (
    ADPCM_MAGIC,
    MAX_ADPCM_BYTES,
    MAX_IMAGE_B64_BYTES,
    decode_adpcm_to_wav,
    decode_ima_adpcm,
    encode_adpcm,
    encode_ima_adpcm,
    extract_vision_payloads,
)

# Standard test WAV (16-bit mono 16 kHz, 0.1s sine wave)
SAMPLE_RATE = 16000
NUM_SAMPLES = 1600
SAMPLES = [int(16000 * math.sin(2 * math.pi * 440 * i / SAMPLE_RATE)) for i in range(NUM_SAMPLES)]
PCM_16K = struct.pack(f"<{NUM_SAMPLES}h", *SAMPLES)
WAV_16K = gateway.pcm_to_wav(PCM_16K, SAMPLE_RATE)


@pytest.fixture()
def server():
    srv = gateway.Server(("127.0.0.1", 0), gateway.GatewayHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    gateway._ambient_reset_state()
    monkeypatch.setattr(gateway, "NODES_PATH", tmp_path / "nodes.json")
    monkeypatch.setattr(gateway, "NODE_TARGETS_PATH", tmp_path / "node-targets.json")
    monkeypatch.setattr(gateway, "WATCH_TOKENS_PATH", tmp_path / "watch-tokens.json")
    monkeypatch.delenv("MVG_NODE_TARGETS", raising=False)
    monkeypatch.delenv("MVG_TRUSTED_KEY", raising=False)
    monkeypatch.setattr(gateway, "VOICE_ID_GRACE_SEC", 0.5)
    monkeypatch.setattr(gateway, "synthesize", lambda text: (WAV_16K, "fake"))
    yield
    gateway._ambient_reset_state()


class BrainRecorder:
    def __init__(self):
        self.calls = []
        self.kwargs_list = []
        self.ctx_list = []

    def __call__(self, prompt, session_id, node_id, speaker, speaker_score, timeout, on_partial, **kwargs):
        ctx = getattr(gateway._FRAME_CTX, "ctx", None)
        self.calls.append({"prompt": prompt, "session_id": session_id, "node_id": node_id})
        self.kwargs_list.append(kwargs)
        self.ctx_list.append(dict(ctx) if ctx else {})
        reply = "I see your photo and hear you."
        on_partial(reply)
        return reply


def parse_sse(raw: bytes):
    events, ev = [], None
    for line in raw.decode().splitlines():
        if line.startswith("event:"):
            ev = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            events.append((ev, json.loads(line.split(":", 1)[1])))
    return events


def build_multipart(parts: dict[str, tuple[str, bytes] | bytes | str]):
    boundary = "----vision-test-" + uuid.uuid4().hex
    body = bytearray()
    for k, v in parts.items():
        body += f"--{boundary}\r\n".encode()
        if isinstance(v, tuple):
            filename, data = v
            body += f'Content-Disposition: form-data; name="{k}"; filename="{filename}"\r\n'.encode()
            body += b"Content-Type: application/octet-stream\r\n\r\n"
            body += data + b"\r\n"
        elif isinstance(v, (bytes, bytearray)):
            body += f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode()
            body += v + b"\r\n"
        else:
            body += f'Content-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    body += f"--{boundary}--\r\n".encode()
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def post_turn(port, parts: dict[str, tuple[str, bytes] | bytes | str]):
    body, ctype = build_multipart(parts)
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("POST", "/voice/turn", body=body, headers={"Content-Type": ctype})
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, data


# ============================================================================
# 1. ADPCM Constants & IMA Codec Tests
# ============================================================================

def test_adpcm_magic_constant():
    assert ADPCM_MAGIC == b"APCM"


def test_decode_adpcm_passthrough_on_non_adpcm():
    assert decode_adpcm_to_wav(b"") == b""
    assert decode_adpcm_to_wav(b"RIF") == b"RIF"
    # Standard RIFF WAV must be passed through bit-for-bit unchanged
    assert decode_adpcm_to_wav(WAV_16K) == WAV_16K
    # Arbitrary binary data not starting with APCM must not be touched
    arbitrary = b"\x00\x01\x02\x03\x04\x05\x06\x07"
    assert decode_adpcm_to_wav(arbitrary) == arbitrary


def test_ima_adpcm_raw_encode_decode():
    encoded = encode_ima_adpcm(PCM_16K)
    # IMA ADPCM compresses 16-bit (2 bytes) per sample into 4 bits (0.5 byte) per sample: 4:1 compression
    assert len(encoded) == len(PCM_16K) // 4
    decoded_pcm = decode_ima_adpcm(encoded, num_samples=NUM_SAMPLES)
    assert len(decoded_pcm) == len(PCM_16K)

    # Decompress back to samples and verify quantization fidelity
    orig_samples = struct.unpack(f"<{NUM_SAMPLES}h", PCM_16K)
    dec_samples = struct.unpack(f"<{NUM_SAMPLES}h", decoded_pcm)
    mean_err = sum(abs(o - d) for o, d in zip(orig_samples, dec_samples)) / len(orig_samples)
    assert mean_err < 300
    # After initial step-size adaptation (sample 20+), peak tracking error is well bounded
    post_lock_max = max(abs(o - d) for o, d in zip(orig_samples[20:], dec_samples[20:]))
    assert post_lock_max < 1000


def test_adpcm_container_roundtrip():
    adpcm_container = encode_adpcm(WAV_16K, sample_rate=16000)
    assert adpcm_container[:4] == ADPCM_MAGIC
    # Header is 16 bytes: <4sIIhBB
    assert len(adpcm_container) == 16 + (len(PCM_16K) // 4)

    # Decode container to WAV
    wav_out = decode_adpcm_to_wav(adpcm_container)
    assert wav_out[:4] == b"RIFF"
    assert wav_out[8:12] == b"WAVE"
    parsed_sr = struct.unpack_from("<I", wav_out, 24)[0]
    assert parsed_sr == 16000
    # WAV has 44-byte header + decoded PCM
    assert len(wav_out) == 44 + len(PCM_16K)


def test_decode_adpcm_truncated_header():
    # Only magic bytes
    res = decode_adpcm_to_wav(b"APCM")
    assert res[:4] == b"RIFF"
    assert len(res) == 44  # empty WAV

    # Truncated 10 bytes
    res2 = decode_adpcm_to_wav(b"APCM\x80\x3e\x00\x00\x01\x02")
    assert res2[:4] == b"RIFF"


# ============================================================================
# 2. Vision Payload Extraction Tests
# ============================================================================

def test_extract_vision_payloads_empty_and_invalid():
    assert extract_vision_payloads({}) == (None, [])
    assert extract_vision_payloads(None) == (None, [])
    assert extract_vision_payloads("invalid") == (None, [])
    assert extract_vision_payloads({"audio": b"..."}) == (None, [])


def test_extract_vision_payloads_single_image():
    raw_img = b"\xff\xd8\xff\xe0\x00\x10JFIF"
    img_b64 = base64.b64encode(raw_img).decode("ascii")

    # Raw bytes in 'image'
    img, burst = extract_vision_payloads({"image": raw_img})
    assert img == raw_img
    assert burst == []

    # Base64 string in 'image_b64'
    img, burst = extract_vision_payloads({"image_b64": img_b64})
    assert img == raw_img
    assert burst == []

    # Base64 bytes in 'image_b64'
    img, burst = extract_vision_payloads({"image_b64": img_b64.encode("ascii")})
    assert img == raw_img
    assert burst == []

    # Base64 string in 'image'
    img, burst = extract_vision_payloads({"image": img_b64})
    assert img == raw_img
    assert burst == []


def test_extract_vision_payloads_burst_frames():
    f0 = b"\xff\xd8\xff" + b"frame-0-bytes"
    f1 = b"\xff\xd8\xff" + b"frame-1-bytes"
    f2 = b"\xff\xd8\xff" + b"frame-2-bytes"

    # Numbered fields image_0 .. image_2
    img, burst = extract_vision_payloads({"image_0": f0, "image_1": f1, "image_2": f2})
    assert img is None
    assert burst == [f0, f1, f2]

    # burst_images_b64 list
    b64_list = [base64.b64encode(x).decode("ascii") for x in (f0, f1, f2)]
    img, burst = extract_vision_payloads({"burst_images_b64": b64_list})
    assert img is None
    assert burst == [f0, f1, f2]

    # burst_images raw bytes list
    img, burst = extract_vision_payloads({"burst_images": [f0, f1, f2]})
    assert img is None
    assert burst == [f0, f1, f2]


def test_extract_vision_payloads_combined_single_and_burst():
    main_img = b"\xff\xd8\xff" + b"main-camera-frame"
    burst0 = b"\xff\xd8\xff" + b"burst-frame-0"
    burst1 = b"\xff\xd8\xff" + b"burst-frame-1"

    img, burst = extract_vision_payloads({"image": main_img, "image_0": burst0, "image_1": burst1})
    assert img == main_img
    assert burst == [burst0, burst1]


def test_decode_adpcm_size_cap():
    oversized = ADPCM_MAGIC + b"\x00" * (MAX_ADPCM_BYTES + 1)
    with pytest.raises(ValueError, match="exceeds"):
        decode_adpcm_to_wav(oversized)

    with pytest.raises(ValueError, match="exceeds"):
        decode_ima_adpcm(b"\x00" * (MAX_ADPCM_BYTES + 1))


def test_extract_vision_payloads_validation():
    # Non-base64 text strings must be rejected
    img, burst = extract_vision_payloads({"image": "not-valid-base64-text!"})
    assert img is None
    assert burst == []

    img, burst = extract_vision_payloads({"image_b64": "invalid base64 ???"})
    assert img is None
    assert burst == []

    # Oversized base64 string must be rejected
    oversized_str = "A" * (MAX_IMAGE_B64_BYTES + 10)
    img, burst = extract_vision_payloads({"image_b64": oversized_str})
    assert img is None
    assert burst == []

    # burst_images_b64 must be a list
    img, burst = extract_vision_payloads({"burst_images_b64": "not-a-list"})
    assert img is None
    assert burst == []

    # burst_images_b64 filters out non-string and non-base64 items
    valid_b64 = base64.b64encode(b"\xff\xd8\xff" + b"frame1").decode("ascii")
    img, burst = extract_vision_payloads({"burst_images_b64": [valid_b64, "invalid_b64!", 12345]})
    assert img is None
    assert burst == [b"\xff\xd8\xff" + b"frame1"]


# ============================================================================
# 3. Gateway Integration: Context, Brain Call & Environment Gating
# ============================================================================

def test_make_frame_ctx_vision_fields():
    ctx_clean = gateway.make_frame_ctx(None)
    assert "image_b64" not in ctx_clean
    assert "burst_images_b64" not in ctx_clean

    ctx_with_vision = gateway.make_frame_ctx(
        None,
        image_b64="main_b64_str",
        burst_images_b64=["f0_b64", "f1_b64"],
    )
    assert ctx_with_vision["image_b64"] == "main_b64_str"
    assert ctx_with_vision["burst_images_b64"] == ["f0_b64", "f1_b64"]


def test_http_brain_call_vision_gating(monkeypatch):
    posted_payloads = []

    class DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps({"reply": "ok"}).encode("utf-8")

    def dummy_urlopen(req, timeout=None):
        data = json.loads(req.data.decode("utf-8"))
        posted_payloads.append(data)
        return DummyResponse()

    monkeypatch.setattr(gateway, "MVG_BRAIN_URL", "http://brain.local/v1/chat")
    monkeypatch.setattr(gateway, "urlopen", dummy_urlopen)

    # 1. When MVG_ENABLE_VISION is disabled (0), image fields are omitted
    monkeypatch.setenv("MVG_ENABLE_VISION", "0")
    monkeypatch.setattr(gateway, "ENABLE_VISION", False)
    gateway._FRAME_CTX.ctx = {"image_b64": "img123", "burst_images_b64": ["b1"]}
    gateway.http_brain_call("hello", "sess1", "node1", "speaker1", 1.0, 5)
    assert "image_b64" not in posted_payloads[-1]
    assert "burst_images_b64" not in posted_payloads[-1]

    # 2. When MVG_ENABLE_VISION is enabled (1), image fields are included
    monkeypatch.setenv("MVG_ENABLE_VISION", "1")
    monkeypatch.setattr(gateway, "ENABLE_VISION", True)
    gateway.http_brain_call("hello", "sess1", "node1", "speaker1", 1.0, 5)
    assert posted_payloads[-1]["image_b64"] == "img123"
    assert posted_payloads[-1]["burst_images_b64"] == ["b1"]


# ============================================================================
# 4. Gateway /voice/turn HTTP Multipart & ADPCM End-to-End Tests
# ============================================================================

def test_voice_turn_adpcm_ingest(server, monkeypatch):
    recorder = BrainRecorder()
    monkeypatch.setattr(gateway, "speech_gate", lambda *args, **kw: ("passed", 900.0, 1.0))
    monkeypatch.setattr(gateway, "transcribe", lambda wav_bytes: "transcribed prompt")
    monkeypatch.setattr(gateway, "identify_speaker", lambda *args, **kw: ("", 0.0))
    monkeypatch.setattr(gateway, "decide_addressed", lambda *args, **kw: {"score": 0.9, "classifier": "jev", "laya_score": None})
    monkeypatch.setattr(gateway, "STREAM_BRAIN_CALL", recorder)

    # Encode test audio to ADPCM container
    adpcm_data = encode_adpcm(WAV_16K, sample_rate=16000)

    # Enable ADPCM
    monkeypatch.setenv("MVG_ENABLE_ADPCM", "1")
    monkeypatch.setattr(gateway, "ENABLE_ADPCM", True)

    parts = {
        "node_id": "test_node",
        "audio": ("audio.apcm", adpcm_data),
        "tts": "none",
    }
    status, data = post_turn(server, parts)
    assert status == 200
    events = parse_sse(data)
    assert any(e == "reply" for e, _ in events)
    assert len(recorder.calls) == 1
    assert recorder.calls[0]["prompt"] == "transcribed prompt"


def test_voice_turn_multimodal_camera_multipart(server, monkeypatch):
    recorder = BrainRecorder()
    monkeypatch.setattr(gateway, "speech_gate", lambda *args, **kw: ("passed", 900.0, 1.0))
    monkeypatch.setattr(gateway, "transcribe", lambda wav_bytes: "what is this object")
    monkeypatch.setattr(gateway, "identify_speaker", lambda *args, **kw: ("", 0.0))
    monkeypatch.setattr(gateway, "decide_addressed", lambda *args, **kw: {"score": 0.9, "classifier": "jev", "laya_score": None})
    monkeypatch.setattr(gateway, "STREAM_BRAIN_CALL", recorder)

    test_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF-TEST-IMAGE"
    burst_f0 = b"\xff\xd8\xff" + b"burst-frame-0-jpeg"
    burst_f1 = b"\xff\xd8\xff" + b"burst-frame-1-jpeg"

    expected_img_b64 = base64.b64encode(test_jpeg).decode("ascii")
    expected_b0_b64 = base64.b64encode(burst_f0).decode("ascii")
    expected_b1_b64 = base64.b64encode(burst_f1).decode("ascii")

    # Case A: MVG_ENABLE_VISION=1 -> vision payloads passed into brain kwargs and ctx
    monkeypatch.setenv("MVG_ENABLE_VISION", "1")
    monkeypatch.setattr(gateway, "ENABLE_VISION", True)

    parts = {
        "node_id": "test_node",
        "audio": ("audio.wav", WAV_16K),
        "image": ("camera.jpg", test_jpeg),
        "image_0": ("f0.jpg", burst_f0),
        "image_1": ("f1.jpg", burst_f1),
        "tts": "none",
    }
    status, data = post_turn(server, parts)
    assert status == 200
    events = parse_sse(data)
    assert any(e == "reply" for e, _ in events)

    assert len(recorder.kwargs_list) == 1
    assert recorder.kwargs_list[0]["image_b64"] == expected_img_b64
    assert recorder.kwargs_list[0]["burst_images_b64"] == [expected_b0_b64, expected_b1_b64]
    assert recorder.ctx_list[0]["image_b64"] == expected_img_b64
    assert recorder.ctx_list[0]["burst_images_b64"] == [expected_b0_b64, expected_b1_b64]

    # Case B: MVG_ENABLE_VISION=0 -> vision payloads ignored (disabled by default)
    monkeypatch.setenv("MVG_ENABLE_VISION", "0")
    monkeypatch.setattr(gateway, "ENABLE_VISION", False)

    status, data = post_turn(server, parts)
    assert status == 200
    assert len(recorder.kwargs_list) == 2
    assert "image_b64" not in recorder.kwargs_list[1]
    assert "burst_images_b64" not in recorder.kwargs_list[1]
    assert "image_b64" not in recorder.ctx_list[1]
    assert "burst_images_b64" not in recorder.ctx_list[1]
