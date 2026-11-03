#!/usr/bin/env python3
"""Direct unit tests for review findings on PR #5:
- ADPCM 512KB size cap (413 Payload Too Large) & array.array('h') decoding
- Upfront inspect.signature check in _stream_brain_turn (no TypeError retries)
- Vision payload validation (reject non-base64 text, enforce 5MB cap, type check)
- Dual env var flag support (MVG_ENABLE_VISION/MVG_VISION_ENABLED, MVG_ENABLE_ADPCM/MVG_ADPCM_ENABLED, MVG_VISION_ADPCM)
"""

from __future__ import annotations

import base64
import http.client
import json
import math
import struct
import threading

import pytest

import gateway
from vision_adpcm import (
    ADPCM_MAGIC,
    MAX_ADPCM_BYTES,
    MAX_IMAGE_B64_BYTES,
    decode_adpcm_to_wav,
    decode_ima_adpcm,
    encode_adpcm,
    extract_vision_payloads,
)


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


def test_adpcm_upload_size_cap_413(server, monkeypatch):
    """An ADPCM upload exceeding MAX_ADPCM_BYTES (512 KiB) returns HTTP 413."""
    monkeypatch.setenv("MVG_ENABLE_ADPCM", "1")
    oversized_adpcm = ADPCM_MAGIC + b"\x00" * (MAX_ADPCM_BYTES + 64)

    conn = http.client.HTTPConnection("127.0.0.1", server, timeout=5)
    headers = {
        "Content-Type": "audio/wav",
        "Content-Length": str(len(oversized_adpcm)),
    }
    conn.request("POST", "/voice/turn?node_id=test-node", body=oversized_adpcm, headers=headers)
    resp = conn.getresponse()
    body = resp.read()
    conn.close()

    assert resp.status == 413
    data = json.loads(body.decode("utf-8"))
    assert data.get("error") == "payload_too_large"


def test_adpcm_array_decoding():
    """Verify decode_ima_adpcm accurately decodes via array.array('h')."""
    sr = 16000
    n = 800
    samples = [int(12000 * math.sin(2 * math.pi * 440 * i / sr)) for i in range(n)]
    pcm_in = struct.pack(f"<{n}h", *samples)
    adpcm = encode_adpcm(pcm_in, sample_rate=sr)

    wav_out = decode_adpcm_to_wav(adpcm, sample_rate=sr)
    assert len(wav_out) >= 44 + n * 2
    pcm_out = wav_out[44: 44 + n * 2]

    # Direct test of decode_ima_adpcm array decoding
    raw_pcm = decode_ima_adpcm(adpcm[16:], num_samples=n)
    assert len(raw_pcm) == n * 2

    # Verify unpacked decoded samples match within IMA ADPCM quantization loss
    out_samples = struct.unpack(f"<{n}h", pcm_out)
    diffs = [abs(a - b) for a, b in zip(samples, out_samples)]
    avg_diff = sum(diffs) / len(diffs)
    assert avg_diff < 1500  # typical IMA ADPCM error threshold for 16-bit sine


def test_stream_brain_turn_signature_inspection_no_kwargs():
    """A brain callable without **kwargs or vision args is invoked cleanly via signature filtering."""
    calls = []

    def simple_brain(prompt, session_id, node_id, speaker, speaker_score, timeout, on_partial):
        calls.append({"prompt": prompt, "node_id": node_id})
        on_partial("Answer from simple brain.")
        return "Answer from simple brain."

    class DummyHandler:
        def _sse(self, event, data):
            return True

    handler = DummyHandler()
    ctx = {
        "node_id": "test-node",
        "image_b64": base64.b64encode(b"img").decode("ascii"),
        "burst_images_b64": [base64.b64encode(b"f1").decode("ascii")],
    }
    opts = {"tts": "none"}

    orig_brain = gateway.STREAM_BRAIN_CALL
    try:
        gateway.STREAM_BRAIN_CALL = simple_brain
        # Call _stream_brain_turn on the dummy handler with vision enabled
        outcome, reply = gateway.GatewayHandler._stream_brain_turn(
            handler,
            t0=0.0,
            turn_id="turn-1",
            prompt="Hello brain",
            session_id="sess-1",
            node_id="test-node",
            speaker="alice",
            speaker_score=0.9,
            ctx=ctx,
            endpoint="/ask/stream",
            opts=opts,
        )
        assert reply == "Answer from simple brain."
        assert len(calls) == 1
    finally:
        gateway.STREAM_BRAIN_CALL = orig_brain


def test_stream_brain_turn_internal_type_error_never_retried():
    """An internal TypeError raised inside the brain callable must NEVER cause a second call."""
    calls = []

    def buggy_brain(prompt, session_id, node_id, speaker, speaker_score, timeout, on_partial, **kwargs):
        calls.append(len(calls) + 1)
        # Internal TypeError caused by a bug inside the brain implementation
        raise TypeError("internal bug: unsupported operand type for +: 'NoneType' and 'int'")

    class DummyHandler:
        def _sse(self, event, data):
            return True

    handler = DummyHandler()
    ctx = {"node_id": "test-node"}
    opts = {"tts": "none"}

    orig_brain = gateway.STREAM_BRAIN_CALL
    try:
        gateway.STREAM_BRAIN_CALL = buggy_brain
        outcome, reply = gateway.GatewayHandler._stream_brain_turn(
            handler,
            t0=0.0,
            turn_id="turn-err",
            prompt="Trigger error",
            session_id="sess-err",
            node_id="test-node",
            speaker="bob",
            speaker_score=0.8,
            ctx=ctx,
            endpoint="/ask/stream",
            opts=opts,
        )
        # Must only be called ONCE - no retry on TypeError!
        assert len(calls) == 1
    finally:
        gateway.STREAM_BRAIN_CALL = orig_brain


def test_vision_payload_validation_rejects_non_base64_text():
    """Non-base64 text strings must be rejected and not treated as raw bytes."""
    # Plain text strings in 'image' or 'image_b64' must return None
    img, burst = extract_vision_payloads({"image": "not a base64 encoded string!"})
    assert img is None
    assert burst == []

    img, burst = extract_vision_payloads({"image_b64": "hello world plain text"})
    assert img is None
    assert burst == []


def test_vision_payload_validation_size_caps():
    """Vision payloads exceeding 5MB must be rejected."""
    oversized = "A" * (MAX_IMAGE_B64_BYTES + 1024)
    img, burst = extract_vision_payloads({"image_b64": oversized})
    assert img is None

    oversized_bytes = b"\xff\xd8\xff" + b"\x00" * (5 * 1024 * 1024 + 1)
    img, burst = extract_vision_payloads({"image": oversized_bytes})
    assert img is None


def test_vision_payload_validation_burst_types():
    """burst_images_b64 must be a list of valid base64 strings."""
    # Non-list rejected
    assert extract_vision_payloads({"burst_images_b64": "single-string"}) == (None, [])
    assert extract_vision_payloads({"burst_images_b64": 12345}) == (None, [])

    # Filter out non-base64 elements
    frame_a = b"\xff\xd8\xff\xe0" + b"frame-a"
    frame_b = b"\xff\xd8\xff\xe0" + b"frame-b"
    valid1 = base64.b64encode(frame_a).decode("ascii")
    valid2 = base64.b64encode(frame_b).decode("ascii")
    img, burst = extract_vision_payloads({
        "burst_images_b64": [valid1, "invalid-text", valid2, None, 999]
    })
    assert img is None
    assert burst == [frame_a, frame_b]


def test_dual_env_flag_support(monkeypatch):
    """Both MVG_ENABLE_VISION/MVG_VISION_ENABLED and MVG_ENABLE_ADPCM/MVG_ADPCM_ENABLED must work."""
    # Test MVG_VISION_ENABLED
    monkeypatch.delenv("MVG_ENABLE_VISION", raising=False)
    monkeypatch.setenv("MVG_VISION_ENABLED", "1")
    assert gateway.is_vision_enabled() is True

    # Test MVG_ADPCM_ENABLED
    monkeypatch.delenv("MVG_ENABLE_ADPCM", raising=False)
    monkeypatch.setenv("MVG_ADPCM_ENABLED", "1")
    assert gateway.is_adpcm_enabled() is True

    # Test MVG_VISION_ADPCM alias
    monkeypatch.delenv("MVG_ENABLE_VISION", raising=False)
    monkeypatch.delenv("MVG_VISION_ENABLED", raising=False)
    monkeypatch.delenv("MVG_ENABLE_ADPCM", raising=False)
    monkeypatch.delenv("MVG_ADPCM_ENABLED", raising=False)
    monkeypatch.setenv("MVG_VISION_ADPCM", "1")
    assert gateway.is_vision_enabled() is True
    assert gateway.is_adpcm_enabled() is True


def _build_multipart_payload(parts: dict[str, tuple[str, bytes] | bytes | str]):
    boundary = "----forge-test-" + "12345678"
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


def test_voice_turn_multipart_b64_validation_errors(server, monkeypatch):
    """Multipart requests with invalid base64 or non-image image_b64/burst_images_b64 return 400."""
    monkeypatch.setenv("MVG_ENABLE_VISION", "1")
    monkeypatch.setattr(gateway, "ENABLE_VISION", True)
    monkeypatch.setattr(gateway, "speech_gate", lambda *args, **kw: ("passed", 900.0, 1.0))
    monkeypatch.setattr(gateway, "transcribe", lambda wav_bytes: "prompt")
    monkeypatch.setattr(gateway, "identify_speaker", lambda *args, **kw: ("", 0.0))
    monkeypatch.setattr(gateway, "decide_addressed", lambda *args, **kw: {"score": 0.9, "classifier": "jev", "laya_score": None})
    monkeypatch.setattr(gateway, "STREAM_BRAIN_CALL", lambda *a, **k: "ok")

    wav_16k = struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36, b"WAVE", b"fmt ", 16, 1, 1, 16000, 32000, 2, 16, b"data", 0)

    # 1. Invalid base64 in image_b64
    body, ctype = _build_multipart_payload({
        "node_id": "test-node",
        "audio": ("audio.wav", wav_16k),
        "image_b64": "invalid_base64_???",
        "tts": "none",
    })
    conn = http.client.HTTPConnection("127.0.0.1", server, timeout=5)
    conn.request("POST", "/voice/turn", body=body, headers={"Content-Type": ctype})
    resp = conn.getresponse()
    assert resp.status == 400
    conn.close()

    # 2. Valid base64 but not a valid image format (not JPEG, PNG, WEBP)
    non_img_b64 = base64.b64encode(b"this is plain text, not image bytes").decode("ascii")
    body, ctype = _build_multipart_payload({
        "node_id": "test-node",
        "audio": ("audio.wav", wav_16k),
        "image_b64": non_img_b64,
        "tts": "none",
    })
    conn = http.client.HTTPConnection("127.0.0.1", server, timeout=5)
    conn.request("POST", "/voice/turn", body=body, headers={"Content-Type": ctype})
    resp = conn.getresponse()
    assert resp.status == 400
    conn.close()

    # 3. Invalid base64 in burst_images_b64 JSON array
    body, ctype = _build_multipart_payload({
        "node_id": "test-node",
        "audio": ("audio.wav", wav_16k),
        "burst_images_b64": json.dumps(["invalid_base64_!"]),
        "tts": "none",
    })
    conn = http.client.HTTPConnection("127.0.0.1", server, timeout=5)
    conn.request("POST", "/voice/turn", body=body, headers={"Content-Type": ctype})
    resp = conn.getresponse()
    assert resp.status == 400
    conn.close()

    # 4. Valid base64 in burst_images_b64, but not an image
    body, ctype = _build_multipart_payload({
        "node_id": "test-node",
        "audio": ("audio.wav", wav_16k),
        "burst_images_b64": json.dumps([non_img_b64]),
        "tts": "none",
    })
    conn = http.client.HTTPConnection("127.0.0.1", server, timeout=5)
    conn.request("POST", "/voice/turn", body=body, headers={"Content-Type": ctype})
    resp = conn.getresponse()
    assert resp.status == 400
    conn.close()

    # 5. Valid base64 JPEG succeeds
    valid_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF-TEST-IMAGE"
    valid_jpeg_b64 = base64.b64encode(valid_jpeg).decode("ascii")
    body, ctype = _build_multipart_payload({
        "node_id": "test-node",
        "audio": ("audio.wav", wav_16k),
        "image_b64": valid_jpeg_b64,
        "tts": "none",
    })
    conn = http.client.HTTPConnection("127.0.0.1", server, timeout=5)
    conn.request("POST", "/voice/turn", body=body, headers={"Content-Type": ctype})
    resp = conn.getresponse()
    assert resp.status == 200
    conn.close()



_BAD_IMG_B64 = base64.b64encode(b"not an image at all").decode("ascii")
_OK_JPEG_B64 = base64.b64encode(b"\xff\xd8\xff\xe0\x00\x10JFIF-TEST-IMAGE").decode("ascii")


def _post_json(port, path, obj):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("POST", path, body=json.dumps(obj), headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    status, data = resp.status, resp.read()
    conn.close()
    return status, data


@pytest.mark.parametrize("path", ["/ask", "/ask/stream"])
@pytest.mark.parametrize(
    "extra",
    [
        {"image_b64": _BAD_IMG_B64},
        {"burst_images_b64": [_OK_JPEG_B64, _BAD_IMG_B64]},
    ],
)
def test_ask_decoded_b64_image_must_be_image(server, monkeypatch, path, extra):
    """With MVG_VISION_ADPCM on, decoded image_b64/burst_images_b64 get the same magic-byte check as image/burst_images."""
    monkeypatch.setenv("MVG_VISION_ADPCM", "1")
    monkeypatch.setattr(gateway, "BRAIN_CALL", lambda *a, **k: "ok")
    monkeypatch.setattr(gateway, "STREAM_BRAIN_CALL", lambda *a, **k: "ok")
    status, _ = _post_json(server, path, {"prompt": "hi", **extra})
    assert status == 400
    good = {"image_b64": _OK_JPEG_B64, "burst_images_b64": [_OK_JPEG_B64]}
    status, _ = _post_json(server, path, {"prompt": "hi", **good})
    assert status == 200


@pytest.mark.parametrize("path", ["/ask", "/ask/stream"])
def test_ask_bad_b64_image_ignored_when_vision_off(server, monkeypatch, path):
    """Flag off: a non-image image_b64 is not rejected (behaviour as on main)."""
    for k in ("MVG_ENABLE_VISION", "MVG_VISION_ENABLED", "MVG_VISION_ADPCM"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(gateway, "ENABLE_VISION", False)
    monkeypatch.setattr(gateway, "BRAIN_CALL", lambda *a, **k: "ok")
    monkeypatch.setattr(gateway, "STREAM_BRAIN_CALL", lambda *a, **k: "ok")
    status, _ = _post_json(server, path, {"prompt": "hi", "image_b64": _BAD_IMG_B64})
    assert status == 200


def test_voice_turn_query_string_bad_b64_image_400(server, monkeypatch):
    """On /voice/turn, query-string image_b64/burst_images_b64 holding non-images return 400 (not only multipart)."""
    from urllib.parse import quote

    monkeypatch.setenv("MVG_VISION_ADPCM", "1")
    monkeypatch.setattr(gateway, "speech_gate", lambda *args, **kw: ("passed", 900.0, 1.0))
    monkeypatch.setattr(gateway, "transcribe", lambda wav_bytes: "prompt")
    monkeypatch.setattr(gateway, "identify_speaker", lambda *args, **kw: ("", 0.0))
    monkeypatch.setattr(gateway, "decide_addressed", lambda *args, **kw: {"score": 0.9, "classifier": "jev", "laya_score": None})
    monkeypatch.setattr(gateway, "STREAM_BRAIN_CALL", lambda *a, **k: "ok")
    wav_16k = struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36, b"WAVE", b"fmt ", 16, 1, 1, 16000, 32000, 2, 16, b"data", 0)

    def post(query):
        conn = http.client.HTTPConnection("127.0.0.1", server, timeout=5)
        conn.request("POST", f"/voice/turn?node_id=test-node&tts=none&{query}", body=wav_16k,
                     headers={"Content-Type": "audio/wav"})
        resp = conn.getresponse()
        resp.read()
        conn.close()
        return resp.status

    assert post("image_b64=" + quote(_BAD_IMG_B64)) == 400
    assert post("burst_images_b64=" + quote(json.dumps([_OK_JPEG_B64, _BAD_IMG_B64]))) == 400
    assert post("image_b64=" + quote(_OK_JPEG_B64)) == 200
