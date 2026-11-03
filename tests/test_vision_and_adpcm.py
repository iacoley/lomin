#!/usr/bin/env python3
"""Tests for high-performance ADPCM codec and multimodal vision payloads.

Covers:
- adpcm_codec: optimized IMA ADPCM encode/decode CPU performance (<100-250ms for 512KB cap vs 2.6s baseline)
- adpcm_codec: bit-for-bit fidelity, boundary conditions, odd sample counts, container wrapping
- vision_adpcm: camera and burst image payload extraction and validation
"""

from __future__ import annotations

import base64
import math
import struct
import time

import pytest

from adpcm_codec import (
    ADPCM_HEADER_SIZE,
    ADPCM_MAGIC,
    MAX_ADPCM_BYTES,
    decode_adpcm_to_wav,
    decode_ima_adpcm,
    encode_adpcm,
    encode_ima_adpcm,
    encode_ima_adpcm_with_state,
)
from vision_adpcm import (
    MAX_IMAGE_B64_BYTES,
    extract_vision_payloads,
)
import vision_adpcm

# Standard test WAV (16-bit mono 16 kHz, 0.1s sine wave)
SAMPLE_RATE = 16000
NUM_SAMPLES = 1600
SAMPLES = [int(16000 * math.sin(2 * math.pi * 440 * i / SAMPLE_RATE)) for i in range(NUM_SAMPLES)]
PCM_16K = struct.pack(f"<{NUM_SAMPLES}h", *SAMPLES)


# ============================================================================
# 1. ADPCM Constants & IMA Codec Tests
# ============================================================================

def test_adpcm_constants():
    assert ADPCM_MAGIC == b"APCM"
    assert ADPCM_HEADER_SIZE == 16
    assert MAX_ADPCM_BYTES == 512 * 1024


def test_decode_adpcm_passthrough_on_non_adpcm():
    assert decode_adpcm_to_wav(b"") == b""
    assert decode_adpcm_to_wav(b"RIF") == b"RIF"
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
    adpcm_container = encode_adpcm(PCM_16K, sample_rate=16000)
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


def test_decode_adpcm_size_cap():
    oversized = ADPCM_MAGIC + b"\x00" * (MAX_ADPCM_BYTES + 1)
    with pytest.raises(ValueError, match="exceeds"):
        decode_adpcm_to_wav(oversized)

    with pytest.raises(ValueError, match="exceeds"):
        decode_ima_adpcm(b"\x00" * (MAX_ADPCM_BYTES + 1))


def test_adpcm_codec_parity_with_reference():
    """Verify bit-for-bit parity between adpcm_codec and vision_adpcm reference."""
    # Test varying byte sizes including edge cases
    for n_bytes in [0, 1, 2, 7, 8, 9, 15, 16, 17, 33, 100, 500]:
        raw = bytes([((i * 37 + 13) & 0xFF) for i in range(n_bytes)])
        for num_samples in [None, 0, 1, 2, n_bytes, n_bytes * 2 - 1, n_bytes * 2]:
            ref = vision_adpcm.decode_ima_adpcm(raw, num_samples=num_samples)
            opt = decode_ima_adpcm(raw, num_samples=num_samples)
            assert ref == opt, f"Decode mismatch on n_bytes={n_bytes}, num_samples={num_samples}"

    # Verify encode parity
    for n_samples in [0, 1, 2, 7, 8, 9, 15, 16, 17, 33, 100, 500]:
        pcm = bytes([((i * 47 + 29) & 0xFF) for i in range(n_samples * 2)])
        for init_p in [None, 0, 100, -500]:
            for init_idx in [0, 5, 88]:
                ref = vision_adpcm.encode_ima_adpcm_with_state(
                    pcm, initial_predictor=init_p, initial_step_index=init_idx
                )
                opt = encode_ima_adpcm_with_state(
                    pcm, initial_predictor=init_p, initial_step_index=init_idx
                )
                assert ref == opt, f"Encode mismatch on n_samples={n_samples}"


# ============================================================================
# 2. Performance & Timing Regression Test
# ============================================================================

def test_adpcm_decoder_performance_max_capped_payload():
    """Performance regression test: decoding a max-capped ADPCM upload (512 KiB)
    takes <250ms (locally ~100-140ms), proving a ~10x-20x speedup over the 2.6s baseline.
    """
    # Create max-capped payload (512 KiB)
    payload = bytes([i % 256 for i in range(MAX_ADPCM_BYTES)])

    # Warm up JIT/bytecode caches
    _ = decode_ima_adpcm(payload[:1024])

    t0 = time.perf_counter()
    decoded = decode_ima_adpcm(payload)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert len(decoded) == MAX_ADPCM_BYTES * 4
    # Verify strict performance budget: well under 250ms (locally ~100-140ms, vs 2600ms baseline)
    assert elapsed_ms < 250.0, (
        f"Optimized decode_ima_adpcm took {elapsed_ms:.1f}ms for 512KB payload; "
        f"expected < 250ms (unoptimized baseline is 2600ms)"
    )


def test_adpcm_decoder_performance_typical_voice_turn():
    """Verify that a typical 8-second voice turn upload (~64 KiB ADPCM) decodes in <35ms."""
    typical_payload = bytes([i % 256 for i in range(64 * 1024)])

    t0 = time.perf_counter()
    decoded = decode_ima_adpcm(typical_payload)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert len(decoded) == 64 * 1024 * 4
    assert elapsed_ms < 35.0, (
        f"Typical voice turn decode took {elapsed_ms:.1f}ms; expected < 35ms"
    )


# ============================================================================
# 3. Vision Payload Extraction Tests
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
    f0 = b"\xff\xd8\xff\xe0" + b"frame-0-bytes"
    f1 = b"\xff\xd8\xff\xe0" + b"frame-1-bytes"
    f2 = b"\xff\xd8\xff\xe0" + b"frame-2-bytes"

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
    main_img = b"\xff\xd8\xff\xe0" + b"main-camera-frame"
    burst0 = b"\xff\xd8\xff\xe0" + b"burst-frame-0"
    burst1 = b"\xff\xd8\xff\xe0" + b"burst-frame-1"

    img, burst = extract_vision_payloads({"image": main_img, "image_0": burst0, "image_1": burst1})
    assert img == main_img
    assert burst == [burst0, burst1]


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
    frame1 = b"\xff\xd8\xff\xe0" + b"frame1"
    valid_b64 = base64.b64encode(frame1).decode("ascii")
    img, burst = extract_vision_payloads({"burst_images_b64": [valid_b64, "invalid_b64!", 12345]})
    assert img is None
    assert burst == [frame1]
