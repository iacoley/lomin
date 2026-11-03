"""Multimodal camera & BLE IMA ADPCM ingest for Lomin Agent Voice Gateway.

Stdlib only (struct, base64, array, os).
Gated behind MVG_ENABLE_VISION (or MVG_VISION_ENABLED / MVG_VISION_ADPCM)
and MVG_ENABLE_ADPCM (or MVG_ADPCM_ENABLED / MVG_VISION_ADPCM).
"""

from __future__ import annotations

import array
import base64
import struct
import sys
from typing import Optional, Tuple

ADPCM_MAGIC = b"APCM"
ADPCM_HEADER_SIZE = 16  # <4sIIhBB: magic (4s), sample_rate (u32), num_samples (u32), predictor (i16), step_index (u8), reserved (u8)
MAX_ADPCM_BYTES = 512 * 1024  # 512 KiB strict upload cap
MAX_IMAGE_B64_BYTES = 5 * 1024 * 1024  # 5 MiB base64 payload cap
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MiB decoded image bytes cap

IMA_INDEX_TABLE: Tuple[int, ...] = (
    -1, -1, -1, -1, 2, 4, 6, 8,
    -1, -1, -1, -1, 2, 4, 6, 8,
)

IMA_STEP_TABLE: Tuple[int, ...] = (
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17,
    19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
    50, 55, 60, 66, 73, 80, 88, 97, 107, 118,
    130, 143, 157, 173, 190, 209, 230, 253, 279, 307,
    337, 371, 408, 449, 494, 544, 598, 658, 724, 796,
    876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066,
    2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358,
    5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899,
    15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767,
)


def _encode_ima_sample(sample: int, predictor: int, step_index: int) -> Tuple[int, int, int]:
    step = IMA_STEP_TABLE[step_index]
    diff = sample - predictor
    nibble = 0
    if diff < 0:
        nibble = 8
        diff = -diff
    if diff >= step:
        nibble |= 4
        diff -= step
    if diff >= (step >> 1):
        nibble |= 2
        diff -= (step >> 1)
    if diff >= (step >> 2):
        nibble |= 1

    vpdiff = step >> 3
    if nibble & 4:
        vpdiff += step
    if nibble & 2:
        vpdiff += step >> 1
    if nibble & 1:
        vpdiff += step >> 2

    if nibble & 8:
        predictor -= vpdiff
    else:
        predictor += vpdiff

    if predictor > 32767:
        predictor = 32767
    elif predictor < -32768:
        predictor = -32768

    step_index += IMA_INDEX_TABLE[nibble]
    if step_index < 0:
        step_index = 0
    elif step_index > 88:
        step_index = 88

    return nibble & 0x0F, predictor, step_index


def _decode_ima_nibble(nibble: int, predictor: int, step_index: int) -> Tuple[int, int]:
    step = IMA_STEP_TABLE[step_index]
    vpdiff = step >> 3
    if nibble & 4:
        vpdiff += step
    if nibble & 2:
        vpdiff += step >> 1
    if nibble & 1:
        vpdiff += step >> 2

    if nibble & 8:
        predictor -= vpdiff
    else:
        predictor += vpdiff

    if predictor > 32767:
        predictor = 32767
    elif predictor < -32768:
        predictor = -32768

    step_index += IMA_INDEX_TABLE[nibble & 0x0F]
    if step_index < 0:
        step_index = 0
    elif step_index > 88:
        step_index = 88

    return predictor, step_index


def encode_ima_adpcm_with_state(
    pcm_bytes: bytes,
    initial_predictor: Optional[int] = None,
    initial_step_index: int = 0,
) -> Tuple[bytes, int, int, int, int]:
    """Compress 16-bit signed little-endian mono PCM into 4-bit IMA ADPCM nibbles."""
    num_samples = len(pcm_bytes) // 2
    if num_samples <= 0:
        init_p = int(initial_predictor or 0)
        init_idx = max(0, min(88, int(initial_step_index)))
        return b"", init_p, init_idx, init_p, init_idx

    samples = struct.unpack(f"<{num_samples}h", pcm_bytes[: num_samples * 2])
    init_p = int(samples[0] if initial_predictor is None else initial_predictor)
    init_p = max(-32768, min(32767, init_p))
    init_idx = max(0, min(88, int(initial_step_index)))

    out = bytearray((num_samples + 1) // 2)
    predictor = init_p
    step_index = init_idx

    for i in range(0, num_samples, 2):
        nib_lo, predictor, step_index = _encode_ima_sample(samples[i], predictor, step_index)
        if i + 1 < num_samples:
            nib_hi, predictor, step_index = _encode_ima_sample(samples[i + 1], predictor, step_index)
        else:
            nib_hi = 0
        out[i // 2] = (nib_lo & 0x0F) | ((nib_hi & 0x0F) << 4)

    return bytes(out), init_p, init_idx, predictor, step_index


def encode_ima_adpcm(
    pcm_bytes: bytes,
    initial_predictor: int = 0,
    initial_step_index: int = 0,
) -> bytes:
    """Compress 16-bit signed mono PCM into 4-bit IMA ADPCM nibbles (4:1 compression)."""
    adpcm_bytes, _, _, _, _ = encode_ima_adpcm_with_state(
        pcm_bytes,
        initial_predictor=initial_predictor,
        initial_step_index=initial_step_index,
    )
    return adpcm_bytes


def decode_ima_adpcm(
    adpcm_bytes: bytes,
    num_samples: Optional[int] = None,
    initial_predictor: int = 0,
    initial_step_index: int = 0,
) -> bytes:
    """Decompress 4-bit IMA ADPCM nibbles back into 16-bit signed little-endian mono PCM.

    Uses array.array('h') to avoid allocating millions of boxed Python int objects.
    """
    if not adpcm_bytes:
        return b""
    if len(adpcm_bytes) > MAX_ADPCM_BYTES:
        raise ValueError(
            f"IMA ADPCM payload size {len(adpcm_bytes)} exceeds MAX_ADPCM_BYTES ({MAX_ADPCM_BYTES})"
        )

    total_nibbles = len(adpcm_bytes) * 2
    target_samples = total_nibbles if num_samples is None else min(max(0, int(num_samples)), total_nibbles)
    if target_samples <= 0:
        return b""

    out = array.array("h")
    predictor = max(-32768, min(32767, int(initial_predictor)))
    step_index = max(0, min(88, int(initial_step_index)))

    for byte_val in adpcm_bytes:
        nib_lo = byte_val & 0x0F
        predictor, step_index = _decode_ima_nibble(nib_lo, predictor, step_index)
        out.append(predictor)
        if len(out) >= target_samples:
            break

        nib_hi = (byte_val >> 4) & 0x0F
        predictor, step_index = _decode_ima_nibble(nib_hi, predictor, step_index)
        out.append(predictor)
        if len(out) >= target_samples:
            break

    if sys.byteorder != "little":
        out.byteswap()
    return out.tobytes()


def _wrap_pcm_in_wav(pcm_bytes: bytes, sample_rate: int = 16000) -> bytes:
    """Wrap 16-bit mono PCM in a standard 44-byte RIFF/WAVE header."""
    riff_size = 36 + len(pcm_bytes)
    byte_rate = sample_rate * 2
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        riff_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        sample_rate,
        byte_rate,
        2,
        16,
        b"data",
        len(pcm_bytes),
    )
    return header + pcm_bytes


def _extract_pcm_from_wav(wav_or_pcm: bytes, default_sample_rate: int = 16000) -> Tuple[bytes, int]:
    """Extract raw PCM bytes and sample_rate from WAV if header present, else return as-is."""
    if len(wav_or_pcm) >= 44 and wav_or_pcm[:4] == b"RIFF" and wav_or_pcm[8:12] == b"WAVE":
        sample_rate = default_sample_rate
        try:
            parsed_sr = struct.unpack_from("<I", wav_or_pcm, 24)[0]
            if 4000 <= parsed_sr <= 96000:
                sample_rate = parsed_sr
        except struct.error:
            pass
        offset = 12
        total_len = len(wav_or_pcm)
        while offset + 8 <= total_len:
            chunk_id = wav_or_pcm[offset: offset + 4]
            chunk_size = struct.unpack_from("<I", wav_or_pcm, offset + 4)[0]
            offset += 8
            if chunk_id == b"data":
                end = min(total_len, offset + chunk_size) if chunk_size < 0xFFFFFFFF else total_len
                return wav_or_pcm[offset:end], sample_rate
            offset += chunk_size + (chunk_size % 2)
        return wav_or_pcm[44:], sample_rate
    return wav_or_pcm, default_sample_rate


def encode_adpcm(
    wav_or_pcm: bytes,
    sample_rate: int = 16000,
    initial_predictor: Optional[int] = None,
    initial_step_index: int = 0,
) -> bytes:
    """Encode 16-bit mono WAV or raw PCM into an APCM 4-bit IMA ADPCM container."""
    pcm_bytes, sr = _extract_pcm_from_wav(wav_or_pcm, default_sample_rate=sample_rate)
    num_samples = len(pcm_bytes) // 2
    adpcm_payload, init_pred, init_idx, _, _ = encode_ima_adpcm_with_state(
        pcm_bytes,
        initial_predictor=initial_predictor,
        initial_step_index=initial_step_index,
    )
    header = struct.pack(
        "<4sIIhBB",
        ADPCM_MAGIC,
        int(sr),
        int(num_samples),
        int(init_pred),
        int(init_idx) & 0xFF,
        0,
    )
    return header + adpcm_payload


encode_wav_to_adpcm = encode_adpcm


def decode_adpcm_to_wav(data: bytes, sample_rate: int = 16000) -> bytes:
    """Decode an APCM 4-bit IMA ADPCM container into a 16-bit mono PCM RIFF WAV buffer.

    If data does not start with ADPCM_MAGIC (b'APCM'), returns data unchanged.
    Enforces MAX_ADPCM_BYTES upload cap, raising ValueError if exceeded.
    """
    if not data or len(data) < 4 or data[:4] != ADPCM_MAGIC:
        return data

    if len(data) > MAX_ADPCM_BYTES:
        raise ValueError(
            f"ADPCM data size {len(data)} exceeds maximum allowed {MAX_ADPCM_BYTES} bytes"
        )

    sr = sample_rate
    num_samples: Optional[int] = None
    init_pred = 0
    init_idx = 0
    raw_adpcm = b""

    if len(data) >= ADPCM_HEADER_SIZE:
        try:
            _, parsed_sr, parsed_num_samples, p, idx, _ = struct.unpack("<4sIIhBB", data[:ADPCM_HEADER_SIZE])
            if 4000 <= parsed_sr <= 96000 and 0 <= idx <= 88:
                sr = parsed_sr
                num_samples = parsed_num_samples
                init_pred = p
                init_idx = idx
                raw_adpcm = data[ADPCM_HEADER_SIZE:]
        except struct.error:
            pass

    if not raw_adpcm and len(data) >= 12:
        try:
            _, parsed_sr, p, idx = struct.unpack("<4sIhh", data[:12])
            if 4000 <= parsed_sr <= 96000 and 0 <= idx <= 88:
                sr = parsed_sr
                init_pred = p
                init_idx = idx
                raw_adpcm = data[12:]
        except struct.error:
            pass

    if not raw_adpcm:
        offset = ADPCM_HEADER_SIZE if len(data) >= ADPCM_HEADER_SIZE else (12 if len(data) >= 12 else 4)
        raw_adpcm = data[offset:]

    pcm_bytes = decode_ima_adpcm(
        raw_adpcm,
        num_samples=num_samples,
        initial_predictor=init_pred,
        initial_step_index=init_idx,
    )
    return _wrap_pcm_in_wav(pcm_bytes, sample_rate=sr)



def is_valid_image_bytes(data: bytes) -> bool:
    """Validate image magic bytes for JPEG, PNG, or WEBP."""
    if not isinstance(data, (bytes, bytearray)) or len(data) < 3:
        return False
    if data[:3] == b"\xff\xd8\xff":
        return True
    if len(data) >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return True
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return True
    return False


def extract_vision_payloads(parts_or_json: dict) -> Tuple[Optional[bytes], list[bytes]]:
    """Extract optional image and burst images from a multipart parts dict or JSON payload.

    Returns (image_bytes, burst_images).
    image_bytes is bytes or None.
    burst_images is a list of bytes (up to 5 frames).

    Validates base64 strings and enforces MAX_IMAGE_B64_BYTES / MAX_IMAGE_BYTES.
    Non-base64 text and invalid types are strictly rejected (return None / skipped).
    """
    if not isinstance(parts_or_json, dict):
        return None, []

    def _to_bytes(val, is_b64_hint: bool = False) -> Optional[bytes]:
        if val is None:
            return None
        if isinstance(val, (bytes, bytearray)):
            b = bytes(val)
            if not b:
                return None
            if is_b64_hint:
                if len(b) > MAX_IMAGE_B64_BYTES:
                    return None
                try:
                    decoded = base64.b64decode(b, validate=True)
                    if decoded and len(decoded) <= MAX_IMAGE_BYTES and is_valid_image_bytes(decoded):
                        return decoded
                except Exception:
                    return None
                return None
            if len(b) > MAX_IMAGE_BYTES or not is_valid_image_bytes(b):
                return None
            return b
        if isinstance(val, str):
            s = val.strip()
            if not s or len(s) > MAX_IMAGE_B64_BYTES:
                return None
            try:
                decoded = base64.b64decode(s, validate=True)
                if decoded and len(decoded) <= MAX_IMAGE_BYTES and is_valid_image_bytes(decoded):
                    return decoded
            except Exception:
                return None
            return None
        return None

    # Single image extraction: check 'image', then 'image_b64'
    image_bytes: Optional[bytes] = None
    if "image" in parts_or_json and parts_or_json["image"]:
        image_bytes = _to_bytes(parts_or_json["image"])
    elif "image_b64" in parts_or_json and parts_or_json["image_b64"]:
        image_bytes = _to_bytes(parts_or_json["image_b64"], is_b64_hint=True)

    # Burst images extraction
    burst_images: list[bytes] = []
    # 1. Discrete numbered parts image_0 .. image_4
    for i in range(5):
        k = f"image_{i}"
        if k in parts_or_json and parts_or_json[k]:
            b = _to_bytes(parts_or_json[k])
            if b:
                burst_images.append(b)

    # 2. burst_images_b64 list
    if not burst_images and "burst_images_b64" in parts_or_json:
        raw_list = parts_or_json["burst_images_b64"]
        if isinstance(raw_list, list):
            for item in raw_list[:5]:
                if isinstance(item, (str, bytes, bytearray)):
                    b = _to_bytes(item, is_b64_hint=True)
                    if b:
                        burst_images.append(b)

    # 3. burst_images list
    if not burst_images and "burst_images" in parts_or_json:
        raw_list = parts_or_json["burst_images"]
        if isinstance(raw_list, list):
            for item in raw_list[:5]:
                if isinstance(item, (str, bytes, bytearray)):
                    b = _to_bytes(item)
                    if b:
                        burst_images.append(b)

    return image_bytes, burst_images
