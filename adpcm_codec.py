"""High-performance Intel/DVI IMA ADPCM Audio Codec.

Optimized with local table bindings, bytearray/struct.pack_into buffers,
and precomputed 2-nibble lookup tables for low-latency voice streaming.
"""

from __future__ import annotations

import struct
from typing import Optional, Tuple

ADPCM_MAGIC = b"APCM"
ADPCM_HEADER_SIZE = 16  # <4sIIhBB: magic (4s), sample_rate (u32), num_samples (u32), predictor (i16), step_index (u8), reserved (u8)
MAX_ADPCM_BYTES = 512 * 1024  # 512 KiB strict upload cap

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

# Precompute single-nibble diff and next-step tables
_DIFF_TABLE: list[tuple[int, ...]] = []
_NEXT_STEP_TABLE: list[tuple[int, ...]] = []

for _s_idx in range(89):
    _step = IMA_STEP_TABLE[_s_idx]
    _d_row: list[int] = []
    _ns_row: list[int] = []
    for _nib in range(16):
        _vpdiff = _step >> 3
        if _nib & 4:
            _vpdiff += _step
        if _nib & 2:
            _vpdiff += _step >> 1
        if _nib & 1:
            _vpdiff += _step >> 2
        _d_row.append(-_vpdiff if (_nib & 8) else _vpdiff)
        _next_s = _s_idx + IMA_INDEX_TABLE[_nib]
        if _next_s < 0:
            _next_s = 0
        elif _next_s > 88:
            _next_s = 88
        _ns_row.append(_next_s)
    _DIFF_TABLE.append(tuple(_d_row))
    _NEXT_STEP_TABLE.append(tuple(_ns_row))

DIFF_TABLE = tuple(_DIFF_TABLE)
NEXT_STEP_TABLE = tuple(_NEXT_STEP_TABLE)

# Precompute 2-nibble (byte-level) lookup table: TABLE_COMBO[step_idx][byte_val] -> (delta1, delta2, next_step_idx)
_BYTE_DIFF1: list[list[int]] = []
_BYTE_DIFF2: list[list[int]] = []
_BYTE_NEXT_STEP: list[list[int]] = []

for _s_idx in range(89):
    _b_d1: list[int] = []
    _b_d2: list[int] = []
    _b_ns: list[int] = []
    for _b in range(256):
        _nib_lo = _b & 0x0F
        _nib_hi = (_b >> 4) & 0x0F
        _d1 = DIFF_TABLE[_s_idx][_nib_lo]
        _s1 = NEXT_STEP_TABLE[_s_idx][_nib_lo]
        _d2 = DIFF_TABLE[_s1][_nib_hi]
        _s2 = NEXT_STEP_TABLE[_s1][_nib_hi]
        _b_d1.append(_d1)
        _b_d2.append(_d2)
        _b_ns.append(_s2)
    _BYTE_DIFF1.append(_b_d1)
    _BYTE_DIFF2.append(_b_d2)
    _BYTE_NEXT_STEP.append(_b_ns)

TABLE_COMBO: list[list[tuple[int, int, int]]] = [
    [
        (_BYTE_DIFF1[s][b], _BYTE_DIFF2[s][b], _BYTE_NEXT_STEP[s][b])
        for b in range(256)
    ]
    for s in range(89)
]

_PACK16 = struct.Struct("<16h").pack_into
_PACK1 = struct.Struct("<h").pack_into


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

    samples = memoryview(pcm_bytes).cast("h")
    init_p = int(samples[0] if initial_predictor is None else initial_predictor)
    init_p = max(-32768, min(32767, init_p))
    init_idx = max(0, min(88, int(initial_step_index)))

    out = bytearray((num_samples + 1) // 2)
    predictor = init_p
    step_index = init_idx

    step_table = IMA_STEP_TABLE
    diff_table = DIFF_TABLE
    next_step_table = NEXT_STEP_TABLE

    for i in range(0, num_samples, 2):
        s0 = samples[i]
        step = step_table[step_index]
        diff = s0 - predictor
        nib_lo = 0
        if diff < 0:
            nib_lo = 8
            diff = -diff
        if diff >= step:
            nib_lo |= 4
            diff -= step
        if diff >= (step >> 1):
            nib_lo |= 2
            diff -= (step >> 1)
        if diff >= (step >> 2):
            nib_lo |= 1

        d0 = diff_table[step_index][nib_lo]
        step_index = next_step_table[step_index][nib_lo]
        predictor += d0
        if predictor > 32767:
            predictor = 32767
        elif predictor < -32768:
            predictor = -32768

        if i + 1 < num_samples:
            s1 = samples[i + 1]
            step = step_table[step_index]
            diff = s1 - predictor
            nib_hi = 0
            if diff < 0:
                nib_hi = 8
                diff = -diff
            if diff >= step:
                nib_hi |= 4
                diff -= step
            if diff >= (step >> 1):
                nib_hi |= 2
                diff -= (step >> 1)
            if diff >= (step >> 2):
                nib_hi |= 1

            d1 = diff_table[step_index][nib_hi]
            step_index = next_step_table[step_index][nib_hi]
            predictor += d1
            if predictor > 32767:
                predictor = 32767
            elif predictor < -32768:
                predictor = -32768
        else:
            nib_hi = 0

        out[i // 2] = nib_lo | (nib_hi << 4)

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

    Optimized using precomputed 2-nibble lookup tables, local bindings, and
    struct.pack_into batches for high CPU throughput.
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

    buf = bytearray(target_samples * 2)
    packer = _PACK16
    pack_one = _PACK1
    tbl = TABLE_COMBO
    offset = 0

    p = max(-32768, min(32767, int(initial_predictor)))
    step_index = max(0, min(88, int(initial_step_index)))

    full_chunks = target_samples // 16
    if full_chunks > 0:
        chunk_bytes = adpcm_bytes[: full_chunks * 8]
        it = [iter(chunk_bytes)] * 8
        for b0, b1, b2, b3, b4, b5, b6, b7 in zip(*it):
            d1, d2, step_index = tbl[step_index][b0]
            p1 = p + d1
            if p1 > 32767:
                p1 = 32767
            elif p1 < -32768:
                p1 = -32768
            p2 = p1 + d2
            if p2 > 32767:
                p2 = 32767
            elif p2 < -32768:
                p2 = -32768

            d3, d4, step_index = tbl[step_index][b1]
            p3 = p2 + d3
            if p3 > 32767:
                p3 = 32767
            elif p3 < -32768:
                p3 = -32768
            p4 = p3 + d4
            if p4 > 32767:
                p4 = 32767
            elif p4 < -32768:
                p4 = -32768

            d5, d6, step_index = tbl[step_index][b2]
            p5 = p4 + d5
            if p5 > 32767:
                p5 = 32767
            elif p5 < -32768:
                p5 = -32768
            p6 = p5 + d6
            if p6 > 32767:
                p6 = 32767
            elif p6 < -32768:
                p6 = -32768

            d7, d8, step_index = tbl[step_index][b3]
            p7 = p6 + d7
            if p7 > 32767:
                p7 = 32767
            elif p7 < -32768:
                p7 = -32768
            p8 = p7 + d8
            if p8 > 32767:
                p8 = 32767
            elif p8 < -32768:
                p8 = -32768

            d9, d10, step_index = tbl[step_index][b4]
            p9 = p8 + d9
            if p9 > 32767:
                p9 = 32767
            elif p9 < -32768:
                p9 = -32768
            p10 = p9 + d10
            if p10 > 32767:
                p10 = 32767
            elif p10 < -32768:
                p10 = -32768

            d11, d12, step_index = tbl[step_index][b5]
            p11 = p10 + d11
            if p11 > 32767:
                p11 = 32767
            elif p11 < -32768:
                p11 = -32768
            p12 = p11 + d12
            if p12 > 32767:
                p12 = 32767
            elif p12 < -32768:
                p12 = -32768

            d13, d14, step_index = tbl[step_index][b6]
            p13 = p12 + d13
            if p13 > 32767:
                p13 = 32767
            elif p13 < -32768:
                p13 = -32768
            p14 = p13 + d14
            if p14 > 32767:
                p14 = 32767
            elif p14 < -32768:
                p14 = -32768

            d15, d16, step_index = tbl[step_index][b7]
            p15 = p14 + d15
            if p15 > 32767:
                p15 = 32767
            elif p15 < -32768:
                p15 = -32768
            p16 = p15 + d16
            if p16 > 32767:
                p16 = 32767
            elif p16 < -32768:
                p16 = -32768

            p = p16
            packer(buf, offset, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16)
            offset += 32

    rem_samples = target_samples - (full_chunks * 16)
    if rem_samples > 0:
        rem_bytes = adpcm_bytes[full_chunks * 8 :]
        for b in rem_bytes:
            d1, d2, step_index = tbl[step_index][b]
            p1 = p + d1
            if p1 > 32767:
                p1 = 32767
            elif p1 < -32768:
                p1 = -32768
            pack_one(buf, offset, p1)
            offset += 2
            rem_samples -= 1
            p = p1
            if rem_samples <= 0:
                break

            p2 = p1 + d2
            if p2 > 32767:
                p2 = 32767
            elif p2 < -32768:
                p2 = -32768
            pack_one(buf, offset, p2)
            offset += 2
            rem_samples -= 1
            p = p2
            if rem_samples <= 0:
                break

    return bytes(buf)


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
            if 4000 <= parsed_sr <= 96000:
                sr = parsed_sr
            if parsed_num_samples > 0:
                num_samples = parsed_num_samples
            init_pred = p
            init_idx = idx
            raw_adpcm = data[ADPCM_HEADER_SIZE:]
        except struct.error:
            raw_adpcm = data[4:]
    else:
        raw_adpcm = data[4:]

    pcm_bytes = decode_ima_adpcm(
        raw_adpcm,
        num_samples=num_samples,
        initial_predictor=init_pred,
        initial_step_index=init_idx,
    )
    return _wrap_pcm_in_wav(pcm_bytes, sample_rate=sr)
