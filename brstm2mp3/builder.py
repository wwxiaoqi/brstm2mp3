"""A minimal BRSTM *writer* (used by the test-suite, and handy for experiments).

It produces 16 bit PCM or DSP ADPCM streams in exactly the container layout the
reader expects, so the test-suite can do closed loop checks without shipping a
copyrighted .brstm file::

    raw = build_brstm(pcm_int16_array, sample_rate=44100)
    open("test.brstm", "wb").write(raw)
"""

from __future__ import annotations

import struct

import numpy as np

# a normal Nintendo coefficient table (8 predictor pairs, Q11)
DEFAULT_COEFFICIENTS = [
    (1, 2047),
    (2614, -1222),
    (1246, -1272),
    (3518, -1610),
    (659, -93),
    (2759, -938),
    (2027, -1286),
    (3575, -1552),
]

FRAME_SAMPLES = 14
FRAME_BYTES = 8


def _quantize_frame(samples, coefs, h1, h2):
    """Pick the best (predictor, scale) for one frame -> (frame bytes, h1, h2)."""
    best = None
    for predictor, (c1, c2) in enumerate(coefs):
        for scale in range(13):
            ph1, ph2 = h1, h2
            nibbles = []
            error = 0
            for sample in samples:
                predicted = (c1 * ph1 + c2 * ph2) >> 11
                nib = int(round((int(sample) - predicted) / (1 << scale)))
                nib = max(-8, min(7, nib))
                value = (((nib << scale) << 11) + c1 * ph1 + c2 * ph2) >> 11
                value = max(-32768, min(32767, value))
                error += (value - int(sample)) ** 2
                nibbles.append(nib & 0x0F)
                ph2, ph1 = ph1, value
            if best is None or error < best[0]:
                best = (error, predictor, scale, nibbles, ph1, ph2)

    _error, predictor, scale, nibbles, ph1, ph2 = best
    frame = bytearray([(predictor << 4) | scale])
    for i in range(0, FRAME_SAMPLES, 2):
        frame.append((nibbles[i] << 4) | nibbles[i + 1])
    return bytes(frame), ph1, ph2


def encode_dsp_channel(samples: np.ndarray, coefs) -> bytes:
    """Encode one int16 channel to DSP ADPCM (last frame zero padded)."""
    samples = np.asarray(samples, dtype=np.int32)
    pad = (-len(samples)) % FRAME_SAMPLES
    if pad:
        samples = np.concatenate([samples, np.zeros(pad, dtype=np.int32)])
    out = bytearray()
    h1 = h2 = 0
    for start in range(0, len(samples), FRAME_SAMPLES):
        frame, h1, h2 = _quantize_frame(samples[start : start + FRAME_SAMPLES], coefs, h1, h2)
        out += frame
    return bytes(out)


def build_brstm(
    pcm: np.ndarray,
    sample_rate: int = 44100,
    codec: str = "dsp",
    loop_start: int = 0,
    loop_flag: bool = False,
    block_samples: int = FRAME_SAMPLES * 1024,
    coefficients=None,
) -> bytes:
    """Serialise ``pcm`` (int16, shape (samples, channels)) as a BRSTM file."""
    pcm = np.asarray(pcm, dtype=np.int16)
    if pcm.ndim == 1:
        pcm = pcm.reshape(-1, 1)
    channels = pcm.shape[1]
    coefs_all = [list(coefficients or DEFAULT_COEFFICIENTS) for _ in range(channels)]
    original_samples = len(pcm)

    num_blocks = max(1, -(-len(pcm) // block_samples))
    total = num_blocks * block_samples
    if total != len(pcm):
        pcm = np.concatenate([pcm, np.zeros((total - len(pcm), channels), np.int16)])
    last_block_samples = len(pcm) - (num_blocks - 1) * block_samples

    if codec == "dsp":
        block_size = block_samples // FRAME_SAMPLES * FRAME_BYTES
        encoded = [encode_dsp_channel(pcm[: (num_blocks - 1) * block_samples, ch], coefs_all[ch])
                   for ch in range(channels)]
        tail = [encode_dsp_channel(pcm[(num_blocks - 1) * block_samples :, ch], coefs_all[ch])
                for ch in range(channels)]
        codec_id = 2
        last_used = (last_block_samples + FRAME_SAMPLES - 1) // FRAME_SAMPLES * FRAME_BYTES
    elif codec == "pcm16":
        block_size = block_samples * channels * 2
        flat = pcm.reshape(-1)
        raw = flat.astype(">i2").tobytes()
        encoded = [raw[: (num_blocks - 1) * block_size]]
        tail = [raw[(num_blocks - 1) * block_size :]]
        codec_id = 1
        last_used = last_block_samples * channels * 2
    else:
        raise ValueError("codec must be 'dsp' or 'pcm16'")

    padded_last = (last_used + 31) // 32 * 32

    body = bytearray()
    for block in range(num_blocks):
        for ch in range(channels):
            if block == num_blocks - 1:
                chunk = tail[ch][:last_used]
                chunk = chunk.ljust(padded_last, b"\x00")[:padded_last]
            else:
                chunk = encoded[ch][block * block_size : (block + 1) * block_size]
            body += chunk

    # ---------------- HEAD chunk (offsets are relative to head + 8) --------
    def rel(pos: int) -> int:
        return pos - 8

    head = bytearray(b"HEAD\x00\x00\x00\x00")
    head += b"\x00" * 24  # three references, patched below
    stream_pos = len(head)                       # 0x20
    head += bytes([codec_id, 1 if loop_flag else 0, channels, 0])
    head += struct.pack(">HH", sample_rate, 0)
    head += struct.pack(">I", loop_start)
    head += struct.pack(">I", original_samples)
    data_offset_pos = len(head)
    head += struct.pack(">I", 0)                 # patched after layout is known
    head += struct.pack(">I", num_blocks)
    head += struct.pack(">I", block_size)
    head += struct.pack(">I", block_samples)
    head += struct.pack(">I", last_used)
    head += struct.pack(">I", last_block_samples)
    head += struct.pack(">I", padded_last)
    head += struct.pack(">I", block_samples)
    head += struct.pack(">I", 4)
    head = head.ljust(stream_pos + 0x34, b"\x00")

    track_pos = len(head)
    head += struct.pack(">BBH", 1, 1 if loop_flag else 0, 0) + b"\x00" * 8
    head = head.ljust(track_pos + 0x18, b"\x00")

    chan_pos = len(head)
    head += struct.pack(">I", channels)
    struct_pos = chan_pos + 4 + 8 * channels
    for ch in range(channels):
        head += struct.pack(">II", 0x01000000, rel(struct_pos + 0x38 * ch))
    head = head.ljust(struct_pos, b"\x00")

    for ch in range(channels):
        head += struct.pack(">II", 0x01000000, rel(struct_pos + 0x38 * ch + 8))
        head += struct.pack(">16h", *[v for pair in coefs_all[ch] for v in pair])
        head += struct.pack(">hhhh", 0, 0, 0, 0)
        head += b"\x00" * 8

    head_size = (-(-len(head) // 0x20)) * 0x20
    head = head.ljust(head_size, b"\x00")

    def rel(pos: int) -> int:
        return pos - 8

    head[8:12] = struct.pack(">I", 0x01000000)
    head[12:16] = struct.pack(">I", rel(stream_pos))
    head[16:20] = struct.pack(">I", 0x01000000)
    head[20:24] = struct.pack(">I", rel(track_pos))
    head[24:28] = struct.pack(">I", 0x01000000)
    head[28:32] = struct.pack(">I", rel(chan_pos))
    head[0:8] = b"HEAD" + struct.pack(">I", head_size)

    # ---------------- ADPC + DATA -----------------------------------------
    head_offset = 0x40
    adpc_offset = head_offset + head_size
    adpc_entries = num_blocks * channels
    adpc = bytearray(b"ADPC" + struct.pack(">I", 16 + adpc_entries * 4))
    adpc += b"\x00" * (8 + adpc_entries * 4)
    data_offset = adpc_offset + len(adpc)
    audio_offset = data_offset + 0x20
    data_chunk = b"DATA" + struct.pack(">I", 0x20 + len(body)) + b"\x00" * 0x18 + bytes(body)

    head[data_offset_pos : data_offset_pos + 4] = struct.pack(">I", audio_offset)

    file_size = data_offset + len(data_chunk)
    header = bytearray(b"RSTM")
    header += struct.pack(">HH", 0xFEFF, 0x0100)
    header += struct.pack(">I", file_size)
    header += struct.pack(">HH", 0x40, 3)
    header += struct.pack(">II", head_offset, head_size)
    header += struct.pack(">II", adpc_offset, len(adpc))
    header += struct.pack(">II", data_offset, len(data_chunk))
    header = header.ljust(0x40, b"\x00")

    return bytes(header) + bytes(head) + bytes(adpc) + data_chunk
