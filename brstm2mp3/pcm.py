"""Sample decoding (DSP ADPCM, PCM16, PCM8) for BRSTM streams.

DSP ADPCM ("GameCube ADPCM" / THP ADPCM) in one paragraph
---------------------------------------------------------
The audio payload is split in blocks of ``block_size`` bytes per channel.  For
multi channel streams the blocks are interleaved block by block
(ch0 block0, ch1 block0, ch0 block1, ...).  A block is a run of 8 byte frames
and every frame carries 14 samples:

    byte 0      header:  high nibble = predictor index 0..7
                         low  nibble = scale 0..12
    bytes 1..7  14 signed 4 bit nibbles, high nibble first

Each sample is rebuilt with a 2 tap predictor taken from the 8 coefficient
pairs stored in the BRSTM HEAD chunk::

    sample = clamp16((((nib << scale) << 11) + c1 * yn1 + c2 * yn2) >> 11)

``yn1``/``yn2`` are the two previously decoded samples, and — important — that
history is *not* reset at block boundaries: the ADPCM stream is one continuous
signal that only happens to be chopped into interleaved blocks.  (Tools that
reset the history per block produce a tiny click every ~0.3 s; this
implementation keeps it continuous and is bit exact with ffmpeg/vgmstream.)
"""

from __future__ import annotations

from array import array

import numpy as np

# --------------------------------------------------------------------------- #
# lookup tables (built once)
# --------------------------------------------------------------------------- #

# signed value of every possible nibble, as a 256 entry table of byte pairs
_NIBBLES = [(b >> 4, b & 0x0F) for b in range(256)]
_NIBBLE_VALUE = [n - 16 if n >= 8 else n for n in range(16)]

# (nibble << scale) << 11 for every scale, indexed by the raw 0..15 nibble
_SCALED = [
    tuple((_NIBBLE_VALUE[n] << scale) << 11 for n in range(16))
    for scale in range(16)
]


def block_offsets(brstm, channel: int) -> np.ndarray:
    """Absolute file offset of every block belonging to ``channel``.

    Blocks of all channels are interleaved (ch0 block0, ch1 block0, ch0 block1,
    ...) except for the final block, which is only stored once per channel and
    padded up to a 32 byte boundary.  Walking the blocks in file order keeps
    that padding accounted for.
    """
    si = brstm.stream
    last_used = min(si.last_block_used_bytes or si.block_size, si.block_size)
    last_slot = max(si.last_block_padded_bytes, last_used)
    last_slot = last_slot if last_slot >= last_used else (last_used + 31) // 32 * 32
    offsets = []
    pos = si.data_offset
    for block in range(si.num_blocks):
        size, slot = (si.block_size, si.block_size) if block < si.num_blocks - 1 else (last_used, last_slot)
        for ch in range(si.channels):
            if ch == channel:
                offsets.append(pos)
            pos += slot
    return np.asarray(offsets, dtype=np.int64)


def channel_payload(brstm, channel: int) -> bytes:
    """Contiguous copy of one channel's interleaved blocks.

    The final block is nominally shorter (``last_block_used_bytes``) but is
    padded to a 32 byte boundary, and its tail still feeds the last few valid
    samples, so we read up to a full block just like ffmpeg/vgmstream do.
    """
    si = brstm.stream
    raw = brstm.data
    limit = len(raw)
    parts = []
    for off in block_offsets(brstm, channel).tolist():
        if off >= limit:
            break
        parts.append(raw[off : min(off + si.block_size, limit)])
    return b"".join(parts)


def decode_dsp_channel(buf: bytes, coefficients, samples: int, progress=None) -> array:
    """Decode one channel of DSP ADPCM data into an ``array('h')``."""
    coef = tuple((int(c1), int(c2)) for c1, c2 in coefficients)
    scaled = _SCALED
    nibbles = _NIBBLES
    out = array("h")
    append = out.append
    h1 = 0
    h2 = 0
    total = len(buf) & ~7
    done = 0
    step = 8192
    for start in range(0, total, step):
        stop = min(start + step, total)
        for i in range(start, stop, 8):
            header = buf[i]
            c1, c2 = coef[header >> 4]
            row = scaled[header & 0x0F]
            for byte in buf[i + 1 : i + 8]:
                hi, lo = nibbles[byte]

                s = (row[hi] + c1 * h1 + c2 * h2) >> 11
                if s > 32767:
                    s = 32767
                elif s < -32768:
                    s = -32768
                append(s)
                h2 = h1
                h1 = s

                s = (row[lo] + c1 * h1 + c2 * h2) >> 11
                if s > 32767:
                    s = 32767
                elif s < -32768:
                    s = -32768
                append(s)
                h2 = h1
                h1 = s
        done = stop
        if progress is not None and samples:
            progress(min(1.0, done * 14 / 8 / samples))
    if len(out) > samples:
        del out[samples:]
    return out


def decode_dsp(brstm, progress=None) -> np.ndarray:
    """Decode a DSP ADPCM BRSTM stream -> int16 array shaped (samples, channels)."""
    si = brstm.stream
    result = np.zeros((si.num_samples, si.channels), dtype=np.int16)
    for ch in range(si.channels):
        buf = channel_payload(brstm, ch)
        cb = None
        if progress is not None:
            def cb(fraction, _ch=ch):
                progress(_ch, fraction)

        pcm = decode_dsp_channel(buf, brstm.coefficients[ch], si.num_samples, cb)
        data = np.frombuffer(pcm, dtype=np.int16) if pcm else np.zeros(0, np.int16)
        n = min(len(data), si.num_samples)
        result[:n, ch] = data[:n]
        if progress is not None:
            progress(ch, 1.0)
    if progress is not None:
        progress(si.channels, 1.0)
    return result


# --------------------------------------------------------------------------- #
# PCM streams
# --------------------------------------------------------------------------- #

def _pcm_flat(brstm) -> np.ndarray:
    si = brstm.stream
    e = brstm.endian
    data = brstm.data
    wanted = si.num_samples * si.channels
    if si.block_size and si.block_size % (2 * si.channels) == 0:
        parts = []
        for b in range(si.num_blocks):
            off = si.data_offset + b * si.block_size * si.channels
            avail = max(0, min(si.block_size * si.channels, len(data) - off))
            if avail < 2 * si.channels:
                break
            parts.append(
                np.frombuffer(data, dtype=e + "i2", count=avail // 2, offset=off)
            )
        flat = np.concatenate(parts) if parts else np.zeros(0, np.int16)
    else:
        avail = max(0, min(wanted, (len(data) - si.data_offset) // 2))
        flat = np.frombuffer(data, dtype=e + "i2", count=avail, offset=si.data_offset)
    flat = np.asarray(flat, dtype=np.int16)
    if len(flat) < wanted:
        flat = np.concatenate([flat, np.zeros(wanted - len(flat), np.int16)])
    return flat[:wanted]


def decode_pcm16(brstm, progress=None) -> np.ndarray:
    si = brstm.stream
    return _pcm_flat(brstm).reshape(-1, si.channels)


def decode_pcm8(brstm, progress=None) -> np.ndarray:
    si = brstm.stream
    wanted = si.num_samples * si.channels
    avail = max(0, min(wanted, len(brstm.data) - si.data_offset))
    raw = np.frombuffer(brstm.data, dtype=np.uint8, count=avail, offset=si.data_offset)
    pcm = (raw.astype(np.int16) - 128) << 8
    if len(pcm) < wanted:
        pcm = np.concatenate([pcm, np.zeros(wanted - len(pcm), np.int16)])
    return pcm[:wanted].reshape(-1, si.channels)
