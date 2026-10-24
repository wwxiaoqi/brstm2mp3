"""Bit exactness check: built-in decoder vs ffmpeg.

Usage:
    python verify.py song.brstm [more.brstm ...]

ffmpeg is used as the reference decoder (it ships a BRSTM demuxer plus the
"adpcm_thp" decoder).  A correct pure Python port must produce *identical*
int16 samples, not just a similar looking waveform.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from brstm2mp3 import encoder
from brstm2mp3.brstm import BrstmFile


def compare(path: str) -> bool:
    print(f"== {path}")
    ffmpeg = encoder.find_ffmpeg()
    if ffmpeg is None:
        print("   ffmpeg not found, cannot compare")
        return False

    bf = BrstmFile.open(path)
    si = bf.stream
    print(f"   {si.codec_name}, {si.channels} ch, {si.sample_rate} Hz, "
          f"{si.num_samples} samples")

    t0 = time.time()
    mine = bf.decode()
    t1 = time.time()
    ref = encoder.decode_with_ffmpeg(path, ffmpeg)
    t2 = time.time()
    print(f"   python decode {t1 - t0:.2f}s, ffmpeg decode {t2 - t1:.2f}s")

    ref = ref[: mine.size].reshape(-1, si.channels)
    n = min(len(mine), len(ref))
    a = mine[:n].astype(np.int32)
    b = ref[:n].astype(np.int32)

    ok = True
    for ch in range(si.channels):
        diff = np.abs(a[:, ch] - b[:, ch])
        identical = bool(np.array_equal(a[:, ch], b[:, ch]))
        ok &= identical
        status = "IDENTICAL" if identical else "DIFFERENT"
        print(f"   ch{ch}: {status}  max |diff| = {int(diff.max())}, "
              f"mismatched samples = {int((diff != 0).sum())}")
    return ok


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    results = [compare(p) for p in argv]
    print()
    print("all files bit exact" if all(results) else "some files differ from ffmpeg")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
