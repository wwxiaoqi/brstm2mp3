"""Command line interface: ``python -m brstm2mp3 song.brstm``."""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np

from . import __version__
from .brstm import BrstmFile, BrstmFormatError
from . import encoder, looping, pcm as pcm_mod

BRSTM_EXTENSIONS = (".brstm",)


# --------------------------------------------------------------------------- #
# tiny progress bar
# --------------------------------------------------------------------------- #
class Progress:
    def __init__(self, label: str, total: float = 1.0, enabled: bool = True):
        self.label = label
        self.total = total or 1.0
        self.enabled = enabled
        self.tty = enabled and sys.stdout.isatty()
        self.start = time.time()
        self.last_draw = 0.0
        self.value = 0.0
        self._reported = False

    def __call__(self, fraction: float) -> None:
        self.value = min(1.0, max(0.0, float(fraction)))
        if not self.enabled:
            return
        now = time.time()
        if self.tty:
            if now - self.last_draw < 0.05 and self.value < 1.0:
                return
            self.last_draw = now
            width = 28
            filled = int(width * self.value)
            bar = "#" * filled + "-" * (width - filled)
            sys.stdout.write(f"\r  {self.label} [{bar}] {self.value * 100:5.1f}%")
            sys.stdout.flush()
        elif self.value >= 1.0 and not self._reported:
            self._reported = True
            sys.stdout.write(f"  {self.label}: done in {now - self.start:.1f}s\n")
            sys.stdout.flush()

    def close(self) -> None:
        if self.enabled and self.tty:
            sys.stdout.write("\n")
            sys.stdout.flush()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def collect_inputs(paths: list[str]) -> list[str]:
    files: list[str] = []
    for item in paths:
        if os.path.isdir(item):
            for name in sorted(os.listdir(item)):
                if name.lower().endswith(BRSTM_EXTENSIONS):
                    files.append(os.path.join(item, name))
        elif any(ch in item for ch in "*?["):
            matches = sorted(glob.glob(item))
            files.extend(m for m in matches if m.lower().endswith(BRSTM_EXTENSIONS))
        elif os.path.isfile(item):
            files.append(item)
        else:
            raise SystemExit(f"error: no such file or directory: {item}")
    # keep order but drop duplicates
    seen = set()
    unique = []
    for f in files:
        key = os.path.normcase(os.path.abspath(f))
        if key not in seen:
            seen.add(key)
            unique.append(f)
    if not unique:
        raise SystemExit("error: no .brstm files found")
    return unique


def decode_stream(brstm: BrstmFile, decoder: str, show_progress: bool):
    """Decode with either the built-in Python decoder or ffmpeg."""
    si = brstm.stream
    ffmpeg = encoder.find_ffmpeg()

    if decoder == "ffmpeg" or (decoder == "auto" and ffmpeg and si.codec == 2):
        from . import encoder as enc

        prog = Progress("decode (ffmpeg)", enabled=show_progress)
        if not show_progress:
            prog(0.0)
        data = enc.decode_with_ffmpeg(brstm.path, ffmpeg)
        rate = si.sample_rate
        if data.size % si.channels:
            data = data[: data.size - data.size % si.channels]
        prog(1.0)
        prog.close()
        return data.reshape(-1, si.channels), rate

    last = [-1]

    def cb(channel: int, fraction: float) -> None:
        if channel != last[0]:
            last[0] = channel
        overall = (channel + fraction) / si.channels
        prog(overall)

    prog = Progress(f"decode ch {si.channels}x{si.codec_name}", enabled=show_progress)
    if not show_progress:
        prog(0.0)
    data = brstm.decode(progress=cb)
    prog(1.0)
    prog.close()
    return data, si.sample_rate


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="brstm2mp3",
        description="Convert Nintendo BRSTM (DSP ADPCM) files to MP3/FLAC/OGG/WAV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python -m brstm2mp3 song.brstm
  python -m brstm2mp3 *.brstm -o mp3 -f mp3 -b 320k
  python -m brstm2mp3 song.brstm --loops 2 --fade-out 8
  python -m brstm2mp3 song.brstm --info
  python -m brstm2mp3 folder/ -f flac --decoder ffmpeg
""",
    )
    p.add_argument("inputs", nargs="+", help="BRSTM files, directories or glob patterns")
    p.add_argument("-o", "--output", default=None,
                   help="output directory (default: next to each input file)")
    p.add_argument("-f", "--format", default="mp3", choices=sorted(encoder.FORMATS),
                   help="output format (default: mp3)")
    p.add_argument("-b", "--bitrate", default="192k",
                   help="audio bitrate for lossy formats (default: 192k)")
    p.add_argument("--loops", type=int, default=0,
                   help="extra loop passes to append for looping songs (default: 0)")
    p.add_argument("--fade-in", type=float, default=0.0, metavar="SEC",
                   help="fade-in duration in seconds")
    p.add_argument("--fade-out", type=float, default=0.0, metavar="SEC",
                   help="fade-out duration in seconds")
    p.add_argument("--trim-silence", action="store_true",
                   help="strip leading/trailing digital silence")
    p.add_argument("--decoder", default="auto", choices=("auto", "python", "ffmpeg"),
                   help="decode backend: auto (ffmpeg when available), python "
                        "(built-in numpy decoder), ffmpeg (fastest)")
    p.add_argument("--info", action="store_true", help="only print stream information")
    p.add_argument("--overwrite", action="store_true", help="overwrite existing output files")
    p.add_argument("-q", "--quiet", action="store_true", help="no progress output")
    p.add_argument("-V", "--version", action="version", version=f"brstm2mp3 {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    files = collect_inputs(args.inputs)
    show_progress = not args.quiet

    if args.info:
        for path in files:
            bf = BrstmFile.open(path)
            print(bf.describe())
            print()
        return 0

    ffmpeg = encoder.find_ffmpeg()
    if args.format != "wav" and ffmpeg is None:
        print("warning: ffmpeg not found, MP3/FLAC/OGG output is unavailable "
              "(use --format wav or install ffmpeg)", file=sys.stderr)

    errors = 0
    for index, path in enumerate(files, 1):
        name = os.path.basename(path)
        print(f"[{index}/{len(files)}] {name}")
        try:
            bf = BrstmFile.open(path)
            si = bf.stream
            print(f"    {si.codec_name}, {si.channels} ch, {si.sample_rate} Hz, "
                  f"{si.duration / 60:.2f} min"
                  + (f", loop @ {si.loop_start / si.sample_rate:.2f}s" if si.loop_flag else ""))

            out_dir = args.output or os.path.dirname(os.path.abspath(path))
            os.makedirs(out_dir, exist_ok=True)
            stem = os.path.splitext(name)[0]
            ext = encoder.FORMATS[args.format][0]
            out_path = os.path.join(out_dir, f"{stem}.{ext}")

            if os.path.exists(out_path) and not args.overwrite:
                print(f"    skip (exists): {out_path}")
                continue

            t0 = time.time()
            data, rate = decode_stream(bf, args.decoder, show_progress)
            t1 = time.time()

            if args.trim_silence:
                before = len(data)
                data = looping.trim_silence(data)
                if len(data) != before:
                    print(f"    trimmed {(before - len(data)) / rate:.2f}s of silence")

            opts = looping.LoopOptions(
                loops=args.loops,
                fade_in=args.fade_in,
                fade_out=args.fade_out,
                sample_rate=rate,
            )
            if si.loop_flag and args.loops > 0:
                print(f"    appending {args.loops} loop pass(es)")
            data = looping.process(data, si.loop_flag, si.loop_start, opts)

            prog = Progress(f"encode {args.format}", enabled=show_progress)
            out_path = encoder.encode(
                out_path, data, rate, fmt=args.format, bitrate=args.bitrate,
                ffmpeg=ffmpeg, progress=prog,
            )
            prog.close()
            t2 = time.time()

            size = os.path.getsize(out_path) / (1024 * 1024)
            duration = len(data) / rate
            print(f"    -> {out_path}  ({size:.2f} MiB, {duration / 60:.2f} min)")
            print(f"    decode {t1 - t0:.1f}s, encode {t2 - t1:.1f}s")
        except (BrstmFormatError, encoder.EncodeError, OSError, ValueError) as exc:
            errors += 1
            print(f"    error: {exc}", file=sys.stderr)

    if len(files) > 1:
        print(f"done: {len(files) - errors} ok, {errors} failed")
    return 1 if errors else 0
