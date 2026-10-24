"""Audio export: WAV writer plus ffmpeg backed encoders."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import wave

import numpy as np

FFMPEG_NAMES = ("ffmpeg", "ffmpeg.exe")
_COMMON_WINDOWS_PATHS = (
    r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
    r"C:\ffmpeg\bin\ffmpeg.exe",
    r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
)

# format -> (extension, ffmpeg codec args)
FORMATS = {
    "mp3": ("mp3", ["-c:a", "libmp3lame"]),
    "wav": ("wav", None),  # native writer
    "flac": ("flac", ["-c:a", "flac"]),
    "ogg": ("ogg", ["-c:a", "libvorbis"]),
    "m4a": ("m4a", ["-c:a", "aac"]),
}

QUALITY_ARGS = {
    "mp3": "-b:a",
    "flac": None,
    "ogg": "-b:a",
    "m4a": "-b:a",
}


class EncodeError(Exception):
    pass


def find_ffmpeg() -> str | None:
    """Locate an ffmpeg binary (PATH, FFMPEG env var, usual Windows spots)."""
    env = os.environ.get("FFMPEG")
    if env and os.path.isfile(env):
        return env
    for name in FFMPEG_NAMES:
        found = shutil.which(name)
        if found:
            return found
    for path in _COMMON_WINDOWS_PATHS:
        if os.path.isfile(path):
            return path
    return None


def write_wav(path: str, pcm: np.ndarray, sample_rate: int) -> None:
    """Write an int16 (samples, channels) array as a RIFF/WAVE file."""
    channels = 1 if pcm.ndim == 1 else pcm.shape[1]
    data = np.ascontiguousarray(pcm, dtype="<i2")
    with wave.open(path, "wb") as fh:
        fh.setnchannels(channels)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(data.tobytes())


def _ffmpeg_run(ffmpeg: str, args: list[str], pcm: bytes | None = None,
                total_frames: int = 0, progress=None) -> None:
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    if pcm is not None:
        cmd += ["-f", "s16le", "-ar", str(args.pop(0)), "-ac", str(args.pop(0)), "-i", "pipe:0"]
    cmd += args
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE if pcm is not None else subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        if pcm is not None:
            proc.stdin.write(pcm)  # type: ignore[union-attr]
            proc.stdin.close()  # type: ignore[union-attr]
        stderr = proc.stderr.read() if proc.stderr else b""
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
    if proc.returncode != 0:
        raise EncodeError(
            "ffmpeg failed (exit %d):\n%s"
            % (proc.returncode, stderr.decode("utf-8", "replace").strip())
        )


def encode(
    path: str,
    pcm: np.ndarray,
    sample_rate: int,
    fmt: str = "mp3",
    bitrate: str = "192k",
    ffmpeg: str | None = None,
    progress=None,
) -> str:
    """Encode ``pcm`` to ``path``.  Returns the path actually written."""
    fmt = fmt.lower()
    if fmt not in FORMATS:
        raise EncodeError(f"unsupported format {fmt!r}; choose from {', '.join(FORMATS)}")

    ext, codec_args = FORMATS[fmt]
    if not path.lower().endswith("." + ext):
        path += "." + ext

    if fmt == "wav":
        write_wav(path, pcm, sample_rate)
        if progress:
            progress(1.0)
        return path

    if ffmpeg is None:
        ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise EncodeError(
            f"ffmpeg is required to write {fmt.upper()} files.\n"
            "Install it (https://ffmpeg.org/download.html) and make sure it is on PATH,\n"
            "or set the FFMPEG environment variable to the full ffmpeg executable path.\n"
            "Use --format wav to write an uncompressed file without ffmpeg."
        )

    channels = 1 if pcm.ndim == 1 else pcm.shape[1]
    data = np.ascontiguousarray(pcm, dtype="<i2").tobytes()
    args: list[str] = [str(sample_rate), str(channels)]
    args += ["-vn", "-sn"] + list(codec_args or [])
    if QUALITY_ARGS.get(fmt) and bitrate:
        args += [QUALITY_ARGS[fmt], bitrate]
    args += ["-map_metadata", "-1", "-id3v2_version", "3"]
    if fmt == "mp3":
        args += ["-write_xing", "1"]
    args.append(path)

    _ffmpeg_run(ffmpeg, args, pcm=data, total_frames=len(pcm), progress=progress)
    if progress:
        progress(1.0)
    return path


def decode_with_ffmpeg(path: str, ffmpeg: str | None = None):
    """Fast path: let ffmpeg decode the BRSTM to interleaved int16 samples."""
    if ffmpeg is None:
        ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise EncodeError("ffmpeg not found")
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-i", path,
        "-f", "s16le", "-acodec", "pcm_s16le", "-",
    ]
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if out.returncode != 0:
        raise EncodeError(out.stderr.decode("utf-8", "replace"))
    return np.frombuffer(out.stdout, dtype="<i2").copy()


def probe_ffmpeg_sample_rate(path: str, ffmpeg: str | None = None) -> int:
    """Sample rate of the first audio stream, via ffmpeg -i."""
    if ffmpeg is None:
        ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        return 0
    out = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", path], stderr=subprocess.PIPE, stdout=subprocess.DEVNULL
    )
    text = out.stderr.decode("utf-8", "replace")
    for line in text.splitlines():
        if "Audio:" in line:
            for token in line.split(","):
                token = token.strip()
                if token.endswith("Hz"):
                    digits = "".join(ch for ch in token if ch.isdigit())
                    if digits:
                        return int(digits)
    return 0
