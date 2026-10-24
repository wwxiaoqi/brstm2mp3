"""Loop handling, fades and trimming.

BRSTM files usually carry a loop point. 
lets you pick how many times a loop is played, add a
fade-out and trim to the loop, exactly like the converters used by game
soundtrack rips.  This module implements those operations on a numpy PCM array.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LoopOptions:
    """How to turn a looping stream into a linear file.

    loops        number of extra loop passes to append (0 = play through once,
                 1 = play intro + loop body twice, ...)
    fade_in      fade in duration in seconds (0 = none)
    fade_out     fade out duration in seconds (0 = none).  When a loop extra
                 pass is requested the fade is applied to the tail.
    sample_rate  used to convert seconds to samples
    """

    loops: int = 0
    fade_in: float = 0.0
    fade_out: float = 0.0
    sample_rate: int = 44100

    def validate(self) -> None:
        if self.loops < 0:
            raise ValueError("loops must be >= 0")
        if self.fade_in < 0 or self.fade_out < 0:
            raise ValueError("fade durations must be >= 0")


def apply_loop(pcm: np.ndarray, loop_start: int, loops: int) -> np.ndarray:
    """Append ``loops`` extra iterations of the [loop_start:] region."""
    if loops <= 0 or loop_start >= len(pcm):
        return pcm
    body = pcm[loop_start:]
    return np.concatenate([pcm] + [body] * loops) if loops > 1 else np.concatenate([pcm, body])


def fade(pcm: np.ndarray, fade_in: float = 0.0, fade_out: float = 0.0, rate: int = 44100) -> np.ndarray:
    """Apply linear fades in place-safe fashion (returns a new array)."""
    out = pcm
    n_in = int(fade_in * rate)
    n_out = int(fade_out * rate)
    if n_in > 0 or n_out > 0:
        out = pcm.astype(np.float32, copy=True)
        if n_in > 0:
            n_in = min(n_in, len(out))
            ramp = np.linspace(0.0, 1.0, n_in, dtype=np.float32)[:, None]
            out[:n_in] *= ramp
        if n_out > 0:
            n_out = min(n_out, len(out))
            ramp = np.linspace(1.0, 0.0, n_out, dtype=np.float32)[:, None]
            out[-n_out:] *= ramp
        out = np.clip(out, -32768, 32767).astype(np.int16)
    return out


def trim_silence(pcm: np.ndarray, threshold: int = 8) -> np.ndarray:
    """Drop leading/trailing digital silence (useful for rips with padding)."""
    if len(pcm) == 0:
        return pcm
    loud = np.nonzero(np.abs(pcm.astype(np.int32)).max(axis=1) > threshold)[0]
    if len(loud) == 0:
        return pcm[:0]
    return pcm[loud[0] : loud[-1] + 1]


def process(pcm: np.ndarray, loop_flag: bool, loop_start: int, opts: LoopOptions) -> np.ndarray:
    """Full post-processing chain: loop extension, then fades."""
    opts.validate()
    if loop_flag and opts.loops > 0:
        pcm = apply_loop(pcm, loop_start, opts.loops)
    if opts.fade_in or opts.fade_out:
        pcm = fade(pcm, opts.fade_in, opts.fade_out, opts.sample_rate)
    return pcm
