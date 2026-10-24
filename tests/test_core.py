"""Unit tests that do not need a real BRSTM file:  python -m unittest -v"""

from __future__ import annotations

import os
import struct
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brstm2mp3 import builder, looping, pcm  # noqa: E402
from brstm2mp3.brstm import BrstmFile, BrstmFormatError  # noqa: E402


class LoopTests(unittest.TestCase):
    def test_apply_loop_appends_body(self):
        pcm_data = np.arange(10, dtype=np.int16).reshape(10, 1)
        out = looping.apply_loop(pcm_data, loop_start=4, loops=2)
        self.assertEqual(len(out), 10 + 6 + 6)
        np.testing.assert_array_equal(out[10:16, 0], np.arange(4, 10))

    def test_no_loop_when_zero(self):
        pcm_data = np.arange(10, dtype=np.int16).reshape(10, 1)
        np.testing.assert_array_equal(looping.apply_loop(pcm_data, 4, 0), pcm_data)

    def test_fade_endpoints(self):
        pcm_data = np.full((1000, 1), 1000, dtype=np.int16)
        out = looping.fade(pcm_data, fade_in=0.01, fade_out=0.01, rate=1000)
        self.assertEqual(int(out[0, 0]), 0)
        self.assertEqual(int(out[-1, 0]), 0)
        self.assertEqual(int(out[500, 0]), 1000)

    def test_trim_silence(self):
        data = np.zeros((20, 1), dtype=np.int16)
        data[5:15] = 100
        out = looping.trim_silence(data)
        self.assertEqual(len(out), 10)


class ParserTests(unittest.TestCase):
    def test_roundtrip_generated_stream(self):
        """Build a tiny synthetic stereo DSP BRSTM and decode it back."""
        pcm_in = np.zeros((64 * 14, 2), dtype=np.int16)
        t = np.linspace(0, 1, len(pcm_in))
        pcm_in[:, 0] = (np.sin(2 * np.pi * 5 * t) * 8000).astype(np.int16)
        pcm_in[:, 1] = (np.sin(2 * np.pi * 7 * t) * 4000).astype(np.int16)

        raw = builder.build_brstm(pcm_in, sample_rate=32000)
        bf = BrstmFile.from_bytes(raw)
        self.assertEqual(bf.stream.channels, 2)
        self.assertEqual(bf.stream.sample_rate, 32000)
        self.assertEqual(bf.stream.codec, 2)
        decoded = bf.decode()
        self.assertEqual(decoded.shape[1], 2)
        self.assertEqual(len(decoded), len(pcm_in))
        # ADPCM is lossy, but a coarse check keeps the encoder honest
        self.assertLess(np.abs(decoded[:, 0].astype(int) - pcm_in[:, 0]).mean(), 400)

    def test_rejects_non_brstm(self):
        with self.assertRaises(BrstmFormatError):
            BrstmFile.from_bytes(b"RIFF" + b"\x00" * 128)

    def test_bit_exact_against_ffmpeg(self):
        """Our decoder must match ffmpeg sample for sample on generated files."""
        from brstm2mp3 import encoder

        ffmpeg = encoder.find_ffmpeg()
        if ffmpeg is None:
            self.skipTest("ffmpeg not installed")
        t = np.linspace(0, 0.1, 4410)
        pcm_in = np.stack([
            (np.sin(2 * np.pi * 220 * t) * 12000).astype(np.int16),
            (np.sin(2 * np.pi * 330 * t) * 9000).astype(np.int16),
        ], axis=1)
        raw = builder.build_brstm(pcm_in, sample_rate=44100)
        bf = BrstmFile.from_bytes(raw)
        mine = bf.decode()

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".brstm", delete=False) as fh:
            fh.write(raw)
            path = fh.name
        try:
            ref = encoder.decode_with_ffmpeg(path, ffmpeg)
        finally:
            os.unlink(path)
        ref = ref[: mine.size].reshape(-1, 2)
        n = min(len(mine), len(ref))
        np.testing.assert_array_equal(mine[:n], ref[:n])


if __name__ == "__main__":
    unittest.main()
