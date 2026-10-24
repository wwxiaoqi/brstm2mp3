"""BRSTM container parser.

A BRSTM ("RSTM") file is a chunked big-endian container used by many Wii/GC
games (Super Smash Bros. Brawl, Mario Kart Wii, UNDERTALE's console-ish ports,
...).  Layout::

    0x00  "RSTM"                     magic
    0x04  u16 BOM (0xFEFF / 0xFFFE)  byte order of every integer in the file
    0x06  u16 version
    0x08  u32 file size
    0x0C  u16 header size            (0x40)
    0x0E  u16 section count
    0x10  { u32 offset, u32 size }   section count entries: HEAD, ADPC, DATA

    HEAD chunk (holds everything we need to decode)
        0x00 "HEAD"
        0x04 u32 size
        0x08 u32 ref type, 0x0C u32 ref offset  -> stream info
        0x10 u32 ref type, 0x14 u32 ref offset  -> track info
        0x18 u32 ref type, 0x1C u32 ref offset  -> channel info
        ...
        all ref offsets are relative to (HEAD offset + 8)

    Stream info (0x40 bytes, the thing that matters)::

        0x00 u8  codec            0=PCM8, 1=PCM16, 2=DSP ADPCM
        0x01 u8  loop flag
        0x02 u8  channel count
        0x03 u8  padding
        0x04 u16 sample rate
        0x06 u16 padding
        0x08 u32 loop start sample
        0x0C u32 sample count
        0x10 u32 audio data offset (absolute file offset)
        0x14 u32 block count
        0x18 u32 block size (bytes per channel per block)
        0x1C u32 samples per block
        0x20 u32 last block used bytes
        0x24 u32 last block sample count
        0x28 u32 last block padded bytes
        0x2C u32 adpc samples per entry
        0x30 u32 adpc bytes per entry

    Channel info::

        u32 channel count
        { u32 ref type, u32 ref offset } * channels   -> per channel struct
        channel struct: { u32 ref type, u32 ref offset } -> 32 bytes of
                        int16 DSP predictor coefficients (8 pairs)
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from typing import BinaryIO

CODEC_PCM8 = 0
CODEC_PCM16 = 1
CODEC_DSP = 2

CODEC_NAMES = {
    CODEC_PCM8: "PCM8",
    CODEC_PCM16: "PCM16",
    CODEC_DSP: "DSP ADPCM",
}

_MAGIC = b"RSTM"


class BrstmFormatError(Exception):
    """Raised when a file is not a BRSTM we understand."""


@dataclass
class StreamInfo:
    codec: int
    loop_flag: bool
    channels: int
    sample_rate: int
    loop_start: int
    num_samples: int
    data_offset: int
    num_blocks: int
    block_size: int
    samples_per_block: int
    last_block_used_bytes: int
    last_block_samples: int
    last_block_padded_bytes: int = 0
    adpc_samples_per_entry: int = 0
    adpc_bytes_per_entry: int = 0

    @property
    def codec_name(self) -> str:
        return CODEC_NAMES.get(self.codec, f"unknown({self.codec})")

    @property
    def duration(self) -> float:
        return self.num_samples / self.sample_rate if self.sample_rate else 0.0


@dataclass
class Section:
    name: str
    offset: int
    size: int


@dataclass
class BrstmFile:
    """Parsed BRSTM file.  Use :meth:`decode` to get interleaved PCM samples."""

    path: str | None = None
    endian: str = ">"
    version: int = 0
    file_size: int = 0
    sections: dict[str, Section] = field(default_factory=dict)
    stream: StreamInfo = None  # type: ignore[assignment]
    coefficients: list[list[tuple[int, int]]] = field(default_factory=list)
    _data: bytes = b""

    # ------------------------------------------------------------------ parse
    @classmethod
    def open(cls, path: str | os.PathLike) -> "BrstmFile":
        with open(path, "rb") as fh:
            raw = fh.read()
        obj = cls.from_bytes(raw)
        obj.path = os.fspath(path)
        return obj

    @classmethod
    def from_bytes(cls, raw: bytes) -> "BrstmFile":
        if len(raw) < 0x40:
            raise BrstmFormatError("file too small to be a BRSTM")
        if raw[:4] != _MAGIC:
            raise BrstmFormatError(
                f"bad magic {raw[:4]!r}, expected b'RSTM' (BRSTM). "
                "BCSTM/BFSTM/BRWAV files are not supported."
            )

        bom = struct.unpack(">H", raw[4:6])[0]
        if bom == 0xFEFF:
            e = ">"
        elif bom == 0xFFFE:
            e = "<"
        else:
            raise BrstmFormatError(f"unknown BOM 0x{bom:04X}")

        u16 = lambda off: struct.unpack(e + "H", raw[off : off + 2])[0]
        u32 = lambda off: struct.unpack(e + "I", raw[off : off + 4])[0]

        self = cls()
        self.endian = e
        self.version = u16(0x06)
        self.file_size = u32(0x08)
        header_size = u16(0x0C)
        section_count = u16(0x0E)

        if header_size < 0x10 + section_count * 8:
            raise BrstmFormatError("corrupt header: header size too small")

        for i in range(section_count):
            entry = 0x10 + i * 8
            offset = u32(entry)
            size = u32(entry + 4)
            if offset + 8 > len(raw):
                continue
            name = raw[offset : offset + 4].decode("ascii", "replace").strip("\x00")
            self.sections[name] = Section(name, offset, size)

        head = self.sections.get("HEAD")
        if head is None:
            raise BrstmFormatError("no HEAD chunk found")

        base = head.offset + 8

        # ---- stream info ------------------------------------------------
        info_off = base + u32(head.offset + 0x0C)
        if info_off + 0x34 > len(raw):
            raise BrstmFormatError("stream info out of range")

        self.stream = StreamInfo(
            codec=raw[info_off],
            loop_flag=bool(raw[info_off + 1]),
            channels=raw[info_off + 2],
            sample_rate=struct.unpack(e + "H", raw[info_off + 4 : info_off + 6])[0],
            loop_start=u32(info_off + 0x08),
            num_samples=u32(info_off + 0x0C),
            data_offset=u32(info_off + 0x10),
            num_blocks=u32(info_off + 0x14),
            block_size=u32(info_off + 0x18),
            samples_per_block=u32(info_off + 0x1C),
            last_block_used_bytes=u32(info_off + 0x20),
            last_block_samples=u32(info_off + 0x24),
            last_block_padded_bytes=u32(info_off + 0x28),
            adpc_samples_per_entry=u32(info_off + 0x2C),
            adpc_bytes_per_entry=u32(info_off + 0x30),
        )
        si = self.stream
        if si.channels <= 0 or si.channels > 16:
            raise BrstmFormatError(f"unsupported channel count {si.channels}")
        if si.sample_rate <= 0:
            raise BrstmFormatError("invalid sample rate")
        if si.num_samples == 0:
            raise BrstmFormatError("zero length stream")
        if si.block_size == 0 or si.samples_per_block == 0:
            raise BrstmFormatError("invalid block layout")

        # fall back to sane values on files with sparse headers
        if si.last_block_used_bytes == 0:
            si.last_block_used_bytes = si.block_size
        if si.last_block_samples == 0:
            si.last_block_samples = si.samples_per_block

        # ---- channel info / DSP coefficients ----------------------------
        if si.codec == CODEC_DSP:
            ref3 = u32(head.offset + 0x1C)
            chan_info = base + ref3
            channel_refs = []
            for ch in range(si.channels):
                ref = chan_info + 4 + ch * 8
                if ref + 8 > len(raw):
                    break
                channel_refs.append(base + u32(ref + 4))

            if not channel_refs:  # some rips have a flat reference list
                channel_refs = [chan_info + 4 + ch * 8 for ch in range(si.channels)]

            for ref in channel_refs:
                coef_off = base + u32(ref + 4)
                if coef_off + 32 > len(raw):
                    raise BrstmFormatError("coefficient table out of range")
                table = struct.unpack(
                    e + "16h", raw[coef_off : coef_off + 32]
                )
                self.coefficients.append(
                    [(table[2 * i], table[2 * i + 1]) for i in range(8)]
                )

            if len(self.coefficients) < si.channels:
                raise BrstmFormatError("missing DSP coefficient tables")

        self._data = raw
        return self

    # ------------------------------------------------------------------ info
    @property
    def data(self) -> bytes:
        return self._data

    def describe(self) -> str:
        si = self.stream
        lines = [
            f"file        : {self.path or '<memory>'}",
            f"format      : BRSTM version 0x{self.version:04X} "
            f"({'big' if self.endian == '>' else 'little'} endian)",
            f"codec       : {si.codec_name}",
            f"channels    : {si.channels}",
            f"sample rate : {si.sample_rate} Hz",
            f"samples     : {si.num_samples} ({si.duration:.2f} s)",
            f"loop        : "
            + (
                f"yes, start at sample {si.loop_start} "
                f"({si.loop_start / si.sample_rate:.3f} s)"
                if si.loop_flag
                else "no"
            ),
            f"layout      : {si.num_blocks} blocks x {si.block_size} bytes, "
            f"{si.samples_per_block} samples/block",
        ]
        return "\n".join(lines)

    # ----------------------------------------------------------------- decode
    def decode(self, progress=None):
        """Decode the whole stream to a numpy int16 array of shape (samples, channels)."""
        from . import pcm

        si = self.stream
        if si.codec == CODEC_DSP:
            return pcm.decode_dsp(self, progress=progress)
        if si.codec == CODEC_PCM16:
            return pcm.decode_pcm16(self, progress=progress)
        if si.codec == CODEC_PCM8:
            return pcm.decode_pcm8(self, progress=progress)
        raise BrstmFormatError(f"unsupported codec {si.codec}")
