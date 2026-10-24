"""brstm2mp3 - Nintendo BRSTM (ADPCM) -> WAV / MP3 converter written in pure Python.

Modules
-------
brstm    : BRSTM container parser (header, HEAD/ADPC/DATA chunks)
dsp      : GameCube style DSP ADPCM decoder (numpy, block parallel)
pcm      : PCM8 / PCM16 codec paths
looping  : loop expansion, fades and trimming
encoder  : WAV writer + ffmpeg backed encoders (mp3/flac/ogg/m4a)
cli      : command line interface
"""

from .brstm import BrstmFile, BrstmFormatError
from .looping import LoopOptions

__version__ = "1.0.0"
__all__ = ["BrstmFile", "BrstmFormatError", "LoopOptions", "__version__"]
