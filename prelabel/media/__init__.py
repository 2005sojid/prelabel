"""Image and video I/O, plus server-side frame annotation."""

from .drawing import color_for, draw
from .images import decode_data_url, decode_image, read_image, try_decode_image
from .video import Codec, VideoReader, WriteReport, effective_codec, preferred_codec, write_video

__all__ = [
    "Codec",
    "VideoReader",
    "WriteReport",
    "color_for",
    "decode_data_url",
    "decode_image",
    "draw",
    "effective_codec",
    "preferred_codec",
    "read_image",
    "try_decode_image",
    "write_video",
]
