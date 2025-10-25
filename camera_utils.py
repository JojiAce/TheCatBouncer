"""Utility helpers for working with camera sources."""
from __future__ import annotations

from typing import Union


def resolve_camera_source(source: Union[str, int]) -> Union[str, int]:
    """Return a value that can be passed to :class:`cv2.VideoCapture`.

    The configuration allows both integer indices (e.g. ``0``) and string based
    sources such as RTSP/HTTP URLs or file paths.  Historically the code always
    cast the value to :class:`int`, which breaks whenever a non-numeric source is
    used.  This helper keeps numeric strings compatible with the previous
    behaviour while leaving other strings untouched.
    """

    if isinstance(source, int):
        return source

    if not isinstance(source, str):
        raise TypeError(f"Unsupported camera source type: {type(source)!r}")

    stripped = source.strip()
    if not stripped:
        raise ValueError("Camera source string is empty.")

    try:
        # Only return an int if the entire string represents an integer value.
        # ``int`` happily accepts leading ``+``/``-`` so we guard against that
        # behaviour explicitly.
        if stripped.isdigit():
            return int(stripped)
        if stripped[0] in {"+", "-"} and stripped[1:].isdigit():
            return int(stripped)
    except (IndexError, ValueError):
        pass

    return stripped
