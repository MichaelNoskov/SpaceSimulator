from __future__ import annotations

from enum import Enum


class SurfaceType(str, Enum):
    """Surface type for world, landing, and minimap."""

    LAND = "land"
    LAKE = "lake"

