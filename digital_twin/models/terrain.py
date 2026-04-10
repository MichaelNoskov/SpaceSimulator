from __future__ import annotations

"""
Legacy wrapper.

This module used to hold procedural surface generation.
World generation now lives in `digital_twin.world.WorldGen` with seed support.
"""

from ..types import SurfaceType
from ..world import WorldGen

_DEFAULT_WORLD = WorldGen(seed=551)


def surface_at(x_m: float, z_m: float) -> SurfaceType:
    """Return surface type (LAND/LAKE) for legacy call sites."""

    return _DEFAULT_WORLD.surface_type_at(float(x_m), float(z_m))
