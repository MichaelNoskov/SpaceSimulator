from __future__ import annotations

import math

import numpy as np


def wind_vec_mps(h_m: float) -> tuple[float, float]:
    """
    Wind vector as a function of altitude.

    Returns:
    - (w_x, w_z) [m/s]

    Model:
    - wind speed grows smoothly with altitude
    - direction varies slowly with altitude (deterministic)
    """

    h = float(np.clip(float(h_m) / 100_000.0, 0.0, 1.0))
    mag = float(0.5 + (50.0 - 0.5) * (h**1.2))
    ang = float(0.6 + 1.4 * (h**0.9))
    return mag * math.cos(ang), mag * math.sin(ang)
