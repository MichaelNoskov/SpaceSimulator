from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import BodyConfig
from ..state import AtmosphereSample


@dataclass(frozen=True)
class AtmosphereTable:
    """Tabulated atmosphere profile."""

    alt_m: np.ndarray
    rho_kg_m3: np.ndarray
    t_ext_c: np.ndarray
    p_bar: np.ndarray


def load_atmosphere_table(path: Path) -> Optional[AtmosphereTable]:
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        alt_m = np.asarray(obj["alt_m"], dtype=float)
        rho = np.asarray(obj["rho_kg_m3"], dtype=float)
        t_ext_c = np.asarray(obj["t_ext_c"], dtype=float)
        p_bar = np.asarray(obj["p_bar"], dtype=float)
        if alt_m.ndim != 1 or alt_m.size < 2:
            return None
        if not (alt_m.size == rho.size == t_ext_c.size == p_bar.size):
            return None
        order = np.argsort(alt_m)
        return AtmosphereTable(
            alt_m=alt_m[order],
            rho_kg_m3=rho[order],
            t_ext_c=t_ext_c[order],
            p_bar=p_bar[order],
        )
    except Exception:
        return None


def sample_atmosphere(h_m: float, body: BodyConfig, table: Optional[AtmosphereTable]) -> AtmosphereSample:
    """
    Atmosphere parameters at altitude.

    Modes:
    - table: interpolate tabulated profile
    - fallback: exponential approximation if no table
    """

    h = max(0.0, float(h_m))

    if table is not None:
        rho = float(np.interp(h, table.alt_m, table.rho_kg_m3))
        t_ext_c = float(np.interp(h, table.alt_m, table.t_ext_c))
        p_bar = float(np.interp(h, table.alt_m, table.p_bar))
        return AtmosphereSample(rho=rho, t_ext_c=t_ext_c, p_bar=p_bar)

    # Fallback:
    # rho(h) = rho0 * exp(-h/H)
    rho = float(body.rho_surface_kg_m3) * math.exp(-h / float(body.scale_height_m))

    # Simple placeholders for temperature/pressure (replace with a richer model if needed).
    t_ext_c = -179.0 + 80.0 * math.exp(-h / 60_000.0)
    p_bar = 1.5 * math.exp(-h / 20_000.0)

    return AtmosphereSample(rho=rho, t_ext_c=t_ext_c, p_bar=p_bar)

