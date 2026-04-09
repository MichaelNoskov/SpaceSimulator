from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class WindTable:
    """Tabulated horizontal wind vs altitude (MSL height above local surface in the sim)."""

    alt_m: np.ndarray
    v_x_mps: np.ndarray
    v_z_mps: np.ndarray


def load_wind_table(path: Path) -> Optional[WindTable]:
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        alt_m = np.asarray(obj["alt_m"], dtype=float)
        vx = np.asarray(obj["v_x_mps"], dtype=float)
        vz = np.asarray(obj["v_z_mps"], dtype=float)
        if alt_m.ndim != 1 or alt_m.size < 2:
            return None
        if not (alt_m.size == vx.size == vz.size):
            return None
        order = np.argsort(alt_m)
        return WindTable(alt_m=alt_m[order], v_x_mps=vx[order], v_z_mps=vz[order])
    except Exception:
        return None


_TABLE: Optional[WindTable] = load_wind_table(_REPO_ROOT / "data" / "titan_wind_huygens.json")
if _TABLE is None:
    z = np.array([0.0, 1.5e6], dtype=float)
    _TABLE = WindTable(alt_m=z, v_x_mps=np.zeros(2), v_z_mps=np.zeros(2))


def wind_vec_mps(h_m: float) -> tuple[float, float]:
    """Horizontal wind (w_x, w_z) [m/s] from `data/titan_wind_huygens.json` (Huygens DWE)."""
    h = float(h_m)
    wx = float(np.interp(h, _TABLE.alt_m, _TABLE.v_x_mps))
    wz = float(np.interp(h, _TABLE.alt_m, _TABLE.v_z_mps))
    return wx, wz
