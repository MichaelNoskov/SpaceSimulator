from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class WindTables:
    """Mean wind (DWE zonal + derived meridional) and 1σ retrieval uncertainty vs altitude."""

    alt_m: np.ndarray
    v_x_mps: np.ndarray
    v_z_mps: np.ndarray
    sigma_mps: np.ndarray


def _load_tables(path: Path) -> Optional[WindTables]:
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
        sig = obj.get("sigma_zonal_mps")
        if sig is None:
            sigma = 0.45 + 0.025 * np.abs(vx)
        else:
            sigma = np.asarray(sig, dtype=float)
            if sigma.size != alt_m.size:
                return None
        order = np.argsort(alt_m)
        return WindTables(
            alt_m=alt_m[order],
            v_x_mps=vx[order],
            v_z_mps=vz[order],
            sigma_mps=np.maximum(0.05, sigma[order]),
        )
    except Exception:
        return None


_TABLES: Optional[WindTables] = _load_tables(_REPO_ROOT / "data" / "titan_wind_huygens.json")
if _TABLES is None:
    z = np.array([0.0, 1.5e6], dtype=float)
    _TABLES = WindTables(
        alt_m=z,
        v_x_mps=np.zeros(2),
        v_z_mps=np.zeros(2),
        sigma_mps=np.full(2, 0.5),
    )


def wind_mean_vec_mps(h_m: float) -> tuple[float, float]:
    """DWE-based mean horizontal wind (zonal + meridional) [m/s]."""
    h = float(h_m)
    wx = float(np.interp(h, _TABLES.alt_m, _TABLES.v_x_mps))
    wz = float(np.interp(h, _TABLES.alt_m, _TABLES.v_z_mps))
    return wx, wz


def wind_sigma_zonal_mps(h_m: float) -> float:
    """DWE formal 1σ zonal uncertainty (m/s), interpolated; used as turbulence intensity scale."""
    h = float(h_m)
    return float(np.interp(h, _TABLES.alt_m, _TABLES.sigma_mps))


def wind_vec_mps(h_m: float) -> tuple[float, float]:
    """Backward-compatible alias: mean wind only (gusts live on PhysicsModel)."""
    return wind_mean_vec_mps(h_m)
