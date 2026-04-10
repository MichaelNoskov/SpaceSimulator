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
    Atmosphere at altitude: interpolate `titan_atm.json` when loaded.

    - `np.interp` outside [min,max] keeps edge values — unphysical for ρ(h); beyond таблицы
      используется экспоненциальное продолжение с масштабом `scale_height_m`.
    - Выше `atmosphere_upper_anchor_alt_m` плотность задаётся как ρ(hₐ)·exp(−(h−hₐ)/H_up):
      табличные значения на экстремальных высотах часто неполные; так согласуем раннее торможение с барометрическим хвостом.
    """

    h = max(0.0, float(h_m))
    H_edge = float(body.scale_height_m)

    if table is not None:
        alt = table.alt_m
        rho_a = table.rho_kg_m3
        t_a = table.t_ext_c
        p_a = table.p_bar
        h_lo = float(alt[0])
        h_hi = float(alt[-1])

        def _interp_rho(hv: float) -> float:
            if hv < h_lo:
                r0 = float(rho_a[0])
                return r0 * math.exp((h_lo - hv) / H_edge)
            if hv > h_hi:
                r1 = float(rho_a[-1])
                return r1 * math.exp(-(hv - h_hi) / H_edge)
            return float(np.interp(hv, alt, rho_a))

        def _interp_tp(hv: float) -> tuple[float, float]:
            if hv < h_lo:
                t0 = float(t_a[0])
                p0 = float(p_a[0])
                rho0 = float(rho_a[0])
                rh = _interp_rho(hv)
                return t0, p0 * (rh / max(1e-30, rho0))
            if hv > h_hi:
                t1 = float(t_a[-1])
                p1 = float(p_a[-1])
                r1 = float(rho_a[-1])
                rh = _interp_rho(hv)
                return t1, p1 * (rh / max(1e-30, r1))
            return float(np.interp(hv, alt, t_a)), float(np.interp(hv, alt, p_a))

        ha = float(body.atmosphere_upper_anchor_alt_m)
        Hup = float(body.atmosphere_upper_scale_height_m)
        if ha > 0.0 and h > ha:
            ha_use = min(ha, h_hi - 1.0)
            rho0 = float(np.interp(ha_use, alt, rho_a))
            t0, p0 = float(np.interp(ha_use, alt, t_a)), float(np.interp(ha_use, alt, p_a))
            rho = rho0 * math.exp(-(h - ha_use) / max(1.0, Hup))
            t_ext_c = t0
            p_bar = p0 * (rho / max(1e-30, rho0))
            return AtmosphereSample(rho=rho, t_ext_c=t_ext_c, p_bar=p_bar)

        rho = _interp_rho(h)
        t_ext_c, p_bar = _interp_tp(h)
        return AtmosphereSample(rho=rho, t_ext_c=t_ext_c, p_bar=p_bar)

    rho = float(body.rho_surface_kg_m3) * math.exp(-h / float(body.scale_height_m))
    t_ext_c = -179.0 + 80.0 * math.exp(-h / 60_000.0)
    p_bar = 1.5 * math.exp(-h / 20_000.0)

    return AtmosphereSample(rho=rho, t_ext_c=t_ext_c, p_bar=p_bar)

