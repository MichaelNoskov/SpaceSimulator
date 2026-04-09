#!/usr/bin/env python3
"""Write engine JSON tables from PDS Huygens TAB files under data/nasa_pds/."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

R_SPECIFIC_TITAN = 296.8  # N2-dominated, same as DigitalTwinConfig

REPO = Path(__file__).resolve().parents[1]
NASA_PDS = REPO / "data" / "nasa_pds"
HASI = NASA_PDS / "hasi_profiles"
DWE_DIR = NASA_PDS / "dwe"


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def parse_hasi_entry(path: Path) -> list[tuple[float, float, float, float]]:
    """Rows: time_ms, alt_m, P_pa, T_k -> list of (alt_m, P_pa, T_k, rho)."""
    out: list[tuple[float, float, float, float]] = []
    for line in _read_lines(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(";")]
        if len(parts) < 4:
            continue
        try:
            alt_m = float(parts[1])
            p_pa = float(parts[2])
            t_k = float(parts[3])
        except ValueError:
            continue
        if alt_m <= 0 or p_pa <= 0 or t_k <= 0:
            continue
        rho = p_pa / (R_SPECIFIC_TITAN * t_k)
        out.append((alt_m, p_pa, t_k, rho))
    return out


def parse_hasi_descent(path: Path) -> list[tuple[float, float, float, float]]:
    """Rows: time_ms, alt_m, P_pa, T_k, rho_kg_m3."""
    out: list[tuple[float, float, float, float]] = []
    for line in _read_lines(path):
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(";")]
        if len(parts) < 5:
            continue
        try:
            alt_m = float(parts[1])
            p_pa = float(parts[2])
            t_k = float(parts[3])
            rho = float(parts[4])
        except ValueError:
            continue
        if alt_m < 0 or p_pa <= 0 or t_k <= 0 or rho <= 0:
            continue
        out.append((alt_m, p_pa, t_k, rho))
    return out


def merge_atmosphere(entry: list[tuple], descent: list[tuple]) -> list[tuple[float, float, float, float]]:
    """
    Single vertical profile, altitude ascending [m].
    On duplicate altitudes (rounded to 0.1 m), descent supersedes entry.
    """
    by_key: dict[float, tuple[float, float, float, float]] = {}
    for alt_m, p_pa, t_k, rho in entry:
        k = round(alt_m, 1)
        by_key[k] = (alt_m, p_pa, t_k, rho)
    for alt_m, p_pa, t_k, rho in descent:
        k = round(alt_m, 1)
        by_key[k] = (alt_m, p_pa, t_k, rho)
    rows = sorted(by_key.values(), key=lambda r: r[0])
    return rows


def build_titan_atm_json(rows: Iterable[tuple[float, float, float, float]]) -> dict:
    alt_m: list[float] = []
    t_ext_c: list[float] = []
    p_bar: list[float] = []
    rho_kg_m3: list[float] = []
    for alt, p_pa, t_k, rho in rows:
        alt_m.append(round(alt, 3))
        t_ext_c.append(round(t_k - 273.15, 6))
        # Keep sub-nanobar pressures / densities at entry altitudes (JSON must not round to 0).
        p_bar.append(float(p_pa / 100_000.0))
        rho_kg_m3.append(float(rho))
    return {
        "units": {
            "alt_m": "m",
            "rho_kg_m3": "kg/m^3",
            "t_ext_c": "degC",
            "p_bar": "bar",
        },
        "note": (
            "Huygens HASI L4 atmosphere (HP-SSA-HASI-2-3-4-MISSION): entry and descent "
            f"products merged. Entry: density from ideal gas P/(R·T), R={R_SPECIFIC_TITAN} J/(kg·K). "
            "Descent: HASI density column."
        ),
        "sources": [
            "https://atmos.nmsu.edu/PDS/data/PDS4/Huygens/hphasi_bundle/",
            "https://pds.nasa.gov/ds-view/pds/viewDataset.jsp?dsid=HP-SSA-HASI-2-3-4-MISSION-V1.1",
            "https://www.esa.int/Science_Exploration/Space_Science/Cassini-Huygens",
        ],
        "alt_m": alt_m,
        "t_ext_c": t_ext_c,
        "p_bar": p_bar,
        "rho_kg_m3": rho_kg_m3,
    }


def parse_zonal_wind(path: Path) -> tuple[list[float], list[float]]:
    """DWE ZONALWIND.TAB: UTC, alt_km, zonal_mps, err_mps (whitespace separated)."""
    alt_m: list[float] = []
    vx: list[float] = []
    for line in _read_lines(path):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        if "T" not in parts[0]:
            continue
        try:
            alt_km = float(parts[1])
            w = float(parts[2])
        except ValueError:
            continue
        alt_m.append(round(alt_km * 1000.0, 3))
        vx.append(round(w, 5))
    # ascending altitude for interpolation
    pairs = sorted(zip(alt_m, vx), key=lambda z: z[0])
    alt_m = [p[0] for p in pairs]
    vx = [p[1] for p in pairs]
    return alt_m, vx


def extend_wind_for_entry(alt_m: list[float], vx: list[float], top_m: float = 1_400_000.0) -> tuple[list[float], list[float]]:
    """DWE ends near ~144 km; hold last zonal value up to top_m for entry simulations."""
    if not alt_m:
        return [0.0, top_m], [0.0, 0.0]
    if alt_m[-1] < top_m - 1.0:
        alt_m = alt_m + [float(top_m)]
        vx = vx + [vx[-1]]
    return alt_m, vx


def build_wind_json(alt_m: list[float], vx: list[float]) -> dict:
    vz = [0.0] * len(alt_m)
    return {
        "units": {"alt_m": "m", "v_x_mps": "m/s (zonal, +x)", "v_z_mps": "m/s (meridional, +z)"},
        "note": (
            "Huygens DWE zonal wind (HP-SSA-DWE-2-3-DESCENT), ZONALWIND.TAB. "
            "Above the top DWE altitude the last measured zonal value is held to 1400 km "
            "to match the simulator entry altitude."
        ),
        "sources": [
            "https://atmos.nmsu.edu/PDS/data/PDS4/Huygens/hpdwe_bundle/",
            "https://pds.nasa.gov/ds-view/pds/viewDataset.jsp?dsid=HP-SSA-DWE-2-3-DESCENT-V1.0",
            "https://doi.org/10.1038/nature04060",
        ],
        "alt_m": alt_m,
        "v_x_mps": vx,
        "v_z_mps": vz,
    }


def parse_velocity(path: Path, stride: int = 2) -> dict:
    """HASI_L4_VELOCITY_PROFILE: time_ms; velocity_mps. Stride reduces JSON size."""
    t_ms: list[float] = []
    v_mps: list[float] = []
    for i, line in enumerate(_read_lines(path)):
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(";")]
        if len(parts) < 2:
            continue
        try:
            t = float(parts[0])
            v = float(parts[1])
        except ValueError:
            continue
        if i % stride != 0:
            continue
        t_ms.append(t)
        v_mps.append(round(v, 3))
    return {
        "units": {"time_ms": "ms from HASI T0", "velocity_mps": "m/s probe speed magnitude"},
        "note": "Huygens HASI L4 probe speed magnitude vs time (HASI_L4_VELOCITY_PROFILE.TAB).",
        "sources": [
            "https://atmos.nmsu.edu/PDS/data/PDS4/Huygens/hphasi_bundle/DATA/PROFILES/HASI_L4_VELOCITY_PROFILE.TAB",
        ],
        "time_ms": t_ms,
        "velocity_mps": v_mps,
    }


def main() -> int:
    entry_p = HASI / "HASI_L4_ATMO_PROFILE_ENTRY.TAB"
    desc_p = HASI / "HASI_L4_ATMO_PROFILE_DESCEN.TAB"
    wind_p = DWE_DIR / "ZONALWIND.TAB"
    vel_p = HASI / "HASI_L4_VELOCITY_PROFILE.TAB"
    for p in (entry_p, desc_p, wind_p, vel_p):
        if not p.exists():
            print(f"Missing {p}", file=sys.stderr)
            return 1

    merged = merge_atmosphere(parse_hasi_entry(entry_p), parse_hasi_descent(desc_p))
    atm = build_titan_atm_json(merged)
    (REPO / "data" / "titan_atm.json").write_text(json.dumps(atm, indent=2) + "\n", encoding="utf-8")

    wa, wx = parse_zonal_wind(wind_p)
    wa, wx = extend_wind_for_entry(wa, wx)
    wind = build_wind_json(wa, wx)
    (REPO / "data" / "titan_wind_huygens.json").write_text(json.dumps(wind, indent=2) + "\n", encoding="utf-8")

    telem = parse_velocity(vel_p, stride=2)
    (REPO / "data" / "huygens_velocity_telemetry.json").write_text(
        json.dumps(telem, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"Wrote titan_atm.json ({len(atm['alt_m'])} pts), "
        f"titan_wind_huygens.json ({len(wind['alt_m'])} pts), "
        f"huygens_velocity_telemetry.json ({len(telem['time_ms'])} pts)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
