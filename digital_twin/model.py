from __future__ import annotations

import csv
import math
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Optional, TextIO, Tuple

import numpy as np

from .config import DigitalTwinConfig, entry_velocity_inertial_mps
from .dynamics import (
    aero_params_for_state,
    drag_force_vector_n,
    fuel_burn_kg,
    heatshield_skin_dTdt,
    heatshield_skin_step,
    thermal_relaxation_step,
)
from .models.atmosphere import AtmosphereTable, load_atmosphere_table, sample_atmosphere
from .models.wind import wind_mean_vec_mps, wind_sigma_zonal_mps
from .world import WorldGen
from .types import SurfaceType


class SimResult(str, Enum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"


@dataclass
class Target:
    """Landing target in world xz [m]; surface type is used for success check."""

    x_m: float = 0.0
    z_m: float = 0.0
    surface_type: SurfaceType = SurfaceType.LAND


class PhysicsModel:
    """
    Physics-only model; state is exposed via getters.

    Base parameters are private fields accessed as `@property` (with setters where needed).
    Derived quantities are computed in `@property` methods using only other properties.

    Per-tick cache: expensive derived values are recomputed at most once per `step()`
    (stable within one render frame).
    """

    def __init__(self, config: Optional[DigitalTwinConfig] = None, resource_root: Optional[Path] = None, seed: int = 1337):
        self._cfg = config or DigitalTwinConfig()
        self._resource_root = resource_root
        self._atm_table: Optional[AtmosphereTable] = None
        if resource_root is not None:
            self._atm_table = load_atmosphere_table(resource_root / "data" / "titan_atm.json")

        # Procedural world (terrain + lakes), deterministic from seed.
        self._world = WorldGen(seed=int(seed))

        # Full mission timeline for dossier plots (unbounded; do not drop early samples).
        self._telemetry_history: list[dict[str, float]] = []
        # (sim_time_s, tag_id, detail) — successful commands for mission plots (auto + manual).
        self._plot_action_log: Deque[Tuple[float, str, str]] = deque(maxlen=12000)
        self._tick_id = 0
        self._cache: dict[str, tuple[int, object]] = {}
        self._physics_seed = int(seed) & 0xFFFFFFFF
        # Mission start parameters (editable via UI before / between flights).
        self._mission_entry_start_altitude_m = float(self._cfg.body.entry_start_altitude_m)
        self._mission_entry_speed_mps = float(self._cfg.body.entry_speed_mps)
        self._mission_fuel_kg = 50.0
        self._mission_dry_mass_kg = 270.0
        self._mission_heatshield_mass_kg = 30.0
        self._mission_a_ref_m2 = 2.5
        self._mission_cd_base = 1.2
        self._mission_drogue_area_m2 = float(self._cfg.drogue_area_m2)
        self._mission_drogue_cd = float(self._cfg.drogue_cd)
        self._mission_main_chute_area_m2 = float(self._cfg.main_chute_area_m2)
        self._mission_main_chute_cd = float(self._cfg.main_chute_cd)
        self._mission_engine_t_max_n = float(self._cfg.engine.t_max_n)
        self._mission_engine_isp_s = float(self._cfg.engine.isp_s)
        self._reset_state()
        tx, tz = self._nearest_land_point_m(0.0, 0.0)
        self._target = self._make_target_at(tx, tz)
        self._csv_file: Optional[TextIO] = None
        self._csv_writer: Optional[Any] = None
        self._csv_log_path: Optional[str] = None

    # -------------------------
    # Base state (private)
    # -------------------------
    def _reset_state(self) -> None:
        # Simulation time from start (telemetry, phase logic).
        self._time_s = 0.0
        self._wind_gust_x_mps = 0.0
        self._wind_gust_z_mps = 0.0
        self._rng = np.random.default_rng(self._physics_seed)

        # Absolute vertical position [m]: z_msl = terrain(x,z) + height above it.
        # Terrain from world.height_m_at; height above local surface = z_msl - terrain (see altitude_m).
        terrain0 = float(self._world.height_m_at(0.0, 0.0))
        self._z_msl_m = float(self._mission_entry_start_altitude_m) + terrain0

        body_entry = replace(self._cfg.body, entry_speed_mps=float(self._mission_entry_speed_mps))
        vv0, vx0, vz0 = entry_velocity_inertial_mps(body_entry)
        self._vertical_speed_mps = vv0
        self._speed_x_mps = vx0
        self._speed_z_mps = vz0

        # World horizontal position [m] for minimap, target, surface type.
        self._pos_x_m = 0.0
        self._pos_z_m = 0.0

        # Dry mass [kg]; drives m_total_kg and accelerations.
        self._dry_mass_kg = float(self._mission_dry_mass_kg)

        # Heatshield mass [kg]; removed from dry_mass_kg on jettison.
        self._heatshield_mass_kg = float(self._mission_heatshield_mass_kg)

        # Fuel mass [kg]; limits thrust and total mass.
        self._fuel_kg = float(self._mission_fuel_kg)

        # Base drag coefficient (no chutes/jettison); used for drag.
        self._cd_base = float(self._mission_cd_base)

        # Reference area [m^2]; modified by chutes/heatshield via derived getters.
        self._a_ref_m2 = float(self._mission_a_ref_m2)

        # System flags: affect aero and which actions are allowed next.
        self._heatshield_jettisoned = False
        self._drogue_deployed = False
        self._main_deployed = False
        self._chute_jettisoned = False

        # Engine and throttle: thrust, fuel use, final landing.
        self._engine_on = False
        self._throttle_0_1 = 0.0

        # Internal bay temperature [°C]; fail limits (Tmin/Tmax) and HUD.
        self._internal_temp_c = -20.0

        # Heatshield skin [°C]: start at free-stream + stagnation/friction offset (not a fixed cold soak
        # below T_ext — that was wrong once the probe is already in the atmosphere at high speed).
        self._invalidate_cache()
        hcfg = self._cfg.heatshield_thermal
        t_ext0 = float(self.atm_temp_ext_c)
        rho0 = float(self.atm_density_kg_m3)
        v0 = max(0.0, float(self.air_rel_speed_mps))
        dT0 = heatshield_skin_dTdt(t_ext0, t_ext0, rho0, v0, hcfg)
        rho_r = max(1e-15, float(hcfg.rho_ambient_ref_kg_m3))
        gas_w0 = rho0 / (rho0 + rho_r)
        k_eff = float(hcfg.k_ambient_1ps) * gas_w0 + float(hcfg.k_ambient_radiative_1ps)
        delta_stag = float(dT0) / max(1e-12, k_eff)
        self._heatshield_skin_temp_c = float(
            np.clip(t_ext0 + delta_stag, float(hcfg.t_min_c), float(hcfg.t_max_c))
        )

        # Outcome: `failed` means a fault was recorded; `result` is final (success/failure).
        self._result = SimResult.RUNNING
        self._failed = False
        self._failure_reasons: list[str] = []
        self._failure_reason_keys_seen: set[str] = set()
        self._landing_finished: bool = False
        self._water_landed: bool = False
        self._t_main_deployed_s: Optional[float] = None
        # Snapshot at step() entry; soft landing uses these (before Euler integration).
        self._vv_step_start_mps = 0.0
        self._speed_x_step_start_mps = 0.0
        self._speed_z_step_start_mps = 0.0
        self._last_physics_dt_s = 0.0
        self._telemetry_history.clear()
        self._plot_action_log.clear()

    def reset(self) -> None:
        self._close_csv_log()
        self._reset_state()
        tx, tz = self._nearest_land_point_m(0.0, 0.0)
        self._target = self._make_target_at(tx, tz)
        self._invalidate_cache()

    @property
    def mission_entry_start_altitude_m(self) -> float:
        """MSL entry altitude above local terrain anchor [m] (mission setup)."""
        return float(self._mission_entry_start_altitude_m)

    @property
    def mission_entry_speed_mps(self) -> float:
        """Entry speed magnitude used for initial velocity vector [m/s]."""
        return float(self._mission_entry_speed_mps)

    @property
    def mission_fuel_kg(self) -> float:
        """Initial fuel load for the next / current reset [kg]."""
        return float(self._mission_fuel_kg)

    @property
    def mission_dry_mass_kg(self) -> float:
        return float(self._mission_dry_mass_kg)

    @property
    def mission_heatshield_mass_kg(self) -> float:
        return float(self._mission_heatshield_mass_kg)

    @property
    def mission_a_ref_m2(self) -> float:
        return float(self._mission_a_ref_m2)

    @property
    def mission_cd_base(self) -> float:
        return float(self._mission_cd_base)

    @property
    def mission_drogue_area_m2(self) -> float:
        return float(self._mission_drogue_area_m2)

    @property
    def mission_drogue_cd(self) -> float:
        return float(self._mission_drogue_cd)

    @property
    def mission_main_chute_area_m2(self) -> float:
        return float(self._mission_main_chute_area_m2)

    @property
    def mission_main_chute_cd(self) -> float:
        return float(self._mission_main_chute_cd)

    @property
    def mission_engine_t_max_n(self) -> float:
        return float(self._mission_engine_t_max_n)

    @property
    def mission_engine_isp_s(self) -> float:
        return float(self._mission_engine_isp_s)

    def set_mission_parameters(
        self,
        entry_start_altitude_m: float,
        entry_speed_mps: float,
        fuel_kg: float,
        dry_mass_kg: float,
        heatshield_mass_kg: float,
        a_ref_m2: float,
        cd_base: float,
        drogue_area_m2: float,
        drogue_cd: float,
        main_chute_area_m2: float,
        main_chute_cd: float,
        engine_t_max_n: float,
        engine_isp_s: float,
    ) -> None:
        """
        Store vehicle + entry conditions applied on the next `reset()`.

        Altitude: height above terrain at origin (same convention as BodyConfig.entry_start_altitude_m).
        Speed: magnitude of entry velocity; direction follows entry_flight_path_angle_from_horizontal_deg.
        """
        h = float(entry_start_altitude_m)
        v = float(entry_speed_mps)
        f = float(fuel_kg)
        d = float(dry_mass_kg)
        hs = float(heatshield_mass_kg)
        ar = float(a_ref_m2)
        cd0 = float(cd_base)
        da = float(drogue_area_m2)
        dcd = float(drogue_cd)
        ma = float(main_chute_area_m2)
        mcd = float(main_chute_cd)
        tmax = float(engine_t_max_n)
        isp = float(engine_isp_s)
        if not (30_000.0 <= h <= 3_000_000.0):
            raise ValueError("entry altitude must be between 30 km and 3000 km (MSL-style)")
        if not (500.0 <= v <= 12_000.0):
            raise ValueError("entry speed must be between 500 m/s and 12 km/s")
        if not (0.0 <= f <= 500.0):
            raise ValueError("fuel must be between 0 and 500 kg")
        if not (10.0 <= d <= 5000.0):
            raise ValueError("dry mass must be between 10 and 5000 kg")
        if not (0.0 <= hs <= d):
            raise ValueError("heatshield mass must be between 0 and dry mass")
        if not (0.2 <= ar <= 80.0):
            raise ValueError("reference area must be between 0.2 and 80 m²")
        if not (0.1 <= cd0 <= 5.0):
            raise ValueError("base Cd must be between 0.1 and 5")
        if not (0.5 <= da <= 200.0):
            raise ValueError("drogue area must be between 0.5 and 200 m²")
        if not (0.3 <= dcd <= 5.0):
            raise ValueError("drogue Cd must be between 0.3 and 5")
        if not (1.0 <= ma <= 600.0):
            raise ValueError("main chute area must be between 1 and 600 m²")
        if not (0.3 <= mcd <= 5.0):
            raise ValueError("main chute Cd must be between 0.3 and 5")
        if not (0.0 <= tmax <= 80_000.0):
            raise ValueError("engine max thrust must be between 0 and 80000 N")
        if not (1.0 <= isp <= 600.0):
            raise ValueError("engine Isp must be between 1 and 600 s")
        self._mission_entry_start_altitude_m = h
        self._mission_entry_speed_mps = v
        self._mission_fuel_kg = f
        self._mission_dry_mass_kg = d
        self._mission_heatshield_mass_kg = hs
        self._mission_a_ref_m2 = ar
        self._mission_cd_base = cd0
        self._mission_drogue_area_m2 = da
        self._mission_drogue_cd = dcd
        self._mission_main_chute_area_m2 = ma
        self._mission_main_chute_cd = mcd
        self._mission_engine_t_max_n = tmax
        self._mission_engine_isp_s = isp

    def set_mission_start_params(
        self,
        entry_start_altitude_m: float,
        entry_speed_mps: float,
        fuel_kg: float,
    ) -> None:
        """Backward-compatible wrapper: only changes entry altitude/speed/fuel."""
        self.set_mission_parameters(
            entry_start_altitude_m,
            entry_speed_mps,
            fuel_kg,
            self._mission_dry_mass_kg,
            self._mission_heatshield_mass_kg,
            self._mission_a_ref_m2,
            self._mission_cd_base,
            self._mission_drogue_area_m2,
            self._mission_drogue_cd,
            self._mission_main_chute_area_m2,
            self._mission_main_chute_cd,
            self._mission_engine_t_max_n,
            self._mission_engine_isp_s,
        )

    def _nearest_land_point_m(self, x0: float, z0: float) -> tuple[float, float]:
        if self._world.surface_type_at(float(x0), float(z0)) != SurfaceType.LAKE:
            return float(x0), float(z0)
        for ring in range(1, 900):
            r = float(ring) * 320.0
            for k in range(28):
                ang = (2.0 * math.pi * k) / 28.0
                x = float(x0) + r * math.cos(ang)
                z = float(z0) + r * math.sin(ang)
                if self._world.surface_type_at(x, z) == SurfaceType.LAND:
                    return x, z
        return float(x0), float(z0)

    def _make_target_at(self, x_m: float, z_m: float) -> Target:
        st = self._world.surface_type_at(float(x_m), float(z_m))
        return Target(float(x_m), float(z_m), st)

    # -------------------------
    # helpers: cache
    # -------------------------
    def _invalidate_cache(self) -> None:
        self._tick_id += 1
        self._cache.clear()

    def _get_cached(self, key: str):
        item = self._cache.get(key)
        if item is None:
            return None
        tick, val = item
        return val if tick == self._tick_id else None

    def _set_cached(self, key: str, val: object) -> None:
        self._cache[key] = (self._tick_id, val)

    # -------------------------
    # Constants / config (read via getters only)
    # -------------------------
    @property
    def g_titan_mps2(self) -> float:
        return float(self._cfg.body.g_mps2)

    @property
    def g0_mps2(self) -> float:
        return float(self._cfg.engine.g0_mps2)

    @property
    def engine_tmax_n(self) -> float:
        return float(self._mission_engine_t_max_n)

    @property
    def engine_isp_s(self) -> float:
        return float(self._mission_engine_isp_s)

    @property
    def rho_surface_kg_m3(self) -> float:
        return float(self._cfg.body.rho_surface_kg_m3)

    @property
    def scale_height_m(self) -> float:
        return float(self._cfg.body.scale_height_m)

    @property
    def t_int_min_c(self) -> float:
        return float(self._cfg.thermal.t_min_c)

    @property
    def t_int_max_c(self) -> float:
        return float(self._cfg.thermal.t_max_c)

    @property
    def max_overload_g(self) -> float:
        return 15.9

    @property
    def touchdown_v_land_mps(self) -> float:
        return 5.0

    @property
    def touchdown_v_lake_mps(self) -> float:
        return 10.0

    @property
    def touchdown_v_hor_mps(self) -> float:
        return 3.0

    @property
    def water_landed(self) -> bool:
        return self._water_landed

    @property
    def drogue_min_alt_m(self) -> float:
        return float(self._cfg.drogue_floor_alt_m)

    @property
    def main_max_alt_m(self) -> float:
        return float(self._cfg.parachute_main_max_deploy_alt_m)

    @property
    def chute_jettison_max_alt_m(self) -> float:
        """Номинальная высота сброса по ТЗ в конфиге (для API/подсказок; не гейт симулятора)."""
        return float(self._cfg.parachute_jettison_max_alt_m)

    # -------------------------
    # target
    # -------------------------
    @property
    def target_x_m(self) -> float:
        return float(self._target.x_m)

    @property
    def target_z_m(self) -> float:
        return float(self._target.z_m)

    @property
    def target_surface_type(self) -> SurfaceType:
        return self._target.surface_type

    def set_target_world(self, x_m: float, z_m: float) -> None:
        self._target = self._make_target_at(x_m, z_m)

    def _close_csv_log(self) -> None:
        if self._csv_file is not None:
            try:
                self._csv_file.flush()
            except OSError:
                pass
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
            self._csv_log_path = None

    @property
    def csv_logging_enabled(self) -> bool:
        return self._csv_file is not None

    @property
    def csv_log_path(self) -> Optional[str]:
        return self._csv_log_path

    def set_csv_logging(self, enabled: bool) -> None:
        if enabled:
            if self._csv_file is not None:
                return
            root = self._resource_root if self._resource_root is not None else Path.cwd()
            log_dir = root / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            path = log_dir / f"trajectory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            f = open(path, "w", newline="", encoding="utf-8")
            w = csv.writer(f)
            w.writerow(
                [
                    "time_s",
                    "altitude_m",
                    "pos_x_m",
                    "pos_z_m",
                    "target_x_m",
                    "target_z_m",
                    "v_vert_mps",
                    "v_hor_mps",
                    "speed_x_mps",
                    "speed_z_mps",
                    "fuel_kg",
                    "engine_on",
                    "throttle",
                    "phase",
                    "result",
                    "g_load",
                    "pressure_bar",
                    "heatshield_skin_c",
                ]
            )
            self._csv_file = f
            self._csv_writer = w
            self._csv_log_path = str(path)
        else:
            self._close_csv_log()

    def _append_csv_log_row(self) -> None:
        w = self._csv_writer
        if w is None:
            return
        w.writerow(
            [
                f"{self.time_s:.6f}",
                f"{self.altitude_m:.3f}",
                f"{self.pos_x_m:.3f}",
                f"{self.pos_z_m:.3f}",
                f"{self.target_x_m:.3f}",
                f"{self.target_z_m:.3f}",
                f"{self.vertical_speed_mps:.4f}",
                f"{self.horizontal_speed_mps:.4f}",
                f"{self.speed_x_mps:.4f}",
                f"{self.speed_z_mps:.4f}",
                f"{self.fuel_kg:.6f}",
                int(self.engine_on),
                f"{self.throttle_0_1:.4f}",
                self.phase,
                self.result.value,
                f"{self.g_load:.4f}",
                f"{self.atm_pressure_bar:.6f}",
                f"{self.heatshield_skin_temp_c:.3f}",
            ]
        )

    # -------------------------
    # Mutable state (setters where needed)
    # -------------------------
    @property
    def time_s(self) -> float:
        return float(self._time_s)

    def _terrain_height_at_probe(self) -> float:
        c = self._get_cached("terrain_h_probe")
        if c is not None:
            return float(c)
        h = float(self._world.height_m_at(float(self._pos_x_m), float(self._pos_z_m)))
        self._set_cached("terrain_h_probe", h)
        return h

    @property
    def altitude_m(self) -> float:
        return float(self._z_msl_m - self._terrain_height_at_probe())

    @altitude_m.setter
    def altitude_m(self, value: float) -> None:
        self._invalidate_cache()
        th = float(self._world.height_m_at(float(self._pos_x_m), float(self._pos_z_m)))
        self._z_msl_m = float(value) + th

    @property
    def vertical_speed_mps(self) -> float:
        return float(self._vertical_speed_mps)

    @vertical_speed_mps.setter
    def vertical_speed_mps(self, value: float) -> None:
        self._vertical_speed_mps = float(value)
        self._invalidate_cache()

    @property
    def pos_x_m(self) -> float:
        return float(self._pos_x_m)

    @pos_x_m.setter
    def pos_x_m(self, value: float) -> None:
        self._pos_x_m = float(value)
        self._invalidate_cache()

    @property
    def pos_z_m(self) -> float:
        return float(self._pos_z_m)

    @pos_z_m.setter
    def pos_z_m(self, value: float) -> None:
        self._pos_z_m = float(value)
        self._invalidate_cache()

    @property
    def speed_x_mps(self) -> float:
        return float(self._speed_x_mps)

    @speed_x_mps.setter
    def speed_x_mps(self, value: float) -> None:
        self._speed_x_mps = float(value)
        self._invalidate_cache()

    @property
    def speed_z_mps(self) -> float:
        return float(self._speed_z_mps)

    @speed_z_mps.setter
    def speed_z_mps(self, value: float) -> None:
        self._speed_z_mps = float(value)
        self._invalidate_cache()

    @property
    def dry_mass_kg(self) -> float:
        return float(self._dry_mass_kg)

    @dry_mass_kg.setter
    def dry_mass_kg(self, value: float) -> None:
        self._dry_mass_kg = float(value)
        self._invalidate_cache()

    @property
    def heatshield_mass_kg(self) -> float:
        return float(self._heatshield_mass_kg)

    @heatshield_mass_kg.setter
    def heatshield_mass_kg(self, value: float) -> None:
        self._heatshield_mass_kg = float(value)
        self._invalidate_cache()

    @property
    def fuel_kg(self) -> float:
        return float(self._fuel_kg)

    @fuel_kg.setter
    def fuel_kg(self, value: float) -> None:
        self._fuel_kg = float(value)
        self._invalidate_cache()

    @property
    def m_total_kg(self) -> float:
        return max(1.0, float(self.dry_mass_kg + self.fuel_kg))

    @property
    def cd_base(self) -> float:
        return float(self._cd_base)

    @property
    def a_ref_m2(self) -> float:
        return float(self._a_ref_m2)

    @property
    def heatshield_jettisoned(self) -> bool:
        return bool(self._heatshield_jettisoned)

    @property
    def drogue_deployed(self) -> bool:
        return bool(self._drogue_deployed)

    @property
    def main_deployed(self) -> bool:
        return bool(self._main_deployed)

    @property
    def chute_jettisoned(self) -> bool:
        return bool(self._chute_jettisoned)

    @property
    def engine_on(self) -> bool:
        return bool(self._engine_on)

    @property
    def throttle_0_1(self) -> float:
        return float(self._throttle_0_1)

    @property
    def internal_temp_c(self) -> float:
        return float(self._internal_temp_c)

    @internal_temp_c.setter
    def internal_temp_c(self, value: float) -> None:
        self._internal_temp_c = float(value)
        self._invalidate_cache()

    @property
    def heatshield_skin_temp_c(self) -> float:
        """Ablator / outer skin temperature while heatshield is on; frozen after jettison."""
        return float(self._heatshield_skin_temp_c)

    @property
    def result(self) -> SimResult:
        return self._result

    @property
    def failed(self) -> bool:
        return bool(self._failed)

    @property
    def failure_reason(self) -> str:
        """Last recorded failure category id (see `failure_reasons`); empty if none."""
        return self._failure_reasons[-1] if self._failure_reasons else ""

    @property
    def failure_reasons(self) -> tuple[str, ...]:
        """Unique failure category ids, e.g. ``overload``, ``t_int_max`` (UI translates)."""
        return tuple(self._failure_reasons)

    @staticmethod
    def _normalize_failure_reason_text(reason: str) -> str:
        return " ".join(str(reason).split())

    # -------------------------
    # commands / events + validation
    # -------------------------
    @property
    def can_heatshield_jettison(self) -> bool:
        if self.heatshield_jettisoned:
            return False
        return float(self.mach_number) < float(self._cfg.heatshield_jettison_max_mach)

    @property
    def can_drogue(self) -> bool:
        return (
            (not self.drogue_deployed)
            and self.heatshield_jettisoned
            and (self.altitude_m > self.drogue_min_alt_m)
        )

    @property
    def can_main(self) -> bool:
        return (not self.main_deployed) and self.drogue_deployed and (self.altitude_m < self.main_max_alt_m)

    @property
    def can_chute_jettison(self) -> bool:
        """Разрешение на сброс купола: только логика цепочки (основной уже раскрыт). Высоту задаёт автопилот / игрок."""
        return (not self.chute_jettisoned) and self.main_deployed

    def request_heatshield_jettison(self) -> bool:
        if not self.can_heatshield_jettison:
            return False
        drop = float(self.heatshield_mass_kg)
        if drop > 0.0:
            self._dry_mass_kg = max(0.0, float(self.dry_mass_kg) - drop)
            self._heatshield_mass_kg = 0.0
        self._heatshield_jettisoned = True
        self._invalidate_cache()
        return True

    def request_drogue(self) -> bool:
        if not self.can_drogue:
            return False
        self._drogue_deployed = True
        self._invalidate_cache()
        return True

    def request_main(self) -> bool:
        if not self.can_main:
            return False
        self._main_deployed = True
        self._t_main_deployed_s = float(self.time_s)
        self._invalidate_cache()
        return True

    def request_chute_jettison(self) -> bool:
        if not self.can_chute_jettison:
            return False
        self._chute_jettisoned = True
        self._invalidate_cache()
        return True

    def set_engine(self, on: bool) -> None:
        self._engine_on = bool(on)
        if not self._engine_on:
            self._throttle_0_1 = 0.0
        self._invalidate_cache()

    def set_throttle(self, value_0_1: float) -> None:
        self._throttle_0_1 = float(np.clip(float(value_0_1), 0.0, 1.0)) if self.engine_on else 0.0
        self._invalidate_cache()

    # -------------------------
    # derived: environment
    # -------------------------
    @property
    def wind_x_mps(self) -> float:
        cached = self._get_cached("wind_tot")
        if cached is not None:
            return float(cached[0])  # type: ignore[index]
        mx, mz = wind_mean_vec_mps(self.altitude_m)
        tx = float(mx + self._wind_gust_x_mps)
        tz = float(mz + self._wind_gust_z_mps)
        self._set_cached("wind_tot", (tx, tz))
        return tx

    @property
    def wind_z_mps(self) -> float:
        cached = self._get_cached("wind_tot")
        if cached is not None:
            return float(cached[1])  # type: ignore[index]
        mx, mz = wind_mean_vec_mps(self.altitude_m)
        tx = float(mx + self._wind_gust_x_mps)
        tz = float(mz + self._wind_gust_z_mps)
        self._set_cached("wind_tot", (tx, tz))
        return tz

    def wind_vec_mps_at(self, h_m: float) -> tuple[float, float]:
        """Mean wind at h plus current gust (column turbulence; same gust at all h)."""
        mx, mz = wind_mean_vec_mps(float(h_m))
        return float(mx + self._wind_gust_x_mps), float(mz + self._wind_gust_z_mps)

    def _advance_wind_gust(self, dt: float) -> None:
        wcfg = self._cfg.wind
        if not wcfg.turbulence_enabled or dt <= 0.0:
            return
        sig_obs = float(wind_sigma_zonal_mps(float(self.altitude_m)))
        sigma = max(float(wcfg.sigma_floor_mps), float(wcfg.sigma_scale_from_dwe) * sig_obs)
        tau = max(1e-6, float(wcfg.ou_tau_s))
        phi = math.exp(-dt / tau)
        root = math.sqrt(max(0.0, 1.0 - phi * phi))
        gx = float(self._wind_gust_x_mps)
        gz = float(self._wind_gust_z_mps)
        zfac = 0.68
        self._wind_gust_x_mps = phi * gx + sigma * root * float(self._rng.standard_normal())
        self._wind_gust_z_mps = phi * gz + zfac * sigma * root * float(self._rng.standard_normal())

    @property
    def atm_density_kg_m3(self) -> float:
        cached = self._get_cached("atm")
        if cached is not None:
            return float(cached[0])  # type: ignore[index]
        a = sample_atmosphere(self.altitude_m, self._cfg.body, self._atm_table)
        self._set_cached("atm", (float(a.rho), float(a.t_ext_c), float(a.p_bar)))
        return float(a.rho)

    @property
    def atm_temp_ext_c(self) -> float:
        cached = self._get_cached("atm")
        if cached is not None:
            return float(cached[1])  # type: ignore[index]
        a = sample_atmosphere(self.altitude_m, self._cfg.body, self._atm_table)
        self._set_cached("atm", (float(a.rho), float(a.t_ext_c), float(a.p_bar)))
        return float(a.t_ext_c)

    @property
    def atm_pressure_bar(self) -> float:
        cached = self._get_cached("atm")
        if cached is not None:
            return float(cached[2])  # type: ignore[index]
        a = sample_atmosphere(self.altitude_m, self._cfg.body, self._atm_table)
        self._set_cached("atm", (float(a.rho), float(a.t_ext_c), float(a.p_bar)))
        return float(a.p_bar)

    # -------------------------
    # derived: relative air velocity
    # -------------------------
    @property
    def air_rel_speed_x_mps(self) -> float:
        return float(self.speed_x_mps - self.wind_x_mps)

    @property
    def air_rel_speed_z_mps(self) -> float:
        return float(self.speed_z_mps - self.wind_z_mps)

    @property
    def air_rel_speed_vert_mps(self) -> float:
        return float(self.vertical_speed_mps)

    @property
    def air_rel_speed_mps(self) -> float:
        vx = float(self.air_rel_speed_x_mps)
        vz = float(self.air_rel_speed_z_mps)
        vv = float(self.air_rel_speed_vert_mps)
        return float(math.sqrt(vx * vx + vz * vz + vv * vv))

    @property
    def speed_of_sound_mps(self) -> float:
        T_K = float(self.atm_temp_ext_c) + 273.15
        T_K = max(1.0, T_K)
        g = float(self._cfg.atmosphere_gamma)
        R = float(self._cfg.atmosphere_R_specific_j_kg_k)
        return float(math.sqrt(max(0.0, g * R * T_K)))

    @property
    def mach_number(self) -> float:
        a = float(self.speed_of_sound_mps)
        v = float(self.air_rel_speed_mps)
        if a <= 1e-6:
            return float("inf") if v > 1e-6 else 0.0
        return float(v / a)

    # -------------------------
    # derived: aero parameters (Cd, area)
    # -------------------------
    @property
    def cd(self) -> float:
        cached = self._get_cached("aero")
        if cached is not None:
            return float(cached[0])  # type: ignore[index]
        cd, area = self._compute_aero()
        self._set_cached("aero", (float(cd), float(area)))
        return float(cd)

    @property
    def ref_area_m2(self) -> float:
        cached = self._get_cached("aero")
        if cached is not None:
            return float(cached[1])  # type: ignore[index]
        cd, area = self._compute_aero()
        self._set_cached("aero", (float(cd), float(area)))
        return float(area)

    def _compute_aero(self) -> tuple[float, float]:
        # reuse existing logic by constructing a minimal state-like object
        class _S:
            pass

        s = _S()
        s.cd_base = self.cd_base
        s.a_ref_m2 = self.a_ref_m2
        s.drogue_area_m2 = float(self._mission_drogue_area_m2)
        s.drogue_cd = float(self._mission_drogue_cd)
        s.main_chute_area_m2 = float(self._mission_main_chute_area_m2)
        s.main_chute_cd = float(self._mission_main_chute_cd)
        s.drogue_deployed = self.drogue_deployed
        s.main_deployed = self.main_deployed
        s.chute_jettisoned = self.chute_jettisoned
        s.heatshield_jettisoned = self.heatshield_jettisoned
        aero = aero_params_for_state(s)  # type: ignore[arg-type]
        return float(aero.cd), float(aero.area_m2)

    # -------------------------
    # derived: forces, acceleration, overload
    # -------------------------
    @property
    def thrust_n(self) -> float:
        if (not self.engine_on) or self.throttle_0_1 <= 0.0 or self.fuel_kg <= 0.0:
            return 0.0
        return float(self.engine_tmax_n) * float(np.clip(self.throttle_0_1, 0.0, 1.0))

    @property
    def dynamic_pressure_pa(self) -> float:
        cached = self._get_cached("forces")
        if cached is not None:
            return float(cached["q_dyn_pa"])  # type: ignore[index]
        self._compute_forces_cached()
        cached = self._get_cached("forces")
        return float(cached["q_dyn_pa"])  # type: ignore[index]

    def _compute_forces_cached(self) -> None:
        # forces cache contains: drag components, q_dyn, accel components, accel mag
        fx, fz, fv, vmag = drag_force_vector_n(
            self.atm_density_kg_m3,
            self.cd,
            self.ref_area_m2,
            v_rel_x_mps=self.air_rel_speed_x_mps,
            v_rel_z_mps=self.air_rel_speed_z_mps,
            v_rel_vert_mps=self.air_rel_speed_vert_mps,
        )
        m = self.m_total_kg
        g_titan = float(self.g_titan_mps2)
        if (
            (not self._heatshield_jettisoned)
            and float(self.atm_density_kg_m3) < float(self._cfg.body.entry_low_density_rho_kg_m3)
        ):
            g_titan *= float(self._cfg.body.entry_low_density_gravity_scale)
        f_grav_vert = -m * g_titan
        f_thrust_vert = self.thrust_n
        f_vert = float(f_grav_vert + fv + f_thrust_vert)
        q_dyn = 0.5 * float(self.atm_density_kg_m3) * float(vmag * vmag)
        a_vert = f_vert / m
        a_x = float(fx) / m
        a_z = float(fz) / m
        # Proper (specific) acceleration: non-gravitational part of F/m — what a 3-axis
        # accelerometer / crew "g" approximates. Horizontal: only drag; vertical: drag+thrust
        # (gravity removed: a_vert − f_grav/m = a_vert + g_titan with constant g, +up).
        g_t = g_titan
        ap_x = float(a_x)
        ap_z = float(a_z)
        ap_vert = float(a_vert + g_t)
        a_proper_mag = float(math.sqrt(ap_x * ap_x + ap_z * ap_z + ap_vert * ap_vert))
        self._set_cached(
            "forces",
            {
                "f_drag_x_n": float(fx),
                "f_drag_z_n": float(fz),
                "f_drag_vert_n": float(fv),
                "f_grav_vert_n": float(f_grav_vert),
                "f_thrust_vert_n": float(f_thrust_vert),
                "q_dyn_pa": float(q_dyn),
                "a_vert_mps2": float(a_vert),
                "a_x_mps2": float(a_x),
                "a_z_mps2": float(a_z),
                "a_mag_mps2": float(a_proper_mag),
            },
        )

    @property
    def drag_force_x_n(self) -> float:
        cached = self._get_cached("forces")
        if cached is None:
            self._compute_forces_cached()
            cached = self._get_cached("forces")
        return float(cached["f_drag_x_n"])  # type: ignore[index]

    @property
    def drag_force_z_n(self) -> float:
        cached = self._get_cached("forces")
        if cached is None:
            self._compute_forces_cached()
            cached = self._get_cached("forces")
        return float(cached["f_drag_z_n"])

    @property
    def drag_force_vert_n(self) -> float:
        cached = self._get_cached("forces")
        if cached is None:
            self._compute_forces_cached()
            cached = self._get_cached("forces")
        return float(cached["f_drag_vert_n"])  # type: ignore[index]

    @property
    def accel_vert_mps2(self) -> float:
        cached = self._get_cached("forces")
        if cached is None:
            self._compute_forces_cached()
            cached = self._get_cached("forces")
        return float(cached["a_vert_mps2"])  # type: ignore[index]

    @property
    def accel_x_mps2(self) -> float:
        cached = self._get_cached("forces")
        if cached is None:
            self._compute_forces_cached()
            cached = self._get_cached("forces")
        return float(cached["a_x_mps2"])  # type: ignore[index]

    @property
    def accel_z_mps2(self) -> float:
        cached = self._get_cached("forces")
        if cached is None:
            self._compute_forces_cached()
            cached = self._get_cached("forces")
        return float(cached["a_z_mps2"])  # type: ignore[index]

    @property
    def accel_mag_mps2(self) -> float:
        """Magnitude of proper (specific) acceleration [m/s²], not raw |F_total|/m."""
        cached = self._get_cached("forces")
        if cached is None:
            self._compute_forces_cached()
            cached = self._get_cached("forces")
        return float(cached["a_mag_mps2"])  # type: ignore[index]

    @property
    def g_load(self) -> float:
        """Load factor: |a_proper| / g0 (Earth); at contact, adds impact estimate |v|/τ (calibrated τ vs surface)."""
        g0 = float(self.g0_mps2)
        base = float(self.accel_mag_mps2) / g0
        dt = float(self._last_physics_dt_s)
        if dt > 0.0 and float(self.altitude_m) <= 0.0:
            vx = float(self.speed_x_mps)
            vz = float(self.speed_z_mps)
            vv = float(self.vertical_speed_mps)
            vm = float(math.sqrt(vx * vx + vz * vz + vv * vv))
            if vm > 1e-9:
                vref = float(self._cfg.impact_calib_touchdown_speed_mps)
                gref = float(self._cfg.impact_calib_touchdown_g)
                g0_eng = float(self._cfg.engine.g0_mps2)
                tau_land = vref / (gref * g0_eng)
                if self.surface_type_under_probe == SurfaceType.LAKE:
                    tau0 = tau_land * float(self._cfg.impact_lake_tau_multiplier)
                else:
                    tau0 = tau_land
                a_stop = vm / tau0
                base = max(base, a_stop / g0)
        return float(base)

    # -------------------------
    # derived: navigation/surface/phase
    # -------------------------
    @property
    def horizontal_speed_mps(self) -> float:
        return float(math.hypot(self.speed_x_mps, self.speed_z_mps))

    @property
    def distance_to_target_m(self) -> float:
        dx = float(self.target_x_m - self.pos_x_m)
        dz = float(self.target_z_m - self.pos_z_m)
        return float(math.hypot(dx, dz))

    @property
    def surface_type_under_probe(self) -> SurfaceType:
        return self._world.surface_type_at(self.pos_x_m, self.pos_z_m)

    @property
    def world(self) -> WorldGen:
        return self._world

    @property
    def phase(self) -> str:
        if self.result == SimResult.SUCCESS or self._landing_finished:
            return "end"
        if not self.heatshield_jettisoned:
            return "entry"
        if self.main_deployed and (not self.chute_jettisoned):
            if self._t_main_deployed_s is not None:
                elapsed = self.time_s - self._t_main_deployed_s
                if elapsed < float(self._cfg.science_descent_min_s):
                    return "science_descent"
            return "main_chute"
        if self.drogue_deployed:
            return "drogue_chute"
        if self.engine_on:
            return "powered_descent"
        return "descent"

    # -------------------------
    # simulation step
    # -------------------------
    def step(self, dt_s: float) -> None:
        # Touchdown (success or failure): no further physics integration.
        if self._landing_finished:
            return

        dt = float(dt_s)
        if dt <= 0.0:
            return
        self._last_physics_dt_s = dt

        # new tick; derived caches must be recomputed against current state
        self._invalidate_cache()
        self._advance_wind_gust(dt)

        # Snapshot probe velocity immediately before the same Euler update (matches z += vv*dt).
        self._vv_step_start_mps = float(self._vertical_speed_mps)
        self._speed_x_step_start_mps = float(self._speed_x_mps)
        self._speed_z_step_start_mps = float(self._speed_z_mps)

        # Explicit Euler: update position with velocity at step start, then v += a*dt.
        a_vert = float(self.accel_vert_mps2)
        a_x = float(self.accel_x_mps2)
        a_z = float(self.accel_z_mps2)

        vv = float(self.vertical_speed_mps)
        vx = float(self.speed_x_mps)
        vz = float(self.speed_z_mps)
        self._z_msl_m = float(self._z_msl_m + vv * dt)
        self._pos_x_m = float(self.pos_x_m + vx * dt)
        self._pos_z_m = float(self.pos_z_m + vz * dt)
        self._vertical_speed_mps = float(vv + a_vert * dt)
        self._speed_x_mps = float(vx + a_x * dt)
        self._speed_z_mps = float(vz + a_z * dt)

        # fuel burn (based on thrust)
        dm = fuel_burn_kg(
            self._cfg,
            thrust_n=float(self.thrust_n),
            dt_s=dt,
            isp_s=float(self._mission_engine_isp_s),
        )
        if dm > 0.0:
            self._fuel_kg = max(0.0, float(self.fuel_kg) - float(dm))
            if self._fuel_kg <= 0.0:
                self._throttle_0_1 = 0.0
                self._engine_on = False

        # thermal: skin first (friction + equilibration to T_ext), then bays (coupled to skin while shield on)
        hs_on = not self._heatshield_jettisoned
        if hs_on:
            self._heatshield_skin_temp_c = heatshield_skin_step(
                float(self._heatshield_skin_temp_c),
                float(self.atm_temp_ext_c),
                float(self.atm_density_kg_m3),
                float(self.air_rel_speed_mps),
                dt,
                self._cfg.heatshield_thermal,
            )
        thermal_relaxation_step(
            self._cfg,
            _ThermalStateProxy(self),
            t_ext_c=float(self.atm_temp_ext_c),
            q_dyn_pa=float(self.dynamic_pressure_pa),
            dt_s=dt,
            t_skin_c=float(self._heatshield_skin_temp_c),
            heatshield_attached=hs_on,
            rho_kg_m3=float(self.atm_density_kg_m3),
        )

        self._time_s = float(self.time_s + dt)

        # After state update, invalidate derived caches again.
        self._invalidate_cache()
        self._telemetry_history.append(
            {
                "t_s": float(self.time_s),
                "altitude_m": float(self.altitude_m),
                "v_vert_mps": float(self.vertical_speed_mps),
                "g_load": float(self.g_load),
                "p_bar": float(self.atm_pressure_bar),
                "hs": 1.0 if self.heatshield_jettisoned else 0.0,
                "dr": 1.0 if self.drogue_deployed else 0.0,
                "mn": 1.0 if self.main_deployed else 0.0,
                "cj": 1.0 if self.chute_jettisoned else 0.0,
                "eng": 1.0 if self.engine_on else 0.0,
            }
        )
        self._check_end_conditions()
        self._append_csv_log_row()

    def _fail(self, reason: str) -> None:
        self._failed = True
        norm = self._normalize_failure_reason_text(reason)
        key = self._failure_reason_semantic_key(norm)
        if key in self._failure_reason_keys_seen:
            if self._result == SimResult.RUNNING:
                self._result = SimResult.FAILURE
            return
        self._failure_reason_keys_seen.add(key)
        self._failure_reasons.append(key)
        # Failure is non-terminal in flight; after touchdown, physics stops (see _landing_finished).
        if self._result == SimResult.RUNNING:
            self._result = SimResult.FAILURE

    @staticmethod
    def _failure_reason_semantic_key(reason: str) -> str:
        """One key per fault *category* (no degrees, no g margin). Unknown text → key = normalized string."""
        s = PhysicsModel._normalize_failure_reason_text(reason)
        sl = s.lower()
        if sl in (
            "overload",
            "g-load limit exceeded",
            "g_load",
        ) or sl.startswith("overload"):
            return "overload"
        if sl in ("t_int_min", "internal_temperature_min") or "internal temperature below" in sl:
            return "t_int_min"
        if sl in ("t_int_max", "internal_temperature_max") or "internal temperature above" in sl:
            return "t_int_max"
        if sl in ("heatshield", "hs_thermal") or "heatshield thermal failure" in sl:
            return "hs_thermal"
        if sl in ("hard_landing",) or sl.startswith("hard landing"):
            return "hard_landing"
        if sl in ("wrong_site",) or "wrong landing site" in sl:
            return "wrong_site"
        if sl in ("fuel",) or sl.startswith("fuel exhausted"):
            return "fuel"
        if sl in ("terrain",) or sl == "terrain collision":
            return "terrain"
        return s

    @property
    def landing_finished(self) -> bool:
        """True after any touchdown handling (success or failure on surface)."""
        return self._landing_finished

    @property
    def telemetry_history(self) -> list[dict[str, float]]:
        return list(self._telemetry_history)

    @property
    def plot_action_log(self) -> list[Tuple[float, str, str]]:
        """Actions that reached the model (autopilot, levers, target). Used on telemetry graphs."""
        return list(self._plot_action_log)

    def log_plot_action(self, tag_id: str, detail: str = "") -> None:
        """Record a discrete command at current simulation time (call from Controller.apply)."""
        self._plot_action_log.append((float(self.time_s), str(tag_id), str(detail)))

    def _land_cleanup_touchdown(self) -> None:
        self._engine_on = False
        self._throttle_0_1 = 0.0
        th = float(self._world.height_m_at(float(self.pos_x_m), float(self.pos_z_m)))
        self._z_msl_m = th
        self._vertical_speed_mps = 0.0
        self._speed_x_mps = 0.0
        self._speed_z_mps = 0.0
        self._landing_finished = True

    def _check_end_conditions(self) -> None:
        # Terrain / landing first: touchdown can spike g_load; overload must not mask a soft landing.
        pen = float(self._cfg.terrain_penetration_fail_m)
        if self.altitude_m < -pen:
            self._fail("terrain")
            if self.surface_type_under_probe == SurfaceType.LAKE:
                self._water_landed = True
            self._land_cleanup_touchdown()
            return

        if self.altitude_m <= 0.0:
            surf = self.surface_type_under_probe
            v_vert = abs(self.vertical_speed_mps)
            v_hor = self.horizontal_speed_mps
            v_crit_v = float(self._cfg.terrain_collision_v_vert_mps)
            v_crit_h = float(self._cfg.terrain_collision_v_hor_mps)

            # Catastrophic impact: post-step speeds (same frame as touchdown).
            if v_vert >= v_crit_v or v_hor >= v_crit_h:
                self._fail("terrain")
                if surf == SurfaceType.LAKE:
                    self._water_landed = True
                self._land_cleanup_touchdown()
                return

            tol = float(self._cfg.touchdown_v_tol_mps)
            # Worst of approach vs outcome: multi-substep frames can land on a tiny last dt with
            # small pre-step speed while the same frame began faster; Euler can also spike v after step.
            vv_pre = abs(self._vv_step_start_mps)
            vv_post = abs(self.vertical_speed_mps)
            v_vert_soft = max(vv_pre, vv_post)
            h_pre = math.hypot(self._speed_x_step_start_mps, self._speed_z_step_start_mps)
            h_post = float(self.horizontal_speed_mps)
            v_hor_soft = max(h_pre, h_post)
            v_lim = self.touchdown_v_land_mps if surf == SurfaceType.LAND else self.touchdown_v_lake_mps
            ok_vert = v_vert_soft <= v_lim + tol
            ok_hor = v_hor_soft <= self.touchdown_v_hor_mps + tol
            if ok_vert and ok_hor:
                if surf != self._target.surface_type:
                    self._fail("wrong_site")
                elif self.failed:
                    # Prior fault (e.g. overload): cannot upgrade to success.
                    pass
                else:
                    self._result = SimResult.SUCCESS
            else:
                # Always record hard landing when soft limits are exceeded, even if _failed was already set.
                self._fail("hard_landing")

            if surf == SurfaceType.LAKE:
                self._water_landed = True

            self._land_cleanup_touchdown()
            return

        if self.g_load > self.max_overload_g:
            self._fail("overload")

        if self.internal_temp_c < self.t_int_min_c:
            self._fail("t_int_min")
        if self.internal_temp_c > self.t_int_max_c:
            self._fail("t_int_max")

        if not self._heatshield_jettisoned:
            t_fail = float(self._cfg.heatshield_thermal.skin_failure_temp_c)
            if float(self._heatshield_skin_temp_c) >= t_fail:
                self._fail("hs_thermal")

        if self.fuel_kg <= 0.0 and self.altitude_m > 0.0 and self.engine_on:
            self._fail("fuel")


class _ThermalStateProxy:
    """
    Small adapter so `thermal_relaxation_step(cfg, s, ...)` can run without
    exposing PhysicsModel internals to that function.
    """

    def __init__(self, model: PhysicsModel):
        self._m = model

    @property
    def t_int_c(self) -> float:
        return self._m.internal_temp_c

    @t_int_c.setter
    def t_int_c(self, v: float) -> None:
        self._m._internal_temp_c = float(v)

