from __future__ import annotations

import csv
import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Optional, TextIO, Tuple

import numpy as np

from .config import DigitalTwinConfig
from .dynamics import (
    aero_params_for_state,
    drag_force_vector_n,
    fuel_burn_kg,
    heatshield_skin_dTdt,
    heatshield_skin_step,
    thermal_relaxation_step,
)
from .models.atmosphere import AtmosphereTable, load_atmosphere_table, sample_atmosphere
from .models.wind import wind_vec_mps as wind_vec_mps_fn
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

        self._telemetry_history: Deque[dict[str, float]] = deque(maxlen=15000)
        # (sim_time_s, tag_id, detail) — successful commands for mission plots (auto + manual).
        self._plot_action_log: Deque[Tuple[float, str, str]] = deque(maxlen=12000)
        self._tick_id = 0
        self._cache: dict[str, tuple[int, object]] = {}
        self._reset_state()
        self._target = self._make_target_at(0.0, 0.0)
        self._csv_file: Optional[TextIO] = None
        self._csv_writer: Optional[Any] = None
        self._csv_log_path: Optional[str] = None

    # -------------------------
    # Base state (private)
    # -------------------------
    def _reset_state(self) -> None:
        # Simulation time from start (telemetry, phase logic).
        self._time_s = 0.0

        # Absolute vertical position [m]: z_msl = terrain(x,z) + height above it.
        # Terrain from world.height_m_at; height above local surface = z_msl - terrain (see altitude_m).
        terrain0 = float(self._world.height_m_at(0.0, 0.0))
        self._z_msl_m = 1_270_000.0 + terrain0

        # Vertical speed [m/s], +up; negative means descent.
        self._vertical_speed_mps = -6_500.0

        # World horizontal position [m] for minimap, target, surface type.
        self._pos_x_m = 0.0
        self._pos_z_m = 0.0

        # Horizontal velocity [m/s] for wind drift, minimap, landing checks.
        self._speed_x_mps = 0.0
        self._speed_z_mps = 0.0

        # Dry mass [kg]; drives m_total_kg and accelerations.
        self._dry_mass_kg = 270.0

        # Heatshield mass [kg]; removed from dry_mass_kg on jettison.
        self._heatshield_mass_kg = 30.0

        # Fuel mass [kg]; limits thrust and total mass.
        self._fuel_kg = 50.0

        # Base drag coefficient (no chutes/jettison); used for drag.
        self._cd_base = 1.2

        # Reference area [m^2]; modified by chutes/heatshield via derived getters.
        self._a_ref_m2 = 2.5

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
        delta_stag = float(dT0) / max(1e-12, float(hcfg.k_ambient_1ps))
        self._heatshield_skin_temp_c = float(
            np.clip(t_ext0 + delta_stag, float(hcfg.t_min_c), float(hcfg.t_max_c))
        )

        # Outcome: `failed` means a fault was recorded; `result` is final (success/failure).
        self._result = SimResult.RUNNING
        self._failed = False
        self._failure_reasons: list[str] = []
        self._water_landed: bool = False
        self._t_main_deployed_s: Optional[float] = None
        self._telemetry_history.clear()
        self._plot_action_log.clear()

    def reset(self) -> None:
        self._close_csv_log()
        self._reset_state()
        self._target = self._make_target_at(0.0, 0.0)
        self._invalidate_cache()

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
        return float(self._cfg.engine.t_max_n)

    @property
    def engine_isp_s(self) -> float:
        return float(self._cfg.engine.isp_s)

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
        return 15.0

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
        return 0_000.0

    @property
    def main_max_alt_m(self) -> float:
        return 160_000.0

    @property
    def chute_jettison_max_alt_m(self) -> float:
        return 2_000.0

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
        return self._failure_reasons[-1] if self._failure_reasons else ""

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
        return (not self.drogue_deployed) and (self.altitude_m > self.drogue_min_alt_m)

    @property
    def can_main(self) -> bool:
        return (not self.main_deployed) and self.drogue_deployed and (self.altitude_m < self.main_max_alt_m)

    def _science_elapsed_ok_for_chute_jettison(self) -> bool:
        if not self.main_deployed:
            return True
        if self._t_main_deployed_s is None:
            return True
        return (self.time_s - self._t_main_deployed_s) >= float(self._cfg.science_descent_min_s)

    @property
    def can_chute_jettison(self) -> bool:
        return (
            (not self.chute_jettisoned)
            and (self.altitude_m < self.chute_jettison_max_alt_m)
            and self._science_elapsed_ok_for_chute_jettison()
        )

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
        cached = self._get_cached("wind")
        if cached is not None:
            return float(cached[0])  # type: ignore[index]
        w = wind_vec_mps_fn(self.altitude_m)
        self._set_cached("wind", (float(w[0]), float(w[1])))
        return float(w[0])

    @property
    def wind_z_mps(self) -> float:
        cached = self._get_cached("wind")
        if cached is not None:
            return float(cached[1])  # type: ignore[index]
        w = wind_vec_mps_fn(self.altitude_m)
        self._set_cached("wind", (float(w[0]), float(w[1])))
        return float(w[1])

    def wind_vec_mps_at(self, h_m: float) -> tuple[float, float]:
        w = wind_vec_mps_fn(float(h_m))
        return float(w[0]), float(w[1])

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
        f_grav_vert = -m * self.g_titan_mps2
        f_thrust_vert = self.thrust_n
        f_vert = float(f_grav_vert + fv + f_thrust_vert)
        q_dyn = 0.5 * float(self.atm_density_kg_m3) * float(vmag * vmag)
        a_vert = f_vert / m
        a_x = float(fx) / m
        a_z = float(fz) / m
        a_mag = float(math.sqrt(a_vert * a_vert + a_x * a_x + a_z * a_z))
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
                "a_mag_mps2": float(a_mag),
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
        cached = self._get_cached("forces")
        if cached is None:
            self._compute_forces_cached()
            cached = self._get_cached("forces")
        return float(cached["a_mag_mps2"])  # type: ignore[index]

    @property
    def g_load(self) -> float:
        return float(self.accel_mag_mps2) / float(self.g0_mps2)

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
        if self.result != SimResult.RUNNING:
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
        if self.result in (SimResult.SUCCESS, SimResult.FAILURE):
            return

        dt = float(dt_s)
        if dt <= 0.0:
            return

        # new tick; derived caches must be recomputed against current state
        self._invalidate_cache()

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
        dm = fuel_burn_kg(self._cfg, thrust_n=float(self.thrust_n), dt_s=dt)
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
        if (not self._failure_reasons) or (self._failure_reasons[-1] != reason):
            self._failure_reasons.append(reason)

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

    def _check_end_conditions(self) -> None:
        if self.g_load > self.max_overload_g:
            self._fail("Overload > 15g")

        if self.internal_temp_c < self.t_int_min_c:
            self._fail(f"Internal temperature below Tmin ({self.t_int_min_c:.0f}C)")
        if self.internal_temp_c > self.t_int_max_c:
            self._fail(f"Internal temperature above Tmax ({self.t_int_max_c:.0f}C)")

        if self.fuel_kg <= 0.0 and self.altitude_m > 0.0 and self.engine_on:
            self._fail("Fuel exhausted above ground")

        pen = float(self._cfg.terrain_penetration_fail_m)
        if self.altitude_m < -pen:
            self._fail("Terrain collision")
            self._result = SimResult.FAILURE
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

            if v_vert >= v_crit_v or v_hor >= v_crit_h:
                self._fail("Terrain collision")
                self._result = SimResult.FAILURE
                if surf == SurfaceType.LAKE:
                    self._water_landed = True
                self._land_cleanup_touchdown()
                return

            ok_vert = v_vert < (self.touchdown_v_land_mps if surf == SurfaceType.LAND else self.touchdown_v_lake_mps)
            ok_hor = v_hor < self.touchdown_v_hor_mps
            if ok_vert and ok_hor and (not self.failed):
                if surf != self._target.surface_type:
                    self._fail("Wrong landing site (surface mismatch)")
                    self._result = SimResult.FAILURE
                else:
                    self._result = SimResult.SUCCESS
            else:
                self._fail(f"Hard landing ({surf.value})")
                self._result = SimResult.FAILURE

            if surf == SurfaceType.LAKE:
                self._water_landed = True

            self._land_cleanup_touchdown()


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

