from __future__ import annotations

from dataclasses import dataclass

from .types import SurfaceType  # noqa: F401 (re-export for backwards compat)

from enum import Enum


class SimResult(str, Enum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"


@dataclass
class AtmosphereSample:
    """
    Atmosphere parameters at a given altitude.

    Units:
    - rho: kg/m^3
    - t_ext_c: °C
    - p_bar: bar
    """

    rho: float
    t_ext_c: float
    p_bar: float


@dataclass
class SimState:
    """
    Simulation state (internal variables).

    Conventions / units:
    - vertical axis: +up; height above surface h_m [m], v_vert_mps [m/s]
    - horizontal plane: world x (east) and z (north)
      x_m, z_m [m], v_x_mps, v_z_mps [m/s]
    """

    # time
    t_s: float = 0.0

    # kinematics
    h_m: float = 1_270_000.0
    v_vert_mps: float = -6_500.0
    x_m: float = 0.0
    v_x_mps: float = 0.0
    z_m: float = 0.0
    v_z_mps: float = 0.0

    # vehicle mass & aero reference
    # m_dry_kg includes heatshield mass until jettisoned.
    m_dry_kg: float = 270.0
    m_heatshield_kg: float = 30.0
    m_fuel_kg: float = 50.0
    cd_base: float = 1.2
    a_ref_m2: float = 2.5

    # systems / events (mission flags)
    heatshield_jettisoned: bool = False
    drogue_deployed: bool = False
    main_deployed: bool = False
    chute_jettisoned: bool = False

    engine_on: bool = False
    throttle: float = 0.0  # 0..1

    # thermal
    t_int_c: float = -20.0

    # status/telemetry-lite (kept in state for convenience)
    g_load: float = 0.0
    result: SimResult = SimResult.RUNNING
    failed: bool = False
    failure_reasons: list[str] | None = None

    def __post_init__(self) -> None:
        if self.failure_reasons is None:
            self.failure_reasons = []

    @property
    def failure_reason(self) -> str:
        return self.failure_reasons[-1] if self.failure_reasons else ""


@dataclass(frozen=True)
class Telemetry:
    """
    Derived quantities from state + environment.

    Physical values live here so visualization/UI do not duplicate formulas.
    """

    atm: AtmosphereSample
    phase: str
    g_load: float
    q_dyn_pa: float
    v_air_x_mps: float
    v_air_z_mps: float
    v_air_vert_mps: float
