from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BodyConfig:
    """Celestial body / environment constants."""

    name: str = "Titan"
    g_mps2: float = 1.352  # constant gravity approximation

    # Used for exponential fallback atmosphere if no table is provided.
    rho_surface_kg_m3: float = 5.9
    scale_height_m: float = 40_000.0


@dataclass(frozen=True)
class EngineConfig:
    """Engine constants."""

    t_max_n: float = 5000.0
    isp_s: float = 300.0
    g0_mps2: float = 9.80665  # standard gravity for Isp conversion


@dataclass(frozen=True)
class ThermalConfig:
    """
    Simplified thermal model for internal bays.

    Components:
    - relaxation toward external temperature (heat exchange)
    - internal heating (RTG)
    - aerodynamic heating via dynamic pressure q_dyn
    - survival limits
    """

    # Heat exchange (1/s): dT/dt += k_relax*(T_ext - T_int)
    k_relax_1ps: float = 0.02

    # RTG (W) and effective heat capacity (J/°C): dT/dt += rtg_w / heat_capacity
    rtg_w: float = 300.0
    heat_capacity_j_per_c: float = 50_000.0

    # Aerodynamic heating via q_dyn (Pa): dT/dt += k_qdyn * q_dyn
    # Coefficient is a reasonable placeholder for this educational model.
    k_qdyn_c_per_s_per_pa: float = 2e-6

    # Internal bay temperature limits (failure if exceeded).
    t_min_c: float = -180.0
    t_max_c: float = 120.0


@dataclass(frozen=True)
class DigitalTwinConfig:
    """
    Top-level digital twin configuration.

    Usually edited to retarget the simulation to another body or vehicle.
    """

    body: BodyConfig = BodyConfig()
    engine: EngineConfig = EngineConfig()
    thermal: ThermalConfig = ThermalConfig()

    # Heatshield jettison: Mach number limit (spec: below ~2–3 Mach).
    heatshield_jettison_max_mach: float = 2.5
    # Ideal gas for speed of sound (N2-dominated atmosphere per spec).
    atmosphere_gamma: float = 1.4
    atmosphere_R_specific_j_kg_k: float = 296.8

    # Minimum time under main chute before jettison (science descent ~2 h per spec).
    science_descent_min_s: float = 7200.0

    # Terrain collision: penetration below surface [m].
    terrain_penetration_fail_m: float = 3.0
    # High-energy impact thresholds vs surface (separate from hard landing).
    terrain_collision_v_vert_mps: float = 30.0
    terrain_collision_v_hor_mps: float = 10.0

