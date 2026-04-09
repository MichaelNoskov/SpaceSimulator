from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BodyConfig:
    """Celestial body / environment constants."""

    name: str = "Titan"
    g_mps2: float = 1.352  # constant gravity approximation

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

    With heatshield on: bays exchange with ambient and receive heat from the hot backface (T_skin).
    After jettison: direct aerodynamic heating via q_dyn is added again.
    """

    # Heat exchange with ambient gas (1/s): dT/dt += k_relax*(T_ext - T_int)
    k_relax_1ps: float = 0.02

    # RTG (W) and effective heat capacity (J/°C): dT/dt += rtg_w / heat_capacity
    rtg_w: float = 300.0
    heat_capacity_j_per_c: float = 50_000.0

    # Coupling from heatshield skin while attached (1/s): dT/dt += k_couple * (T_skin - T_int)
    k_heatshield_coupling_1ps: float = 3.8e-4

    # Direct aero heating to hull when heatshield is off: dT/dt += k_qdyn * q_dyn [Pa]
    k_qdyn_c_per_s_per_pa: float = 2e-6

    # Internal bay temperature limits (failure if exceeded).
    t_min_c: float = -180.0
    t_max_c: float = 120.0


@dataclass(frozen=True)
class HeatshieldThermalConfig:
    """
    Reduced-order heatshield outer skin (not a single closed-form ρ|v|³ lamp).

    - Convective / stagnation proxy with configurable exponents on ρ and |v_rel|.
    - Rarefaction knee: heating scales with ρ/(ρ+ρ_knee) so the ρ|v|^n law weakens in thin air.
    - Pyrolysis “blowing”: hot skin blocks part of the convective flux (boundary-layer proxy).
    - Ablative sink: removes enthalpy as skin exceeds an onset temperature (no separate mass ODE).

    rho [kg/m^3] and |v_rel| match the drag / dynamic-pressure path (wind included).
    """

    # Leading coefficient for the heating proxy (legacy name: originally ρ|v|³).
    k_friction_rho_v3: float = 1.52e-9
    rho_exponent: float = 1.0
    v_exponent: float = 3.0
    # ~1.0 for N2 ~95% + CH4 ~5% (data/titan_atm.json); adjust to compare to other atmospheres.
    titan_gas_heating_factor: float = 1.0
    # Continuum → rarefied transition [kg/m^3]: heating × ρ/(ρ+ρ_knee).
    rho_knee_kg_m3: float = 8.0e-5
    # Blowing reduces convective heating: factor (1 − min(cap, k_blowing * ṁ_proxy)).
    t_pyrolysis_ref_c: float = 120.0
    k_blowing: float = 0.0025
    blowing_exponent: float = 1.2
    blowing_max_fraction: float = 0.62
    # Ablative / pyrolysis cooling [°C/s scale]: grows with (T−T_onset)^p √(ρ)|v|.
    k_ablation_cooling: float = 2.8e-6
    t_ablation_onset_c: float = 80.0
    ablation_exponent: float = 1.15
    # Newtonian cooling / equilibration toward free-stream temperature [1/s].
    k_ambient_1ps: float = 0.03
    # Initial skin is computed at reset: T_ext + (friction heating rate)/k_ambient (see model._reset_state).
    t_init_c: float = -155.0  # unused; kept for dataclass compatibility
    # Clamp (numerical / display); glow saturates below t_max.
    t_min_c: float = -230.0
    t_max_c: float = 2400.0


@dataclass(frozen=True)
class DigitalTwinConfig:
    """
    Top-level digital twin configuration.

    Usually edited to retarget the simulation to another body or vehicle.
    """

    body: BodyConfig = BodyConfig()
    engine: EngineConfig = EngineConfig()
    thermal: ThermalConfig = ThermalConfig()
    heatshield_thermal: HeatshieldThermalConfig = HeatshieldThermalConfig()

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

