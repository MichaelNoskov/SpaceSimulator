from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BodyConfig:
    """Celestial body / environment constants."""

    name: str = "Titan"
    g_mps2: float = 1.352  # constant gravity approximation

    rho_surface_kg_m3: float = 5.9
    scale_height_m: float = 40_000.0

    # HASI / merged tables: на больших h плотность часто разрежена/сшита; выше якоря — экспонента (см. sample_atmosphere).
    atmosphere_upper_anchor_alt_m: float = 700_000.0
    atmosphere_upper_scale_height_m: float = 80_000.0
    # Пока ρ очень мала, полная гравитация ускоряет вертикаль сильнее, чем тормозит сопротивление горизонталь — визуально «запаздывание» vx.
    entry_low_density_rho_kg_m3: float = 5.0e-6
    entry_low_density_gravity_scale: float = 0.55

    # ТЗ: модуль скорости входа |v| [м/с] (~6.1 км/с); составляющие из угла γ ниже — см. entry_velocity_inertial_mps.
    entry_speed_mps: float = 6100.0
    entry_flight_path_angle_from_horizontal_deg: float = 65.4
    entry_start_altitude_m: float = 1_700_000.0


def entry_velocity_inertial_mps(body: BodyConfig) -> tuple[float, float, float]:
    """
    Inertial entry: flight-path angle γ above local horizontal, horizontal track along +x.

    Invariant (Pythagorean «sum»): sqrt(v_x² + v_z² + v_vert²) == body.entry_speed_mps.
    Returns (v_vert, v_x, v_z); v_vert < 0 (descent).
    """

    v = float(body.entry_speed_mps)
    g = math.radians(float(body.entry_flight_path_angle_from_horizontal_deg))
    v_vert = -v * math.sin(g)
    v_x = v * math.cos(g)
    return float(v_vert), float(v_x), 0.0


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

    README: bays «обмен с наружной средой» + «нагрев от обшивки» — not full equilibration to T_ext
    (Huygens-class insulation). Ambient exchange is ρ-gated; skin path heats bays only when T_skin > T_int.
    """

    # Upper bound on (1/s) gas-side relaxation toward T_ext; scaled by ρ/(ρ+ρ_knee) like thin-air weak exchange.
    k_relax_1ps: float = 5.5e-5
    # Rarefaction knee [kg/m^3]: high altitude → almost no direct soak to free-stream T_ext.
    k_relax_rho_knee_kg_m3: float = 0.12

    # Dissipated bus power (W; README does not fix the number) and effective heat capacity (J/°C).
    rtg_w: float = 360.0
    heat_capacity_j_per_c: float = 50_000.0

    # Coupling from heatshield while attached (1/s): only heating when T_skin > T_int (insulation blocks cold back-soak).
    k_heatshield_coupling_1ps: float = 9.0e-4

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
    # Tuned so peak entry in the sim reaches ~600–750 °C skin (glow + HUD), not cold soak.
    k_friction_rho_v3: float = 5.78e-7
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
    # Gas-side cooling toward T_ext scales with ρ (thin air → weak molecular exchange).
    k_ambient_1ps: float = 0.055
    rho_ambient_ref_kg_m3: float = 0.018
    # Weak always-on sink (radiation to cold sky / line-of-sight) so T_skin is not over-coupled in vacuum.
    k_ambient_radiative_1ps: float = 0.006
    # Initial skin is computed at reset: T_ext + (friction heating rate)/k_ambient (see model._reset_state).
    t_init_c: float = -155.0  # unused; kept for dataclass compatibility
    # Structural / ablator limit while heatshield is attached (mission failure if exceeded).
    skin_failure_temp_c: float = 1350.0
    # Clamp (numerical / display); glow saturates below t_max.
    t_min_c: float = -230.0
    t_max_c: float = 2400.0


@dataclass(frozen=True)
class WindConfig:
    """Mean wind from JSON; gusts are Ornstein–Uhlenbeck in PhysicsModel."""

    turbulence_enabled: bool = True
    ou_tau_s: float = 14.0
    sigma_floor_mps: float = 0.3
    sigma_scale_from_dwe: float = 1.85
    meridional_wavelength_m: float = 55_000.0
    meridional_strength: float = 0.19


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
    wind: WindConfig = WindConfig()

    # Heatshield jettison: Mach number limit (spec: below ~2–3 Mach).
    heatshield_jettison_max_mach: float = 2.5
    # Ideal gas for speed of sound (N2-dominated atmosphere per spec).
    atmosphere_gamma: float = 1.4
    atmosphere_R_specific_j_kg_k: float = 296.8

    # Optional minimum time under main chute before jettison (0 = no hold; telemetry phase `science_descent` only if >0).
    science_descent_min_s: float = 0.0
    # Main deploy: below this altitude [m]. Jettison altitude is enforced by flight program / player, not the physics gate.
    parachute_main_max_deploy_alt_m: float = 160_000.0
    # Nominal ТЗ jettison altitude [m] for docs / sim.parachute_jettison_max_alt_m (default auto script uses this value).
    parachute_jettison_max_alt_m: float = 2_000.0
    drogue_floor_alt_m: float = 0.0

    # Terrain collision: penetration below surface [m].
    terrain_penetration_fail_m: float = 3.0
    # High-energy impact thresholds vs surface (separate from hard landing).
    terrain_collision_v_vert_mps: float = 30.0
    terrain_collision_v_hor_mps: float = 10.0
    # Soft landing limits use pre-step speeds (see PhysicsModel.step); add this margin [m/s] to nominal limits.
    touchdown_v_tol_mps: float = 0.1
    # Touchdown g telemetry: τ_land from v_ref/(g_ref·g0), Earth g0 from EngineConfig; lake uses longer τ.
    impact_calib_touchdown_speed_mps: float = 5.0
    impact_calib_touchdown_g: float = 15.0
    impact_lake_tau_multiplier: float = 2.0

