from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import DigitalTwinConfig, HeatshieldThermalConfig
from .state import AtmosphereSample, SimState


@dataclass(frozen=True)
class AeroParams:
    """
    Aerodynamic parameters for the drag law.

    - cd: drag coefficient (dimensionless)
    - area_m2: reference area [m^2]
    """

    cd: float
    area_m2: float


@dataclass(frozen=True)
class Forces:
    """
    Forces on the vehicle in world axes.

    Conventions:
    - vertical axis: +up
    - x, z: horizontal plane
    """

    f_vert_n: float
    f_x_n: float
    f_z_n: float


@dataclass(frozen=True)
class Accels:
    """Accelerations [m/s^2] in world axes."""

    a_vert_mps2: float
    a_x_mps2: float
    a_z_mps2: float


def aero_params_for_state(s: SimState) -> AeroParams:
    """
    Aero parameters that depend on mission phase.

    Separates vehicle configuration from the integrator and force formulas.
    """

    cd = float(s.cd_base)
    area = float(s.a_ref_m2)

    # Parachutes increase reference area and Cd.
    if s.drogue_deployed and not s.chute_jettisoned:
        area = 5.3
        cd = 1.6

    if s.main_deployed and not s.chute_jettisoned:
        area = 54.0
        cd = 1.7

    # Heatshield changes effective shape.
    if s.heatshield_jettisoned:
        cd *= 0.9
        area *= 0.85

    if s.chute_jettisoned:
        cd = float(s.cd_base) * 0.95
        area = float(s.a_ref_m2)

    return AeroParams(cd=cd, area_m2=area)


def thrust_force_n(cfg: DigitalTwinConfig, s: SimState) -> float:
    """
    Thrust force (upward).

    Formula:
      T = throttle * T_max
    """

    if not s.engine_on:
        return 0.0
    if s.throttle <= 0.0:
        return 0.0
    if s.m_fuel_kg <= 0.0:
        return 0.0
    return float(cfg.engine.t_max_n) * float(np.clip(s.throttle, 0.0, 1.0))


def fuel_burn_kg(cfg: DigitalTwinConfig, thrust_n: float, dt_s: float) -> float:
    """
    Fuel mass burned over dt.

    Mass flow approximation:
      mdot = T / (Isp * g0)
      dm = mdot * dt

    Units:
    - T [N]
    - Isp [s]
    - g0 [m/s^2]
    - mdot [kg/s]
    """

    if thrust_n == 0.0 or dt_s <= 0.0:
        return 0.0
    mdot = abs(float(thrust_n)) / (float(cfg.engine.isp_s) * float(cfg.engine.g0_mps2))
    return float(mdot * dt_s)


def drag_force_vector_n(
    rho_kg_m3: float,
    cd: float,
    area_m2: float,
    v_rel_x_mps: float,
    v_rel_z_mps: float,
    v_rel_vert_mps: float,
) -> tuple[float, float, float, float]:
    """
    Quadratic drag, vector form (per spec).

    Formula:
      F_drag = -1/2 * rho * Cd * A * |v| * v_vec

    v_vec is air-relative velocity. Returns:
    - (F_x, F_z, F_vert, |v|).
    """

    vx = float(v_rel_x_mps)
    vz = float(v_rel_z_mps)
    vv = float(v_rel_vert_mps)
    v_mag = float(math.sqrt(vx * vx + vz * vz + vv * vv))
    if v_mag <= 0.0:
        return 0.0, 0.0, 0.0, 0.0

    k = -0.5 * float(rho_kg_m3) * float(cd) * float(area_m2) * v_mag
    return k * vx, k * vz, k * vv, v_mag


def compute_forces(
    cfg: DigitalTwinConfig,
    s: SimState,
    atm: AtmosphereSample,
    wind_x_mps: float,
    wind_z_mps: float,
) -> tuple[Forces, dict[str, float]]:
    """
    Compute forces on the vehicle.

    Returns forces and intermediate values (relative speeds, q_dyn, etc.).
    """

    aero = aero_params_for_state(s)

    # Air-relative velocity in the horizontal plane.
    v_air_x = float(s.v_x_mps) - float(wind_x_mps)
    v_air_z = float(s.v_z_mps) - float(wind_z_mps)
    v_air_vert = float(s.v_vert_mps)

    # Drag opposite to air-relative velocity.
    f_drag_x, f_drag_z, f_drag_vert, v_air_mag = drag_force_vector_n(
        atm.rho,
        aero.cd,
        aero.area_m2,
        v_rel_x_mps=v_air_x,
        v_rel_z_mps=v_air_z,
        v_rel_vert_mps=v_air_vert,
    )

    # Gravity (constant g).
    # F_g = -m * g
    m_kg = max(1.0, float(s.m_dry_kg + s.m_fuel_kg))
    f_grav = -m_kg * float(cfg.body.g_mps2)

    # Thrust (up).
    f_thrust = thrust_force_n(cfg, s)

    f_vert = f_grav + f_drag_vert + f_thrust

    # Dynamic pressure (telemetry / heating visualization):
    # q = 0.5 * rho * |v_air|^2
    q_dyn = 0.5 * float(atm.rho) * float(v_air_mag * v_air_mag)

    return (
        Forces(f_vert_n=f_vert, f_x_n=f_drag_x, f_z_n=f_drag_z),
        {
            "v_air_x_mps": v_air_x,
            "v_air_z_mps": v_air_z,
            "v_air_vert_mps": v_air_vert,
            "q_dyn_pa": q_dyn,
            "m_kg": m_kg,
            "cd": float(aero.cd),
            "area_m2": float(aero.area_m2),
            "f_thrust_n": float(f_thrust),
            "f_grav_n": float(f_grav),
        },
    )


def accelerations_from_forces(forces: Forces, m_kg: float) -> Accels:
    """Accelerations: a = F / m."""

    m_kg = max(1e-9, float(m_kg))
    return Accels(
        a_vert_mps2=float(forces.f_vert_n) / m_kg,
        a_x_mps2=float(forces.f_x_n) / m_kg,
        a_z_mps2=float(forces.f_z_n) / m_kg,
    )


def integrate_explicit_euler(s: SimState, a: Accels, dt_s: float) -> None:
    """
    Explicit (forward) Euler per spec: position with velocity at step start, then velocity.

      x_{t+dt} = x_t + v_t * dt
      v_{t+dt} = v_t + a_t * dt
    """

    dt = float(dt_s)
    if dt <= 0.0:
        return

    vv = float(s.v_vert_mps)
    vx = float(s.v_x_mps)
    vz = float(s.v_z_mps)
    s.h_m += vv * dt
    s.v_vert_mps = vv + float(a.a_vert_mps2) * dt

    s.x_m += vx * dt
    s.v_x_mps = vx + float(a.a_x_mps2) * dt

    s.z_m += vz * dt
    s.v_z_mps = vz + float(a.a_z_mps2) * dt


def integrate_semi_implicit_euler(s: SimState, a: Accels, dt_s: float) -> None:
    """
    Legacy: semi-implicit Euler (update v, then x with new v).

    The main simulator uses explicit order in `PhysicsModel.step`; this remains for
    older code paths using `SimState`.
    """

    dt = float(dt_s)
    if dt <= 0.0:
        return

    s.v_vert_mps += float(a.a_vert_mps2) * dt
    s.h_m += float(s.v_vert_mps) * dt

    s.v_x_mps += float(a.a_x_mps2) * dt
    s.x_m += float(s.v_x_mps) * dt

    s.v_z_mps += float(a.a_z_mps2) * dt
    s.z_m += float(s.v_z_mps) * dt


def heatshield_skin_dTdt(
    t_skin_c: float,
    t_ext_c: float,
    rho_kg_m3: float,
    v_rel_mag_mps: float,
    cfg: HeatshieldThermalConfig,
) -> float:
    """Time derivative of heatshield skin temperature [°C/s] (explicit thermal ODE right-hand side)."""
    rho = max(0.0, float(rho_kg_m3))
    vm = max(0.0, float(v_rel_mag_mps))
    f_gas = max(0.0, float(cfg.titan_gas_heating_factor))
    kf = float(cfg.k_friction_rho_v3)
    rho_exp = float(cfg.rho_exponent)
    v_exp = float(cfg.v_exponent)
    knee = max(1e-18, float(cfg.rho_knee_kg_m3))
    f_rare = rho / (rho + knee)
    q_base = kf * (rho**rho_exp) * (vm**v_exp) * f_gas * f_rare

    t = float(t_skin_c)
    t_ref = float(cfg.t_pyrolysis_ref_c)
    kb = float(cfg.k_blowing)
    p_bl = float(cfg.blowing_exponent)
    bcap = float(cfg.blowing_max_fraction)
    mdot_proxy = math.sqrt(rho * vm + 1e-30) * (max(0.0, t - t_ref) ** p_bl)
    blow = min(max(0.0, bcap), kb * mdot_proxy)
    q_conv = q_base * (1.0 - blow)

    ta = float(cfg.t_ablation_onset_c)
    k_abl = float(cfg.k_ablation_cooling)
    p_abl = float(cfg.ablation_exponent)
    q_abl = k_abl * (max(0.0, t - ta) ** p_abl) * math.sqrt(rho + 1e-30) * vm

    ka = float(cfg.k_ambient_1ps)
    t_ext = float(t_ext_c)
    return q_conv - q_abl - ka * (t - t_ext)


def heatshield_skin_step(
    t_skin_c: float,
    t_ext_c: float,
    rho_kg_m3: float,
    v_rel_mag_mps: float,
    dt_s: float,
    cfg: HeatshieldThermalConfig,
) -> float:
    """Advance heatshield skin temperature [°C] one explicit Euler step."""
    dt = float(dt_s)
    if dt <= 0.0:
        return float(t_skin_c)
    dTdt = heatshield_skin_dTdt(t_skin_c, t_ext_c, rho_kg_m3, v_rel_mag_mps, cfg)
    t_new = float(t_skin_c) + dTdt * dt
    return float(np.clip(t_new, float(cfg.t_min_c), float(cfg.t_max_c)))


def thermal_relaxation_step(
    cfg: DigitalTwinConfig,
    s: SimState,
    t_ext_c: float,
    q_dyn_pa: float,
    dt_s: float,
    *,
    t_skin_c: float,
    heatshield_attached: bool,
) -> None:
    """
    Internal bay temperature:
      dT_int/dt = k_relax*(T_ext - T_int) + RTG/C
                  + [if heatshield: k_couple*(T_skin - T_int)]
                  + [if not heatshield: k_qdyn*q_dyn]
    """

    dt = float(dt_s)
    if dt <= 0.0:
        return

    t_ext = float(t_ext_c)
    t_int = float(s.t_int_c)
    k_relax = float(cfg.thermal.k_relax_1ps)

    rtg_w = float(cfg.thermal.rtg_w)
    heat_cap = max(1.0, float(cfg.thermal.heat_capacity_j_per_c))
    k_qdyn = float(cfg.thermal.k_qdyn_c_per_s_per_pa)
    k_couple = float(cfg.thermal.k_heatshield_coupling_1ps)

    dTdt = k_relax * (t_ext - t_int) + (rtg_w / heat_cap)
    if heatshield_attached:
        dTdt += k_couple * (float(t_skin_c) - t_int)
    else:
        dTdt += k_qdyn * float(q_dyn_pa)
    s.t_int_c = t_int + dTdt * dt

