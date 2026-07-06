import numpy as np
from params import (
    MU_EARTH, INERTIA,
    MASS, R_CM_B,
    A_FACES, N_FACES, MCP_B,
    SOLAR_CONSTANT, LIGHT_SPEED, C_SPEC, C_DIFFUS,
    EPSILON_AERO, ETA_AERO,
    AIR_DENSITY, ALTITUDE_TABLE,
    RMD_B, T0_JD
)
from dynamics       import quat_to_dcm
from orbital import orbital_altitude


_LOG10_AIR_DENSITY = np.log10(np.maximum(AIR_DENSITY, 1e-30))


def gravity_gradient_torque(r_I: np.ndarray,
                             C_BI: np.ndarray,
                             inertia: np.ndarray | None = None) -> np.ndarray:

    I      = INERTIA if inertia is None else inertia
    r_norm = np.linalg.norm(r_I)
    r_b    = C_BI @ r_I
    T_gg   = (3 * MU_EARTH / r_norm**5) * np.cross(r_b, I @ r_b)
    return T_gg


def _atmosphere_density(altitude_m: float) -> float:
    
    alt = float(np.clip(altitude_m, ALTITUDE_TABLE[0], ALTITUDE_TABLE[-1]))
    log10_rho = np.interp(alt, ALTITUDE_TABLE, _LOG10_AIR_DENSITY)
    return float(10.0 ** log10_rho)


def aerodynamic_torque(r_I: np.ndarray,
                       v_I: np.ndarray,
                       C_BI: np.ndarray,
                       r_cm_b: np.ndarray | None = None) -> np.ndarray:
    
    alt = orbital_altitude(r_I)
    rho = _atmosphere_density(alt)

    v_b = C_BI @ v_I
    v_norm = np.linalg.norm(v_b)
    if v_norm < 1e-10:
        return np.zeros(3)

    v_flow = -(v_b / v_norm)  
    q_dyn = rho * v_norm**2

    F_total        = np.zeros(3)
    numerator_ca   = np.zeros(3)
    denominator_ca = 0.0

    for i in range(6):
        n_i    = N_FACES[i]
        area_i = A_FACES[i]

        n_dot_v = np.dot(n_i, v_flow)
        if n_dot_v <= 0.0: 
            continue

        term1 = (1.0 - EPSILON_AERO) * n_dot_v * v_flow
        term2 = 2.0 * EPSILON_AERO * (n_dot_v**2) * n_i
        term3 = (1.0 - EPSILON_AERO) * ETA_AERO * n_dot_v * n_i

        F_total += -q_dyn * area_i * (term1 + term2 + term3)

       
        weight          = n_dot_v * area_i
        numerator_ca   += MCP_B[i] * weight
        denominator_ca += weight

    if denominator_ca < 1e-12:
        return np.zeros(3)

    c_a = numerator_ca / denominator_ca

    r_cm = R_CM_B if r_cm_b is None else r_cm_b
    lever = c_a - r_cm
    return np.cross(lever, F_total)


def C_IV_from_rv(r_I: np.ndarray, v_I: np.ndarray) -> np.ndarray:
   
    z_V = -r_I / np.linalg.norm(r_I)

    h   =  np.cross(r_I, v_I)
    y_V = -h / np.linalg.norm(h)

    x_V =  np.cross(y_V, z_V)
    x_V =  x_V / np.linalg.norm(x_V)

    C_VI = np.array([x_V, y_V, z_V])
    return C_VI.T

def compute_C_BV(
    r_I: np.ndarray,
    v_I: np.ndarray,
    C_BI: np.ndarray,
) -> np.ndarray:
    
    C_IV = C_IV_from_rv(r_I, v_I)
    return C_BI @ C_IV

def _eclipse_check(
    r_sun_B: np.ndarray,
    C_BV: np.ndarray,
    r_I: np.ndarray,
) -> bool:
    
    norm_sun = np.linalg.norm(r_sun_B)
    if norm_sun == 0:
        raise ValueError("r_sun_B es el vector nulo; comprueba la entrada.")
    sat_sun = r_sun_B / norm_sun

    
    sat_earth_V = np.array([0.0, 0.0, 1.0])
    sat_earth_B = C_BV @ sat_earth_V
    norm_earth = np.linalg.norm(sat_earth_B)
    if norm_earth == 0:
        raise ValueError("C_BV @ [0,0,1] es el vector nulo; comprueba C_BV.")
    sat_earth = sat_earth_B / norm_earth

    R_EARTH = 6_378_137.0
    r_mag = np.linalg.norm(r_I)
    if r_mag <= R_EARTH:
        raise ValueError(f"|r_I| = {r_mag:.1f} m ≤ R_earth; posición inválida.")
    AE_rad = np.arcsin(R_EARTH / r_mag)

   
    cos_alpha = float(np.dot(sat_sun, sat_earth))
    return bool(cos_alpha > np.cos(AE_rad))


SOLAR_CONSTANT = 1361.0          
LIGHT_SPEED    = 299_792_458.0   
J2000          = 2_451_545.0     
AE_RAD         = np.arcsin(6_371e3 / 42_164e3) 


def _sun_direction_inertial(t_sim: float, T0_JD: float) -> np.ndarray:
    
    n = (t_sim / 86400.0 + T0_JD - J2000)   

    L   = np.deg2rad(280.460 + 0.9856474 * n)   
    g   = np.deg2rad(357.528 + 0.9856003 * n)   

    lam = L + np.deg2rad(1.915) * np.sin(g) + np.deg2rad(0.020) * np.sin(2 * g)
    eps = np.deg2rad(23.439 - 4e-7 * n)         

    delta = np.arcsin(np.sin(eps) * np.sin(lam))
    alpha = np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam))

    s_I = np.array([
        np.cos(delta) * np.cos(alpha),
        np.cos(delta) * np.sin(alpha),
        np.sin(delta)
    ])
    return s_I


def solar_pressure_torque(t_sim: float,
                           T0_JD: float,
                           C_BI: np.ndarray,
                           C_BV: np.ndarray,
                           r_I: np.ndarray,
                           r_cm_b: np.ndarray | None = None) -> np.ndarray:
    
    s_I = _sun_direction_inertial(t_sim, T0_JD)

    sun_b = C_BI @ s_I
    sun_b /= np.linalg.norm(sun_b)

    if _eclipse_check(sun_b, C_BV, r_I):
        return np.zeros(3)

    P_srp = SOLAR_CONSTANT / LIGHT_SPEED   # [Pa]

    T_srp = np.zeros(3)
    for i in range(6):
        cos_alpha = np.dot(N_FACES[i], sun_b)
        if cos_alpha <= 0:
            continue

        cs = C_SPEC[i]
        cd = C_DIFFUS[i]

        dF = -P_srp * A_FACES[i] * cos_alpha * (
            (1.0 - cs) * sun_b
            + 2.0 * (cs * cos_alpha + cd / 3.0) * N_FACES[i]
        )

        r_cm = R_CM_B if r_cm_b is None else r_cm_b
        lever_cp = MCP_B[i] - r_cm
        T_srp += np.cross(lever_cp, dF)

    return T_srp


def residual_magnetic_torque(B_b: np.ndarray) -> np.ndarray:
    return np.cross(RMD_B, B_b)


def total_perturbation_torque(r_I: np.ndarray,
                               v_I: np.ndarray,
                               C_BI: np.ndarray,
                               B_b: np.ndarray,
                               t_sim: float,
                               T0_JD: float,
                               C_BV: np.ndarray,
                               inertia: np.ndarray | None = None,
                               r_cm_b: np.ndarray | None = None) -> np.ndarray:
    T = np.zeros(3)
    T += gravity_gradient_torque(r_I, C_BI, inertia=inertia)
    T += aerodynamic_torque(r_I, v_I, C_BI, r_cm_b=r_cm_b)
    T += residual_magnetic_torque(B_b)
    T += solar_pressure_torque(t_sim, T0_JD, C_BI, C_BV, r_I, r_cm_b=r_cm_b)
    return T