import numpy as np
from params import (
    MU_EARTH, A_ORB, ECC, INC, RAAN0, RAAN_RATE, AOP, TA0, N_ORB, T_ORB, J2_RE2
)


def kepler_equation(M: float, e: float,
                    tol: float = 1e-10,
                    max_iter: int = 50) -> float:
    if e < 1e-10:
        return M

    E = M + e * np.sin(M)   
    for _ in range(max_iter):
        f  = E - e * np.sin(E) - M
        fp = 1 - e * np.cos(E)
        dE = -f / fp
        E += dE
        if abs(dE) < tol:
            break

    return E


def eccentric_to_true_anomaly(E: float, e: float) -> float:
    TA = 2.0 * np.arctan2(
        np.sqrt(1.0 + e) * np.sin(E / 2.0),
        np.sqrt(1.0 - e) * np.cos(E / 2.0)
    )
    return TA % (2 * np.pi)


def compute_raan_rate(a: float, e: float, inc: float) -> float:
    n = np.sqrt(MU_EARTH / a**3)
    return -3 * J2_RE2 * n * np.cos(inc) / (2 * (1 - e**2)**2 * a**2)


def propagate_mean_elements(t: float,
                             a:    float = A_ORB,
                             e:    float = ECC,
                             inc:  float = INC,
                             raan0: float = RAAN0,
                             aop:  float = AOP,
                             ta0:  float = TA0,
                             raan_rate: float | None = None) -> dict:
    n = np.sqrt(MU_EARTH / a**3)

    if raan_rate is None:
        raan_rate = compute_raan_rate(a, e, inc)

    RAAN = (raan0 + raan_rate * t) % (2 * np.pi)

    # M0 se obtiene invirtiendo TA0 -> E0 -> M0
    E0 = 2.0 * np.arctan2(
        np.sqrt(1.0 - e) * np.sin(ta0 / 2.0),
        np.sqrt(1.0 + e) * np.cos(ta0 / 2.0)
    )
    M0 = E0 - e * np.sin(E0)
    M  = (M0 + n * t) % (2 * np.pi)

    E  = kepler_equation(M, e)
    TA = eccentric_to_true_anomaly(E, e)

    return {
        'a'   : a,
        'e'   : e,
        'inc' : inc,
        'RAAN': RAAN,
        'AOP' : aop,
        'TA'  : TA,
        'M'   : M,
    }


def kepler_to_cartesian(elements: dict) -> tuple[np.ndarray, np.ndarray]:
    
    a    = elements['a']
    e    = elements['e']
    inc  = elements['inc']
    RAAN = elements['RAAN']
    AOP  = elements['AOP']
    TA   = elements['TA']

    p = a * (1 - e**2)
    r = p / (1 + e * np.cos(TA))

    
    r_pqw = np.array([r * np.cos(TA),
                       r * np.sin(TA),
                       0.0])

    v_pqw = np.sqrt(MU_EARTH / p) * np.array([-np.sin(TA),
                                                e + np.cos(TA),
                                                0.0])

    R = _rot_pqw_to_eci(RAAN, inc, AOP)   

    return R @ r_pqw, R @ v_pqw


def _rot_pqw_to_eci(RAAN: float, inc: float, AOP: float) -> np.ndarray:
    cO, sO = np.cos(RAAN), np.sin(RAAN)
    ci, si = np.cos(inc),  np.sin(inc)
    cw, sw = np.cos(AOP),  np.sin(AOP)

    return np.array([
        [cO*cw - sO*sw*ci,  -cO*sw - sO*cw*ci,  sO*si],
        [sO*cw + cO*sw*ci,  -sO*sw + cO*cw*ci, -cO*si],
        [sw*si,              cw*si,              ci    ]
    ])


def lvlh_dcm(r_I: np.ndarray, v_I: np.ndarray) -> np.ndarray:

    z_O = -r_I / np.linalg.norm(r_I)
    h   =  np.cross(r_I, v_I)
    y_O = -h / np.linalg.norm(h)
    x_O =  np.cross(y_O, z_O)
    x_O =  x_O / np.linalg.norm(x_O)

    return np.array([x_O, y_O, z_O])   


def eci_to_ecef(r_I: np.ndarray, t: float,
                theta_g0: float = 0.0,
                omega_e: float = 7.29e-5) -> np.ndarray:
    theta = theta_g0 + omega_e * t
    c, s  = np.cos(theta), np.sin(theta)
    R_z   = np.array([
        [ c,  s, 0],
        [-s,  c, 0],
        [ 0,  0, 1]
    ])
    return R_z @ r_I


def orbital_altitude(r_I: np.ndarray, R_earth: float = 6_378_137.0) -> float:
    return np.linalg.norm(r_I) - R_earth
