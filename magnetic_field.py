"""
Modelo de campo magnético terrestre: dipolo inclinado.
"""

import numpy as np
from params import (
    H0, DIP_COEL, DIP_LONG, R_EARTH,
    OMEGA_E, THETA_G0, PHI_EARTH_2000, SEC_FROM_2000
)


def dipole_field_eci(r_I: np.ndarray, t: float) -> np.ndarray:
    theta_g = THETA_G0 + OMEGA_E * t
    phi_dip = DIP_LONG + theta_g

    m_hat = np.array([
        np.sin(DIP_COEL) * np.cos(phi_dip),
        np.sin(DIP_COEL) * np.sin(phi_dip),
        np.cos(DIP_COEL)
    ])

    r_norm = np.linalg.norm(r_I)
    r_hat  = r_I / r_norm

    coeff = H0 * 1e-9 * (R_EARTH / r_norm)**3

    return coeff * (3 * np.dot(m_hat, r_hat) * r_hat - m_hat)


def dipole_field_body(r_I: np.ndarray,
                      C_BI: np.ndarray,
                      t: float) -> np.ndarray:
    
    B_I = dipole_field_eci(r_I, t)
    return C_BI @ B_I


def bdot_numerical(B_b_prev: np.ndarray,
                   B_b_curr: np.ndarray,
                   dt: float) -> np.ndarray:
    return (B_b_curr - B_b_prev) / dt

