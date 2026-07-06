"""
Modelos de sensores del UPMSat-2: los tres magnetómetros (calibración +
ruido gaussiano) y la fusión de los que estén activos.

"""

import numpy as np
from params import (
    CM_MM01, CO_MM01,
    CM_MM02, CO_MM02,
    CM_MM03, CO_MM03,
    MM_WORKING,
)

# inversas precalculadas de las matrices de calibración
_CM01_INV = np.linalg.inv(CM_MM01)
_CM02_INV = np.linalg.inv(CM_MM02)
_CM03_INV = np.linalg.inv(CM_MM03)


def _magnetometer_model(B_b_true: np.ndarray,
                         CM: np.ndarray,
                         CM_inv: np.ndarray,
                         CO: np.ndarray,
                         noise_std_nT: float = 10.0,
                         rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    B_nT = B_b_true * 1e9

    V_sim = CM @ B_nT + CO
    B_cal_nT = CM_inv @ (V_sim - CO)   # = B_nT si la calibración es perfecta

    noise_nT = rng.normal(0.0, noise_std_nT, size=3)

    return (B_cal_nT + noise_nT) * 1e-9


def magnetometer_1(B_b_true: np.ndarray,
                   noise_std_nT: float = 10.0,
                   rng: np.random.Generator | None = None) -> np.ndarray:    
    return _magnetometer_model(B_b_true, CM_MM01, _CM01_INV, CO_MM01,
                               noise_std_nT, rng)


def magnetometer_2(B_b_true: np.ndarray,
                   noise_std_nT: float = 10.0,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    return _magnetometer_model(B_b_true, CM_MM02, _CM02_INV, CO_MM02,
                               noise_std_nT, rng)


def magnetometer_3(B_b_true: np.ndarray,
                   noise_std_nT: float = 5.0,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    return _magnetometer_model(B_b_true, CM_MM03, _CM03_INV, CO_MM03,
                               noise_std_nT, rng)


def fused_magnetometer(B_b_true: np.ndarray,
                        mm_working: np.ndarray = MM_WORKING,
                        noise_std_nT: float = 10.0,
                        rng: np.random.Generator | None = None) -> np.ndarray:
    measures = []
    fns      = [magnetometer_1, magnetometer_2, magnetometer_3]

    for i, (fn, active) in enumerate(zip(fns, mm_working)):
        if active:
            measures.append(fn(B_b_true, noise_std_nT, rng))

    if not measures:
        raise RuntimeError("No hay magnetómetros activos (MM_WORKING = [0,0,0]).")

    return np.mean(measures, axis=0)


def tesla_to_nanotesla(B: np.ndarray) -> np.ndarray:
    return B * 1e9


def nanotesla_to_tesla(B_nT: np.ndarray) -> np.ndarray:
    return B_nT * 1e-9