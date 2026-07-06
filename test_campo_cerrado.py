"""
Validación algebraica del modelo de campo magnético: evalúa
dipole_field_body() para unos estados (r_I, q, t) fijos.
"""

import numpy as np
from dynamics import quat_to_dcm, quat_normalize
from magnetic_field import dipole_field_body

# (nombre, r_I [m], q=[q1,q2,q3,q4], t [s])
TEST_CASES = [
    ("Ecuador, actitud identidad, t=0",
     np.array([6978137.0, 0.0, 0.0]),
     np.array([0.0, 0.0, 0.0, 1.0]),
     0.0),

    ("Ecuador, actitud identidad, t=17500",
     np.array([6978137.0, 0.0, 0.0]),
     np.array([0.0, 0.0, 0.0, 1.0]),
     17500.0),

    ("Fuera del plano ecuatorial (y>0), t=17500",
     np.array([4934000.0, 4934000.0, 0.0]),
     np.array([0.0, 0.0, 0.0, 1.0]),
     17500.0),

    ("Cerca del polo (z alto), t=17500",
     np.array([1000000.0, 500000.0, 6870000.0]),
     np.array([0.0, 0.0, 0.0, 1.0]),
     17500.0),

    ("Cuadrante negativo (x<0, y<0), t=17500",
     np.array([-4934000.0, -4934000.0, 0.0]),
     np.array([0.0, 0.0, 0.0, 1.0]),
     17500.0),

    ("Actitud NO trivial (rotación 90° eje Z), t=17500",
     # q = [sin(45°)*ez , cos(45°)] = rotación de 90° sobre Z
     np.array([0.0, 0.0, np.sin(np.deg2rad(45)), np.cos(np.deg2rad(45))]),
     17500.0),

    ("Actitud arbitraria (no unitaria a propósito -> se normaliza), t=17500",
     np.array([1000000.0, 2000000.0, 6600000.0]),
     np.array([0.2, -0.5, 0.3, 0.8]),
     17500.0),
]

# Corrige la tupla mal formada del caso "rotación 90°" (le falta r_I)
TEST_CASES[5] = (
    "Actitud NO trivial (rotación 90° eje Z), t=17500",
    np.array([6978137.0, 0.0, 0.0]),
    np.array([0.0, 0.0, np.sin(np.deg2rad(45)), np.cos(np.deg2rad(45))]),
    17500.0,
)


def run_tests():
    print(f"{'Caso':45s} | {'Bx [nT]':>12s} {'By [nT]':>12s} {'Bz [nT]':>12s} {'|B| [nT]':>12s}")
    print("-" * 100)

    for name, r_I, q_raw, t in TEST_CASES:
        q = quat_normalize(q_raw)
        C_BI = quat_to_dcm(q)
        B_b = dipole_field_body(r_I, C_BI, t)      # Tesla
        B_b_nT = B_b * 1e9
        norm_nT = np.linalg.norm(B_b_nT)

        print(f"{name:45s} | "
              f"{B_b_nT[0]:12.4f} {B_b_nT[1]:12.4f} {B_b_nT[2]:12.4f} {norm_nT:12.4f}")

        # Salida detallada para copiar/comparar manualmente
        print(f"    r_I = {r_I.tolist()}  m")
        print(f"    q   = {q.tolist()}  (normalizado)")
        print(f"    t   = {t} s")
        print(f"    B_b = [{B_b_nT[0]:.6f}, {B_b_nT[1]:.6f}, {B_b_nT[2]:.6f}]  nT\n")


if __name__ == "__main__":
    run_tests()