import numpy as np
from params import INERTIA

_I_INV = np.linalg.inv(INERTIA)


def skew(v: np.ndarray) -> np.ndarray:
    """Matriz antisimétrica (producto vectorial) S(v) tal que S(v)·u = v × u."""
    return np.array([
        [ 0.0,  -v[2],  v[1]],
        [ v[2],   0.0, -v[0]],
        [-v[1],  v[0],   0.0]
    ])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)


def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    q1, q2, q3, q4 = q
    C = np.array([
        [1 - 2*(q2**2 + q3**2),   2*(q1*q2 + q3*q4),     2*(q1*q3 - q2*q4)],
        [2*(q1*q2 - q3*q4),       1 - 2*(q1**2 + q3**2), 2*(q2*q3 + q1*q4)],
        [2*(q1*q3 + q2*q4),       2*(q2*q3 - q1*q4),     1 - 2*(q1**2 + q2**2)]
    ])
    return C


def dcm_to_quat(C: np.ndarray) -> np.ndarray:
    trace = np.trace(C)
    K = np.array([
        [C[0,0]-C[1,1]-C[2,2], C[1,0]+C[0,1],       C[0,2]+C[2,0],       C[2,1]-C[1,2]],
        [C[1,0]+C[0,1],       C[1,1]-C[0,0]-C[2,2], C[2,1]+C[1,2],       C[0,2]-C[2,0]],
        [C[0,2]+C[2,0],       C[2,1]+C[1,2],       C[2,2]-C[0,0]-C[1,1], C[1,0]-C[0,1]],
        [C[2,1]-C[1,2],       C[0,2]-C[2,0],       C[1,0]-C[0,1],       trace          ]
    ]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    q = eigvecs[:, np.argmax(eigvals)]
    return q if q[3] >= 0 else -q  # convenio q4 ≥ 0



def euler_equations(omega: np.ndarray,
                    torque_control: np.ndarray,
                    torque_pert: np.ndarray,
                    inertia: np.ndarray | None = None) -> np.ndarray:
    
    if inertia is None:
        I     = INERTIA
        I_inv = _I_INV
    else:
        I     = inertia
        I_inv = np.linalg.inv(inertia)

    Iw        = I @ omega
    gyro_term = np.cross(omega, Iw)
    omega_dot = I_inv @ (-gyro_term + torque_control + torque_pert)
    return omega_dot


def quaternion_kinematics(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
   
    q1, q2, q3, q4 = q
    Xi = np.array([
        [ q4, -q3,  q2],
        [ q3,  q4, -q1],
        [-q2,  q1,  q4],
        [-q1, -q2, -q3]
    ])
    return 0.5 * Xi @ omega


def attitude_derivatives(state: np.ndarray,
                          torque_control: np.ndarray,
                          torque_pert: np.ndarray,
                          inertia: np.ndarray | None = None) -> np.ndarray:
    
    q     = quat_normalize(state[:4])
    omega = state[4:]

    q_dot     = quaternion_kinematics(q, omega)
    omega_dot = euler_equations(omega, torque_control, torque_pert,
                                inertia=inertia)

    return np.concatenate([q_dot, omega_dot])


def rk4_step(state: np.ndarray,
             torque_control: np.ndarray,
             torque_pert: np.ndarray,
             dt: float,
             inertia: np.ndarray | None = None) -> np.ndarray:
    
    def f(s):
        return attitude_derivatives(s, torque_control, torque_pert,
                                    inertia=inertia)

    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)

    state_new       = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    state_new[:4]   = quat_normalize(state_new[:4])
    return state_new