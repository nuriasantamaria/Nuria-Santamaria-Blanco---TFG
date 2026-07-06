import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional


def step_signal(t: np.ndarray,
                t_start: float,
                amplitude: float = 1.0) -> np.ndarray:
    return np.where(t >= t_start, amplitude, 0.0)


def ramp_signal(t: np.ndarray,
                t_start: float,
                slope: float = 1.0) -> np.ndarray:
    return np.where(t >= t_start, slope * (t - t_start), 0.0)


def sine_signal(t: np.ndarray,
                frequency: float,
                amplitude: float = 1.0,
                phase: float = 0.0) -> np.ndarray:
    return amplitude * np.sin(2 * np.pi * frequency * t + phase)


def impulse_signal(t: np.ndarray,
                   t_impulse: float,
                   dt: float,
                   amplitude: float = 1.0) -> np.ndarray:
    return np.where(
        (t >= t_impulse) & (t < t_impulse + dt),
        amplitude / dt, 0.0
    )


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    v1, s1 = q1[:3], q1[3]
    v2, s2 = q2[:3], q2[3]
    v = s1*v2 + s2*v1 + np.cross(v1, v2)
    s = s1*s2 - np.dot(v1, v2)
    return np.append(v, s)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_error(q_actual: np.ndarray, q_target: np.ndarray) -> np.ndarray:
    return quat_multiply(quat_conjugate(q_target), q_actual)


def angle_from_quat(q: np.ndarray) -> float:
    return 2 * np.arccos(np.clip(abs(q[3]), -1.0, 1.0))


def euler_angles_zyx(C: np.ndarray) -> tuple[float, float, float]:
    pitch = -np.arcsin(C[0, 2])
    roll  =  np.arctan2(C[1, 2], C[2, 2])
    yaw   =  np.arctan2(C[0, 1], C[0, 0])
    return yaw, pitch, roll


def plot_angular_velocity(t: np.ndarray,
                           omega_history: np.ndarray,
                           omega_target: Optional[float] = None,
                           title: str = "Velocidad angular",
                           save_path: Optional[str] = None):
    omega_deg = np.rad2deg(omega_history)
    norm_deg  = np.linalg.norm(omega_deg, axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(t, omega_deg[:, 0], label=r'$\omega_x$')
    axes[0].plot(t, omega_deg[:, 1], label=r'$\omega_y$')
    axes[0].plot(t, omega_deg[:, 2], label=r'$\omega_z$')
    axes[0].set_ylabel("ω [°/s]")
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_title(title)

    axes[1].plot(t, norm_deg, 'k', label=r'$\|\omega\|$')
    if omega_target is not None:
        axes[1].axhline(np.rad2deg(omega_target), color='r',
                         linestyle='--', label='Objetivo')
    axes[1].set_xlabel("Tiempo [s]")
    axes[1].set_ylabel(r"$\|\omega\|$ [°/s]")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_quaternion(t: np.ndarray,
                    quat_history: np.ndarray,
                    title: str = "Cuaternión de actitud",
                    save_path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(10, 4))
    labels = ['q1', 'q2', 'q3', 'q4 (escalar)']
    for i, lbl in enumerate(labels):
        ax.plot(t, quat_history[:, i], label=lbl)
    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel("Componente cuaternión [-]")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_magnetic_field(t: np.ndarray,
                         B_history: np.ndarray,
                         title: str = "Campo magnético en Body frame",
                         save_path: Optional[str] = None):
    """Gráfica de B_b (3 componentes) en µT frente al tiempo."""
    B_uT = B_history * 1e6  # T → µT
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, B_uT[:, 0], label='Bx')
    ax.plot(t, B_uT[:, 1], label='By')
    ax.plot(t, B_uT[:, 2], label='Bz')
    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel("B [µT]")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_control_torque(t: np.ndarray,
                         torque_history: np.ndarray,
                         title: str = "Par de control",
                         save_path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, torque_history[:, 0], label='Tx')
    ax.plot(t, torque_history[:, 1], label='Ty')
    ax.plot(t, torque_history[:, 2], label='Tz')
    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel("Par [N·m]")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def time_to_detumble(t: np.ndarray,
                      omega_history: np.ndarray,
                      omega_threshold: float) -> Optional[float]:
    norms = np.linalg.norm(omega_history, axis=1)
    idx   = np.argwhere(norms < omega_threshold)
    return float(t[idx[0, 0]]) if len(idx) > 0 else None


def rotational_kinetic_energy(omega: np.ndarray, I: np.ndarray) -> float:
    omega = np.asarray(omega)
    I = np.asarray(I)
    return 0.5 * float(omega @ I @ omega)


def episode_metrics(t: np.ndarray,
                     omega_history: np.ndarray,
                     torque_history: np.ndarray,
                     omega_threshold: float,
                     I: np.ndarray) -> dict:
    td       = time_to_detumble(t, omega_history, omega_threshold)
    e_init   = rotational_kinetic_energy(omega_history[0],  I)
    e_final  = rotational_kinetic_energy(omega_history[-1], I)
    norms    = np.linalg.norm(torque_history, axis=1)
    effort   = float(np.trapezoid(norms, t))

    return {
        "t_detumble"       : td,
        "omega_final"      : float(np.linalg.norm(omega_history[-1])),
        "energy_initial"   : e_init,
        "energy_final"     : e_final,
        "energy_dissipated": e_init - e_final,
        "control_effort"   : effort,
    }


def save_trajectory(path: str,
                     t: np.ndarray,
                     state_history: np.ndarray,
                     obs_history: np.ndarray,
                     action_history: np.ndarray,
                     reward_history: np.ndarray):
    np.savez_compressed(
        path,
        t=t,
        state=state_history,
        obs=obs_history,
        action=action_history,
        reward=reward_history,
    )
    print(f"Trayectoria guardada en {path}.npz")


def load_trajectory(path: str) -> dict:
    data = np.load(path if path.endswith('.npz') else path + '.npz')
    return dict(data)
