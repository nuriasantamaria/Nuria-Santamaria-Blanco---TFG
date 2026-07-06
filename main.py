import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import params as p
from environment import UPMSat2Env
from environment_ppo import UPMSat2EnvPPO
from wrappers import HistoryWrapper

from dynamics       import quat_to_dcm, quat_normalize
from magnetic_field import dipole_field_body, bdot_numerical
from sensors        import fused_magnetometer
from utils import (
    plot_angular_velocity,
    plot_quaternion,
    plot_magnetic_field,
    episode_metrics,
    save_trajectory,
)



_POLICY_KWARGS = {
    "ppo": dict(
        net_arch       = [128, 64],
        activation_fn  = torch.nn.ReLU,
        log_std_init   = -0.6903620852865224,  
    ),
    "recurrent_ppo": dict(
        net_arch        = [256, 256],
        activation_fn   = torch.nn.ReLU,
        log_std_init    = -0.02766447324051589,
        n_lstm_layers   = 1,
        lstm_hidden_size= 64,
    ),
    "sac": dict(
        net_arch      = dict(pi=[128, 64], qf=[128, 64]),
        activation_fn = None,
    ),
}


def _make_env(seed: int = 0,
              omega_detumble_thresh: float = 0.0025,
              max_steps: int = 15000,
              lambda_smooth: float = 0.08,
              gaussian_amplitude: float = 0.05,
              gaussian_sigma: float = 0.02,
              history_length: int = 0):
    
    def _init():
        if history_length > 0:
            env = UPMSat2EnvPPO(seed=seed,
                                max_steps=max_steps,
                                lambda_smooth=lambda_smooth,
                                omega_detumble_thresh=omega_detumble_thresh,
                                gaussian_amplitude=gaussian_amplitude,
                                gaussian_sigma=gaussian_sigma)
            return HistoryWrapper(env, history_length=history_length)
        return UPMSat2Env(seed=seed,
                          max_steps=max_steps,
                          lambda_smooth=lambda_smooth,
                          omega_detumble_thresh=omega_detumble_thresh,
                          gaussian_amplitude=gaussian_amplitude,
                          gaussian_sigma=gaussian_sigma)
    return _init


def _resolve_device(device: str) -> str:
    device = device.lower().strip()
    if device == "cuda":
        try:
            import torch
        except ImportError:
            print("[ERROR] PyTorch no instalado — no se puede usar CUDA.")
            sys.exit(1)
        if not torch.cuda.is_available():
            print("[ERROR] CUDA no disponible en este equipo.")
            sys.exit(1)
    elif device != "cpu":
        print(f"[ERROR] Dispositivo no válido: {device}. Usa 'cpu' o 'cuda'.")
        sys.exit(1)
    return device


def _load_algo(algo: str):
    """Devuelve la clase del algoritmo solicitado."""
    algo = algo.lower().strip()
    try:
        if algo == "ppo":
            from stable_baselines3 import PPO
            return PPO, "MlpPolicy"
        elif algo == "recurrent_ppo":
            from sb3_contrib import RecurrentPPO
            return RecurrentPPO, "MlpLstmPolicy"
        elif algo == "sac":
            from stable_baselines3 import SAC
            return SAC, "MlpPolicy"
        else:
            print(f"[ERROR] Algoritmo no soportado: {algo}. Usa 'ppo', 'recurrent_ppo' o 'sac'.")
            sys.exit(1)
    except ImportError:
        print("[ERROR] Faltan dependencias: pip install stable-baselines3 sb3-contrib")
        sys.exit(1)


def _print_header(title: str):
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")


def _print_results(t, omega_deg_norm, m_norm,
                   omega_target_deg, label):
    idx_real  = np.argwhere(omega_deg_norm < omega_target_deg)
    t_real    = f"{t[idx_real[0,0]]:.0f} s" if len(idx_real) else "no alcanzado"
    print(f"\n  {'─'*50}")
    print(f"  {label}")
    print(f"  {'─'*50}")
    print(f"  |ω|_real inicial      : {omega_deg_norm[0]:.3f} °/s")
    print(f"  |ω|_real final        : {omega_deg_norm[-1]:.3f} °/s")
    print(f"  t_detumble (real)     : {t_real}")
    print(f"  |m| medio             : {m_norm.mean():.2f} A·m²")
    print(f"  {'─'*50}\n")


def _add_orbit_lines(axes, x_max_min: float):
    t_orb_min = p.T_ORB / 60.0
    k = 1
    while k * t_orb_min < x_max_min:
        for ax in axes:
            ax.axvline(k * t_orb_min, color='gray', ls='--', lw=0.7, alpha=0.5)
        k += 1


def _first_crossing_min(t_s: np.ndarray, omega_arr: np.ndarray,
                        threshold_deg: float) -> str:
    """Devuelve el tiempo (en minutos) del primer paso en que |ω| < threshold_deg."""
    omega_deg = np.linalg.norm(omega_arr, axis=1) * 180 / np.pi
    idx = np.argwhere(omega_deg < threshold_deg)
    if len(idx) == 0:
        return "no alcanzado"
    return f"{t_s[idx[0, 0]] / 60.0:.1f} min"


def _plot_comparison(ppo: dict, bdot: dict, omega_target: float,
                     title: str = "Algoritmo evaluado vs Bdot clásico",
                     x_max_min: float = None,
                     thresholds_deg: list = None,
                     run_name: str = None):
    t_ppo  = ppo['t']  / 60.0
    t_bdot = bdot['t'] / 60.0

    omega_ppo_deg  = np.linalg.norm(ppo['omega'],  axis=1) * 180 / np.pi
    omega_bdot_deg = np.linalg.norm(bdot['omega'], axis=1) * 180 / np.pi
    target_deg     = omega_target * 180 / np.pi

    if x_max_min is None:
        x_max_min = max(t_ppo[-1], t_bdot[-1]) if len(t_ppo) and len(t_bdot) else 1.0

    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    fig.suptitle(title, fontsize=16)

    axes[0].semilogy(t_ppo,  omega_ppo_deg,  color='steelblue',  lw=1.5, label='RL')
    axes[0].semilogy(t_bdot, omega_bdot_deg, color='darkorange', lw=1.5, label='Bdot', ls='--')
    axes[0].axhline(target_deg, color='red', ls=':', lw=1, label=f'Objetivo {target_deg:.2f} °/s')
    if thresholds_deg:
        for thr in thresholds_deg:
            axes[0].axhline(thr, color='purple', ls='--', lw=0.9, alpha=0.7,
                            label=f'Umbral {thr:.2f} °/s')
    axes[0].set_ylabel('|ω| real [°/s]', fontsize=16)
    axes[0].legend(loc='upper right', fontsize=16)
    axes[0].grid(True, alpha=0.4)

    axes[1].plot(t_ppo,  ppo['m'],  color='steelblue',  lw=1.5, label='RL')
    axes[1].plot(t_bdot, bdot['m'], color='darkorange',  lw=1.5, label='Bdot', ls='--')
    axes[1].axhline(p.M_MAX_MTQ * np.sqrt(3), color='gray', ls=':', lw=1, label=f'|m|_max = {p.M_MAX_MTQ}·√3 ≈ {p.M_MAX_MTQ * np.sqrt(3):.1f} A·m²')
    axes[1].set_xlabel('Tiempo [min]', fontsize=16)
    axes[1].set_ylabel('|m| [A·m²]', fontsize=16)
    axes[1].legend(loc='upper right', fontsize=16)
    axes[1].grid(True, alpha=0.4)

    for ax in axes:
        ax.set_xlim([0, x_max_min])
    _add_orbit_lines(axes, x_max_min)

    plt.tight_layout()
    Path("outputs").mkdir(parents=True, exist_ok=True)
    suffix    = f"_{run_name}" if run_name else ""
    save_path = f"outputs/comparison{suffix}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Gráfica de comparación guardada en: {save_path}")
    plt.close()


def _plot_validate_omega(t_arr, omega_arr, title=""):
    """Una sola gráfica en escala lineal de las componentes de ω (modo validate)."""
    t_s = t_arr  

    fig, ax = plt.subplots(figsize=(10, 5))
    if title:
        ax.set_title(title, fontsize=12)

    colors = ['steelblue', 'darkorange', 'forestgreen']
    labels = [r'$\omega_x$', r'$\omega_y$', r'$\omega_z$']
    for i, (col, lbl) in enumerate(zip(colors, labels)):
        ax.plot(t_s, omega_arr[:, i], color=col, lw=1.2, label=lbl)

    k = 1
    while k * p.T_ORB < t_s[-1]:
        ax.axvline(k * p.T_ORB, color='gray', ls='--', lw=0.7, alpha=0.5)
        k += 1

    ax.set_xlim(0, 30000)
    ax.set_xlabel('Tiempo [s]')
    ax.set_ylabel('Velocidad angular [rad/s]')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    Path("outputs").mkdir(parents=True, exist_ok=True)
    save_path = "outputs/validate_omega.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Gráfica guardada en: {save_path}")
    plt.show(block=True)
    plt.close()


def _plot_full_episode(t_arr, omega_arr, q_arr, B_arr, m_arr,
                       omega_target, title=""):
    plot_angular_velocity(t_arr, omega_arr, omega_target=omega_target,
                          title=f"{title} — Velocidad angular" if title else "Velocidad angular")
    plot_quaternion(t_arr, q_arr,
                    title=f"{title} — Cuaternión" if title else "Cuaternión")
    plot_magnetic_field(t_arr, B_arr,
                        title=f"{title} — B_b (Body)" if title else "B_b (Body)")

    t_min          = t_arr / 60.0
    omega_norm_deg = np.linalg.norm(omega_arr, axis=1) * 180 / np.pi
    target_deg     = omega_target * 180 / np.pi

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    if title:
        fig.suptitle(title, fontsize=12)

    axes[0].semilogy(t_min, omega_norm_deg, color='steelblue', lw=1.5, label='|ω| real')
    axes[0].axhline(target_deg, color='red', ls='--', lw=1, label=f'Objetivo {target_deg:.1f} °/s')
    axes[0].set_ylabel('|ω| real [°/s]')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.4)

    ax_m = axes[1]

    ax_m.plot(t_min, m_arr, color='darkorange', lw=1.5, label='|m|')
    ax_m.axhline(p.M_MAX_MTQ * np.sqrt(3), color='gray', ls=':', lw=1, label=f'|m|_max = {p.M_MAX_MTQ}·√3 ≈ {p.M_MAX_MTQ * np.sqrt(3):.1f} A·m²')
    ax_m.set_xlabel('Tiempo [min]')
    ax_m.set_ylabel('|m| [A·m²]')
    ax_m.legend(loc='upper right')
    ax_m.grid(True, alpha=0.4)

    plt.tight_layout()
    Path("outputs").mkdir(parents=True, exist_ok=True)
    save_path = "outputs/full_episode.png"
    plt.savefig(save_path, dpi=150)
    print(f"  Gráfica de episodio guardada en: {save_path}")
    plt.show(block=True)
    plt.close()


def run_training(algo: str            = "recurrent_ppo",
                 total_timesteps: int = 1_500_000,
                 n_envs: int          = 4,
                 run_name: str        = "rppo_v1",
                 seed: int            = 42,
                 resume: str          = None,
                 device: str          = "cpu",
                 omega_detumble_thresh: float= 0.0025,
                 lr: float            = 0.000181687604274834,   
                 gamma: float         = 0.9961048095728544,
                 gae_lambda: float    = 0.9618800974145757,    
                 clip_range: float    = 0.2,
                 ent_coef: float      = 0.0039946354712228176,   
                 vf_coef: float       = 0.33832927382048866,     
                 n_steps: int         = 512,
                 batch_size: int      = 256,                     
                 n_epochs: int        = 5,
                 max_grad_norm: float = 0.8650252441478041,    
                 max_steps: int       = 15000,
                 lambda_smooth: float = 0.08,
                 gaussian_amplitude: float = 0.05,
                 gaussian_sigma: float     = 0.02,
                 history_length: int       = 0):
    
    import torch

    if history_length > 0 and algo == "recurrent_ppo":
        print(f"[ERROR] --history-length no es compatible con --algo recurrent_ppo "
              f"(el LSTM ya gestiona la memoria temporal; combinar ambos es redundante).")
        sys.exit(1)

    AlgoClass, policy_name = _load_algo(algo)
    device = _resolve_device(device)

    if algo == "sac":
        lr         = 0.00010938754707944945
        gamma      = 0.999
        batch_size = 512

    kwargs = _POLICY_KWARGS[algo].copy()
    if kwargs.get("activation_fn") is None:
        kwargs["activation_fn"] = torch.nn.Tanh

    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    tb_dir  = str(run_dir / "tb")

    _print_header(f"Entrenamiento {algo.upper()} — {run_name}")
    print(f"  algo                : {algo}")
    print(f"  timesteps           : {total_timesteps:,}")
    print(f"  n_envs              : {n_envs}")
    print(f"  gamma               : {gamma}")
    print(f"  max_steps/ep        : {max_steps}  ({max_steps/60:.0f} min)")
    print(f"  lambda_smooth       : {lambda_smooth}")
    print(f"  omega_detumble_thresh : {omega_detumble_thresh} rad/s ({omega_detumble_thresh*180/3.14159:.2f}°/s)")
    print(f"  device              : {device}")
    if history_length > 0:
        stacked_dim = 9 * history_length
        print(f"  obs space           : 6D base [B,Ḃ] + HistoryWrapper(N={history_length}) "
              f"→ {stacked_dim}D apilada ({algo.upper()}+wrappers)")
    else:
        print(f"  obs space           : 12D [B_curr, B_prev, Bdot, a_prev] — solo magnetómetro")
    print(f"  tensorboard         : tensorboard --logdir {tb_dir}")

    from stable_baselines3.common.env_checker import check_env
    print("\n  Verificando entorno...")
    check_env(_make_env(seed=seed, max_steps=max_steps, lambda_smooth=lambda_smooth,
                        history_length=history_length)(), warn=True)
    print("  Entorno OK.\n")

    from stable_baselines3.common.env_util  import make_vec_env
    from stable_baselines3.common.vec_env   import VecNormalize
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

    env_kwargs = dict(omega_detumble_thresh=omega_detumble_thresh,
                      max_steps=max_steps,
                      lambda_smooth=lambda_smooth,
                      gaussian_amplitude=gaussian_amplitude,
                      gaussian_sigma=gaussian_sigma,
                      history_length=history_length)

    train_env = make_vec_env(_make_env(seed=seed, **env_kwargs), n_envs=n_envs, seed=seed)
    eval_env  = make_vec_env(_make_env(seed=seed + 1000,
                                       omega_detumble_thresh=omega_detumble_thresh,
                                       max_steps=max_steps,
                                       lambda_smooth=lambda_smooth,
                                       gaussian_amplitude=gaussian_amplitude,
                                       gaussian_sigma=gaussian_sigma,
                                       history_length=history_length),
                             n_envs=1, seed=seed + 1000)

    is_sac = (algo == "sac")

    if resume:
        resume_path = Path(resume)
        norm_path   = resume_path.parent / "vecnormalize.pkl"
        if not norm_path.exists():
            print(f"  [ERROR] No se encuentra {norm_path}")
            sys.exit(1)

        train_env = VecNormalize.load(str(norm_path), train_env)
        train_env.training    = True
        train_env.norm_reward = True

        eval_env = VecNormalize.load(str(norm_path), eval_env)
        eval_env.training    = False
        eval_env.norm_reward = False

        load_kwargs = dict(tensorboard_log=tb_dir, verbose=1, learning_rate=lr)
        if not is_sac:
            load_kwargs.update(clip_range=clip_range, ent_coef=ent_coef,
                               max_grad_norm=max_grad_norm)

        model = AlgoClass.load(str(resume_path), env=train_env, device=device,
                               **load_kwargs)
        eval_env.clip_obs = train_env.clip_obs
        print(f"  Reanudando desde : {resume_path}.zip")
        print(f"  Pasos realizados : {model.num_timesteps:,}")
        reset_num_timesteps = False

    else:
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True,
                                 clip_obs=10.0, clip_reward=15.0, gamma=gamma)
        eval_env  = VecNormalize(eval_env,  norm_obs=True, norm_reward=False,
                                 clip_obs=10.0, gamma=gamma, training=False)

        if is_sac:
            model = AlgoClass(
                policy          = policy_name,
                env             = train_env,
                device          = device,
                learning_rate   = lr,
                buffer_size     = 300_000,
                learning_starts = 10_000,
                batch_size      = batch_size,
                gamma           = gamma,
                tau             = 0.005,
                ent_coef        = "auto",
                policy_kwargs   = kwargs,
                tensorboard_log = tb_dir,
                verbose         = 1,
                seed            = seed,
            )
        else:
            model = AlgoClass(
                policy          = policy_name,
                env             = train_env,
                device          = device,
                learning_rate   = lr,
                n_steps         = n_steps,
                batch_size      = batch_size,
                n_epochs        = n_epochs,
                gamma           = gamma,
                gae_lambda      = gae_lambda,
                clip_range      = clip_range,
                ent_coef        = ent_coef,
                vf_coef         = vf_coef,
                max_grad_norm   = max_grad_norm,
                policy_kwargs   = kwargs,
                tensorboard_log = tb_dir,
                verbose         = 1,
                seed            = seed,
            )
        reset_num_timesteps = True

    callbacks = CallbackList([
        EvalCallback(eval_env,
                     best_model_save_path = str(run_dir),
                     log_path             = str(run_dir / "eval_logs"),
                     eval_freq            = max(40_000 // n_envs, 1),
                     n_eval_episodes      = 5,
                     deterministic        = True,
                     render               = False),
        CheckpointCallback(save_freq   = max(200_000 // n_envs, 1),
                           save_path   = str(run_dir),
                           name_prefix = "checkpoint"),
    ])

    print(f"  Iniciando entrenamiento...\n")
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callbacks,
                reset_num_timesteps=reset_num_timesteps, progress_bar=True)
    elapsed = (time.time() - t0) / 60.0
    print(f"\n  Entrenamiento completado en {elapsed:.1f} min.")

    suffix     = "resumed" if resume else "final"
    model_path = str(run_dir / f"{run_name}_{suffix}")
    model.save(model_path)
    train_env.save(str(run_dir / "vecnormalize.pkl"))

    print(f"  Modelo guardado  : {model_path}.zip")
    print(f"  Pasos totales    : {model.num_timesteps:,}")
    print(f"\n  Para evaluar:")
    print(f"    python main.py evaluate --model {run_dir/'best_model'} --algo {algo}")
    print(f"  Para continuar:")
    print(f"    python main.py train --algo {algo} --run-name {run_name} "
          f"--resume {run_dir/'best_model'} --timesteps 500000")

    train_env.close()
    eval_env.close()


def _run_bdot_sim(q0: np.ndarray, omega0: np.ndarray,
                  seed: int = 42, k_bdot: float = 1e6,
                  dt: float = 1.0, max_steps: int = 17500,
                  r_I_0: np.ndarray = None, t_0: float = 0.0) -> dict:
    """
    Simula Bdot clásico con las mismas condiciones iniciales que el agente RL.
    r_I_0 / t_0 permiten igualar las ICs orbitales al entorno DRL.
    """
    if max_steps is None:
        max_steps = int(3 * p.T_ORB / dt)

    env = UPMSat2Env(dt=dt, max_steps=max_steps, seed=seed)
    env.reset(seed=seed)
    env._state = np.concatenate([quat_normalize(q0), omega0.copy()])
    if r_I_0 is not None:
        env._r_I = r_I_0.copy()
    env._t    = t_0
    C_BI          = quat_to_dcm(env._state[:4])
    B_b_true_0    = dipole_field_body(env._r_I, C_BI, env._t)
    env._B_b_prev = fused_magnetometer(B_b_true_0, rng=env._rng)
    env._B_b_meas_cache = env._B_b_prev.copy()
    env._B_dot_cache    = np.zeros(3)

    t_hist, omega_hist, q_hist, B_hist, m_hist, align_hist = \
        [], [], [], [], [], []

    done = False
    while not done:
        C_BI  = quat_to_dcm(env._state[:4])
        B_b   = dipole_field_body(env._r_I, C_BI, env._t)
        B_dot = bdot_numerical(env._B_b_prev, B_b, env.dt)
        m_raw = np.clip(-k_bdot * B_dot, -p.M_MAX_MTQ, p.M_MAX_MTQ)
        action = (m_raw / p.M_MAX_MTQ).astype(np.float32)

        _, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        t_hist.append(info['t'])
        omega_hist.append(env._state[4:].copy())
        q_hist.append(env._state[:4].copy())
        B_hist.append(B_b.copy())
        m_hist.append(info['m_norm'])
        align_hist.append(info['alignment'])

    env.close()
    return {
        't':           np.array(t_hist),
        'omega':       np.array(omega_hist),
        'q':           np.array(q_hist),
        'B':           np.array(B_hist),
        'm':           np.array(m_hist),
        'alignment':   np.array(align_hist),
    }


def run_validation(n_orbits: int  = 3,
                   k_bdot:   float = 1e6,
                   dt:       float = 1.0,
                   plot:     bool  = True,
                   save:     bool  = False,
                   open_loop: bool = False):
    """Simula el controlador Bdot clásico. Línea base para comparar con RL.

    Con open_loop=True se fuerza m=0 (sin control), útil para caracterizar
    la dinámica y perturbaciones del satélite en lazo abierto.
    """
    if open_loop:
        _print_header(f"Validación lazo abierto  m=0  |  {n_orbits} órbitas")
    else:
        _print_header(f"Validación Bdot  k={k_bdot:.1e}  |  {n_orbits} órbitas")

    max_steps = int(17500 / dt)
    env = UPMSat2Env(dt=dt, max_steps=max_steps, seed=42)

    print(f"  T_orb     = {p.T_ORB/60:.1f} min")
    print(f"  max_steps = {max_steps}")
    print(f"  omega_0   = {np.rad2deg(p.OMEGA_BI_B0)} °/s")

    env.reset(seed=42)
    env._state    = np.concatenate([quat_normalize(p.QUAT_0), p.OMEGA_BI_B0.copy()])
    C_BI          = quat_to_dcm(env._state[:4])
    B_b_true_0    = dipole_field_body(env._r_I, C_BI, env._t)
    env._B_b_prev = fused_magnetometer(B_b_true_0, rng=env._rng)
    env._B_b_meas_cache = env._B_b_prev.copy()
    env._B_dot_cache    = np.zeros(3)

    print(f"  Delta t: {dt} s")
    print(f"  k_bdot: {k_bdot}")
    print(f"  omega_0: {p.OMEGA_BI_B0 * 180/np.pi} deg/s")
    print(f"  quat_0: {p.QUAT_0}")
    print(f"  B_body inicial: {B_b_true_0 * 1e9} nT")
    print(f"  Saturacion magnetopares: {p.M_MAX_MTQ} A·m²")
    print(f"  Dipolo residual: {p.RMD_B}")

    t_hist, omega_hist, q_hist, B_hist, m_hist = [], [], [], [], []
    state_hist, obs_hist, action_hist, reward_hist = [], [], [], []

    done = False
    while not done:
        C_BI  = quat_to_dcm(env._state[:4])
        B_b   = dipole_field_body(env._r_I, C_BI, env._t)
        if open_loop:
            action = np.zeros(3, dtype=np.float32)
        else:
            B_dot = bdot_numerical(env._B_b_prev, B_b, env.dt)
            m_raw = np.clip(-k_bdot * B_dot, -p.M_MAX_MTQ, p.M_MAX_MTQ)
            action = (m_raw / p.M_MAX_MTQ).astype(np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        t_hist.append(info['t'])
        omega_hist.append(env._state[4:].copy())
        q_hist.append(env._state[:4].copy())
        B_hist.append(B_b.copy())
        m_hist.append(info['m_norm'])
        state_hist.append(env._state.copy())
        obs_hist.append(obs.copy())
        action_hist.append(action.copy())
        reward_hist.append(reward)

        if int(info['t']) % 600 == 0:
            print(f"  t={info['t']/60:5.1f} min  "
                  f"|ω|_real={info['omega_deg_s']:7.3f} °/s  "
                  f"|m|={info['m_norm']:5.2f} A·m²")

    t_arr     = np.array(t_hist)
    omega_arr = np.array(omega_hist)
    q_arr     = np.array(q_hist)
    B_arr     = np.array(B_hist)
    m_arr     = np.array(m_hist)

    torque_dummy = np.zeros_like(omega_arr)
    metrics = episode_metrics(
        t_arr, omega_arr, torque_dummy,
        omega_threshold = env.omega_target,
        I               = env._inertia,
    )

    THRESHOLD_DEG = 0.3
    omega_deg_arr = np.linalg.norm(omega_arr, axis=1) * 180.0 / np.pi
    idx_cross = np.argwhere(omega_deg_arr < THRESHOLD_DEG)
    t_cross = f"{t_arr[idx_cross[0, 0]]:.1f} s" if len(idx_cross) else "no alcanzado"
    omega_final_deg = omega_deg_arr[-1]
    B_final_nT = B_arr[-1] * 1e9  # T → nT
    q_final = q_arr[-1]

    _print_header("RESULTADOS")
    print(f"  Tiempo de detumbling : {metrics['t_detumble']} s")
    print(f"  Energía disipada     : {metrics['energy_dissipated']:.4f} J")
    print(f"  Esfuerzo de control  : {metrics['control_effort']:.4e} N·m·s")
    print()
    print(f"  ── Métricas de interés ──────────────────────────")
    print(f"  t cruce |ω| < {THRESHOLD_DEG} °/s : {t_cross}")
    print(f"  |ω|_real final          : {omega_final_deg:.4f} °/s")
    print(f"  B_b final  [nT]  @ t={t_arr[-1]:.1f}s : Bx={B_final_nT[0]:.1f}  By={B_final_nT[1]:.1f}  Bz={B_final_nT[2]:.1f}")
    print(f"  |B_b| final [nT]        : {np.linalg.norm(B_final_nT):.1f}")
    print(f"  Cuaternión final @ t={t_arr[-1]:.1f}s   : "
          f"q0={q_final[0]:.6f}  q1={q_final[1]:.6f}  q2={q_final[2]:.6f}  q3={q_final[3]:.6f}")

    _print_results(t_arr,
                   np.linalg.norm(omega_arr, axis=1) * 180 / np.pi,
                   m_arr,
                   omega_target_deg = np.rad2deg(env.omega_target),
                   label            = "Bdot clásico")

    if plot:
        _plot_validate_omega(t_arr, omega_arr,
                             title = f"Bdot clásico  k={k_bdot:.1e} — Velocidad angular")
    if save:
        save_trajectory("outputs/validation_bdot",
                        t_arr, np.array(state_hist),
                        np.array(obs_hist), np.array(action_hist),
                        np.array(reward_hist))
        print("  Trayectoria guardada en outputs/validation_bdot.npz")

    env.close()
    return t_arr, omega_arr


def run_evaluation(algo: str,
                   model_path: str,
                   n_episodes: int  = 5,
                   omega_thresh: float = 0.0025,
                   plot:       bool = True,
                   plot_mode:  str  = "full",
                   seed:       int  = None,
                   device:     str  = "cpu",
                   history_length: int = 0,
                   n_orbits:   int  = 7,
                   steady_omega: bool = False):
    """
    plot_mode : "full"       → gráficas completas (vel, cuaternión, B)
                "comparison" → RL vs Bdot clásico
    """
    AlgoClass, _ = _load_algo(algo)

    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    if seed is None:
        seed = np.random.randint(0, 10000)

    model_path = Path(model_path)
    norm_path  = model_path.parent / "vecnormalize.pkl"

    _print_header(f"Evaluación {algo.upper()} — {model_path.name}")

    eval_env = DummyVecEnv([_make_env(seed=seed, omega_detumble_thresh=omega_thresh,
                                      history_length=history_length)])
    if norm_path.exists():
        eval_env = VecNormalize.load(str(norm_path), eval_env)
        eval_env.training    = False
        eval_env.norm_reward = False
        print(f"  Normalización cargada : {norm_path}")
    else:
        print(f"  [AVISO] vecnormalize.pkl no encontrado — sin normalización.")
    print(f"  Semilla de evaluación : {seed}")

    model = AlgoClass.load(str(model_path), env=eval_env, device=device)
    print(f"  Modelo cargado: {model_path}.zip\n")

    all_results   = []
    success_count = 0
    is_recurrent  = algo == "recurrent_ppo"

    for ep in range(n_episodes):
        obs  = eval_env.reset()
        done = False
        lstm_state     = None
        episode_starts = np.ones((eval_env.num_envs,), dtype=bool)
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[None, :]

        ep_t, ep_omega, ep_q, ep_B, ep_m, ep_reward = \
            [], [], [], [], [], []
        ep_action, ep_bdot = [], []

        while not done:
            if is_recurrent:
                action, lstm_state = model.predict(
                    obs, state=lstm_state,
                    episode_start=episode_starts, deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)   
            obs, reward, done_arr, info_arr = eval_env.step(action)
            done           = bool(done_arr[0])
            episode_starts = done_arr
            info           = info_arr[0]

            ep_t.append(info['t'])
            ep_m.append(info['m_norm'])
            ep_reward.append(float(reward[0]))
            ep_omega.append(np.array(
                [info['omega_x'], info['omega_y'], info['omega_z']], dtype=float))
            ep_q.append(np.array(info['q'], dtype=float))
            ep_B.append(np.array(
                [info.get('B_x', 0), info.get('B_y', 0), info.get('B_z', 0)], dtype=float))
            ep_action.append(action[0].copy())
            ep_bdot.append(info['bdot'].copy())

        final_omega_deg  = np.linalg.norm(ep_omega[-1]) * 180 / np.pi
        
        omega_target_deg = np.rad2deg(eval_env.envs[0].unwrapped.omega_target)
        success = final_omega_deg < omega_target_deg
        if success:
            success_count += 1

        print(f"  Ep {ep+1:2d}/{n_episodes}  |  "
              f"pasos={len(ep_t):5d}  |  "
              f"|ω|_final={final_omega_deg:6.2f} °/s  |  "
              f"reward={sum(ep_reward):8.1f}  |  "
              f"{'✓ ÉXITO' if success else '✗ fallo'}")

        all_results.append({
            't':           np.array(ep_t),
            'omega':       np.array(ep_omega),
            'q':           np.array(ep_q),
            'B':           np.array(ep_B),
            'm':           np.array(ep_m),
            'action':      np.array(ep_action),
            'bdot':        np.array(ep_bdot),
        })

    print(f"\n  Tasa de éxito: {success_count}/{n_episodes} "
          f"({100*success_count/n_episodes:.0f}%)")

    if steady_omega:
        _STEADY_THRESH_RAD = np.deg2rad(0.3)
        post_vals = []
        for r in all_results:
            omega_norm = np.linalg.norm(r['omega'], axis=1)
            idx = np.argmax(omega_norm < _STEADY_THRESH_RAD)
            if omega_norm[idx] < _STEADY_THRESH_RAD:
                post_vals.extend(omega_norm[idx:].tolist())
        if post_vals:
            print(f"  |ω| medio tras umbral 0.3 °/s : "
                  f"{np.mean(post_vals)*180/np.pi:.4f} °/s  "
                  f"({len(post_vals)} pasos)")
        else:
            print("  |ω| medio tras umbral 0.3 °/s : no se alcanzó en ningún episodio")

    mean_m_drl = np.mean(np.concatenate([r['m'] for r in all_results]))
    print(f"  Uso medio magnetopares (DRL) : {mean_m_drl:.4f} A·m²")

    
    actions = np.concatenate([r['action'] for r in all_results], axis=0)
    bdot    = np.concatenate([r['bdot']   for r in all_results], axis=0)
    B_arr   = np.concatenate([r['B']      for r in all_results], axis=0)

    print("\n  Estadísticas por eje — campo magnético (body frame):")
    print(f"    {'eje':>4}  {'std(B)':>12}  {'std(Ḃ)':>12}  {'mean|B|':>10}  {'mean|Ḃ|':>10}")
    for i, eje in enumerate(['x', 'y', 'z']):
        print(f"    {eje:>4}  {np.std(B_arr[:,i]):>12.3e}  "
              f"{np.std(bdot[:,i]):>12.3e}  "
              f"{np.mean(np.abs(B_arr[:,i])):>10.3e}  "
              f"{np.mean(np.abs(bdot[:,i])):>10.3e}")

    print("\n  Estadísticas por eje — acción:")
    print(f"    {'eje':>4}  {'std(a)':>10}  {'mean|a|':>10}  {'max|a|':>10}")
    for i, eje in enumerate(['x', 'y', 'z']):
        print(f"    {eje:>4}  {np.std(actions[:,i]):>10.4f}  "
              f"{np.mean(np.abs(actions[:,i])):>10.4f}  "
              f"{np.max(np.abs(actions[:,i])):>10.4f}")

    print("\n  Correlación acción vs -Ḃ (ley Bdot pura → 1.0):")
    for i, eje in enumerate(['x', 'y', 'z']):
        corr = np.corrcoef(actions[:, i], -bdot[:, i])[0, 1]
        print(f"    acción_{eje} vs -Ḃ_{eje} : {corr:.3f}")

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"Interpretabilidad — {algo.upper()} {model_path.name}")
    for i, eje in enumerate(['x', 'y', 'z']):
        bdot_col = bdot[:, i]
        bdot_norm = bdot_col / (np.max(np.abs(bdot_col)) + 1e-12)
        axes[i].scatter(-bdot_norm, actions[:, i], alpha=0.1, s=1)
        axes[i].set_xlabel(r'$-\dot{B}_' + eje + r'$ (norm.)')
        axes[i].set_ylabel(r'$a_' + eje + r'$ (acción agente)')
        axes[i].set_title(f'Eje {eje.upper()}')
    plt.tight_layout()
    Path("outputs").mkdir(parents=True, exist_ok=True)
    interp_path = f"outputs/interpretabilidad_{algo}_{model_path.stem}.png"
    plt.savefig(interp_path, dpi=150)
    print(f"  Figura guardada en: {interp_path}")
    plt.close()

    
    B_arr  = np.concatenate([r['B'] for r in all_results], axis=0)
    B_norm = np.linalg.norm(B_arr, axis=1)
    dBdt   = np.einsum('ij,ij->i', B_arr, bdot) / (B_norm + 1e-30)   
    a_norm = np.linalg.norm(actions, axis=1)

    corr_total = np.corrcoef(dBdt, a_norm)[0, 1]
    print(f"\n  Correlación d|B|/dt vs |acción total| : {corr_total:.3f}")

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    fig2.suptitle(f"Correlación campo magnético–acción — {algo.upper()} {model_path.name}")
    dBdt_norm  = dBdt  / (np.max(np.abs(dBdt))  + 1e-30)
    anorm_norm = a_norm / (np.max(a_norm)         + 1e-30)
    ax2.scatter(dBdt_norm, anorm_norm, alpha=0.15, s=2, color='steelblue')
    ax2.set_xlabel(r'$\frac{d|\mathbf{B}|}{dt}$ (norm.)')
    ax2.set_ylabel(r'$|\mathbf{a}|$ (norm.)')
    ax2.set_title(fr'$\rho$ = {corr_total:.3f}')
    plt.tight_layout()
    corr_path = f"outputs/corr_dBdt_accion_{algo}_{model_path.stem}.png"
    plt.savefig(corr_path, dpi=150)
    print(f"  Figura guardada en: {corr_path}")
    plt.close()

    env_ref = eval_env.envs[0].unwrapped

    if plot and all_results:
        if plot_mode == "comparison":
            _run_comparison_plot(
                algo           = algo,
                model          = model,
                model_path     = model_path,
                norm_path      = norm_path,
                seed           = seed,
                omega_thresh   = omega_thresh,
                history_length = history_length,
                is_recurrent   = is_recurrent,
                n_orbits       = n_orbits,
                omega_target   = env_ref.omega_target,
            )
        else:
            r = all_results[-1]
            _plot_full_episode(
                t_arr           = r['t'],
                omega_arr       = r['omega'],
                q_arr           = r['q'],
                B_arr           = r['B'],
                m_arr           = r['m'],
                omega_target    = env_ref.omega_target,
                title           = f"{algo.upper()} — {model_path.name}",
            )
    eval_env.close()


def _run_comparison_plot(algo, model, model_path, norm_path,
                         seed, omega_thresh, history_length,
                         is_recurrent, n_orbits, omega_target):
    """
    Corre RL y Bdot durante n_orbits con ICs idénticas, sin depender
    del episodio de evaluación (max_steps del eval_env puede ser distinto).
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    n7        = int(n_orbits * p.T_ORB)
    x_max_min = n_orbits * p.T_ORB / 60.0

    _print_header(f"Comparación {n_orbits} órbitas ({n7} pasos)")


    cmp_env = DummyVecEnv([_make_env(seed=seed,
                                     omega_detumble_thresh=omega_thresh,
                                     max_steps=n7,
                                     history_length=history_length)])
    if norm_path.exists():
        cmp_env = VecNormalize.load(str(norm_path), cmp_env)
        cmp_env.training    = False
        cmp_env.norm_reward = False

    obs_cmp = cmp_env.reset()

    inner     = cmp_env.envs[0].unwrapped
    q0_cmp    = inner._state[:4].copy()
    omega0_cmp = inner._state[4:].copy()
    r_I_0_cmp  = inner._r_I.copy()
    t_0_cmp    = float(inner._t)

    print(f"  |ω₀| = {np.linalg.norm(omega0_cmp)*180/np.pi:.4f} °/s  "
          f"|q₀|={np.linalg.norm(q0_cmp):.6f}  "
          f"t₀={t_0_cmp:.1f} s")

    drl_t, drl_omega, drl_m = [], [], []
    done_cmp       = False
    lstm_state_cmp = None
    ep_starts_cmp  = np.ones((1,), dtype=bool)

    while not done_cmp:
        if is_recurrent:
            action_cmp, lstm_state_cmp = model.predict(
                obs_cmp, state=lstm_state_cmp,
                episode_start=ep_starts_cmp, deterministic=True)
        else:
            action_cmp, _ = model.predict(obs_cmp, deterministic=True)

        obs_cmp, _, done_arr_cmp, info_arr_cmp = cmp_env.step(action_cmp)
        done_cmp       = bool(done_arr_cmp[0])
        ep_starts_cmp  = done_arr_cmp
        info_cmp       = info_arr_cmp[0]

        drl_t.append(info_cmp['t'])
        drl_omega.append([info_cmp['omega_x'], info_cmp['omega_y'], info_cmp['omega_z']])
        drl_m.append(info_cmp['m_norm'])

    cmp_env.close()

    drl_data = {
        't':           np.array(drl_t),
        'omega':       np.array(drl_omega, dtype=float),
        'm':           np.array(drl_m),
    }
    print(f"  RL:   {len(drl_t)} pasos  |ω|_final={np.linalg.norm(drl_omega[-1])*180/np.pi:.4f} °/s")

    print(f"  Corriendo Bdot {n_orbits} órbitas ({n7} pasos)...")
    bdot_data = _run_bdot_sim(
        q0=q0_cmp, omega0=omega0_cmp,
        seed=seed, k_bdot=1e6, dt=1.0,
        max_steps=n7,
        r_I_0=r_I_0_cmp, t_0=t_0_cmp,
    )
    print(f"  Bdot: {len(bdot_data['t'])} pasos  |ω|_final={np.linalg.norm(bdot_data['omega'][-1])*180/np.pi:.4f} °/s")

    print(f"\n  Uso medio magnetopares — RL   : {np.mean(drl_data['m']):.4f} A·m²")
    print(f"  Uso medio magnetopares — Bdot : {np.mean(bdot_data['m']):.4f} A·m²")

    _crossing_thresholds = [0.3]
    print(f"\n  {'─'*50}")
    print(f"  Tiempos de primer cruce de umbral:")
    for thr in _crossing_thresholds:
        t_rl   = _first_crossing_min(drl_data['t'],   drl_data['omega'],                 thr)
        t_bdot = _first_crossing_min(bdot_data['t'],  bdot_data['omega'],                thr)
        print(f"    {thr:.2f} °/s  →  RL: {t_rl}   |   Bdot: {t_bdot}")
    print(f"  {'─'*50}\n")

    _plot_comparison(
        ppo               = drl_data,
        bdot              = bdot_data,
        omega_target      = np.deg2rad(0.143),   # 0.143 °/s
        title             = f"{algo.upper()} vs Bdot clásico — {model_path.name}",
        x_max_min         = x_max_min,
        thresholds_deg    = _crossing_thresholds,
        run_name          = model_path.parent.name,
    )


def export_onnx(model_path: str, output_path: str = None, verify: bool = True):
    """ Solo compatible con PPO estándar (MlpPolicy)."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        import torch
    except ImportError as e:
        print(f"[ERROR] {e}\n        pip install stable-baselines3 torch")
        sys.exit(1)

    try:
        import onnxruntime as ort
        has_ort = True
    except ImportError:
        has_ort = False
        if verify:
            print("  [AVISO] onnxruntime no instalado — pip install onnx onnxruntime")

    model_path      = Path(model_path)
    norm_path       = model_path.parent / "vecnormalize.pkl"
    output_path     = Path(output_path) if output_path else \
                      model_path.parent / (model_path.stem + ".onnx")
    norm_stats_path = output_path.with_name(output_path.stem + "_norm_stats.npz")

    _print_header(f"Exportar a ONNX — {model_path.name}")

    dummy_env = DummyVecEnv([_make_env(seed=0)])
    if norm_path.exists():
        dummy_env = VecNormalize.load(str(norm_path), dummy_env)
        dummy_env.training    = False
        dummy_env.norm_reward = False
        print(f"  Normalización cargada: {norm_path}")

    model  = PPO.load(str(model_path), env=dummy_env)
    policy = model.policy
    policy.eval()

    class ActorOnly(torch.nn.Module):
        def __init__(self, pol):
            super().__init__()
            self.features_extractor = pol.features_extractor
            self.mlp_policy_net     = pol.mlp_extractor.policy_net
            self.action_net         = pol.action_net

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            features = self.features_extractor(obs)
            latent   = self.mlp_policy_net(features)
            return torch.tanh(self.action_net(latent))

    actor     = ActorOnly(policy)
    actor.eval()
    obs_dim   = model.observation_space.shape[0]
    dummy_obs = torch.zeros(1, obs_dim, dtype=torch.float32)

    print(f"  obs_dim = {obs_dim}D")
    print(f"  Exportando a {output_path}...")
    torch.onnx.export(
        actor, dummy_obs, str(output_path),
        export_params       = True,
        opset_version       = 17,
        do_constant_folding = True,
        input_names         = ["obs"],
        output_names        = ["action"],
        dynamic_axes        = {"obs": {0: "batch"}, "action": {0: "batch"}},
    )
    print(f"  ONNX guardado.")

    if norm_path.exists():
        np.savez(str(norm_stats_path),
                 obs_mean = dummy_env.obs_rms.mean.astype(np.float32),
                 obs_var  = dummy_env.obs_rms.var.astype(np.float32),
                 clip_obs = np.array([float(dummy_env.clip_obs)]))
        print(f"  Normalización guardada: {norm_stats_path}")

    if verify and has_ort:
        test_obs = torch.randn(1, obs_dim)
        with torch.no_grad():
            act_pt = actor(test_obs).numpy()
        act_onnx = ort.InferenceSession(str(output_path)).run(
            ["action"], {"obs": test_obs.numpy()})[0]
        err = float(np.max(np.abs(act_pt - act_onnx)))
        print(f"  Verificación — error máx PyTorch vs ONNX: {err:.2e}",
              "✓" if err < 1e-5 else "⚠ revisar")

    dummy_env.close()


def _parse():
    parser = argparse.ArgumentParser(
        description="UPMSat-2 RL — Detumbling con observación magnética (12D)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    tr = sub.add_parser("train", help="Entrenar agente PPO o RecurrentPPO")
    tr.add_argument("--algo",         type=str,   default="recurrent_ppo",
                    choices=["ppo", "recurrent_ppo", "sac"],
                    help="Algoritmo: 'ppo', 'recurrent_ppo' o 'sac'")
    tr.add_argument("--omega-thresh", type=float, default=0.0025,
                help="Umbral de éxito en rad/s (0.0025 = 0.14°/s)")
    tr.add_argument("--timesteps",    type=int,   default=1_500_000)
    tr.add_argument("--n-envs",       type=int,   default=4)
    tr.add_argument("--run-name",     type=str,   default="rppo_v1")
    tr.add_argument("--seed",         type=int,   default=42)
    tr.add_argument("--resume",       type=str,   default=None)
    tr.add_argument("--device",       type=str,   default="cpu", choices=["cpu", "cuda"])
    tr.add_argument("--lr",            type=float, default=0.00010938754707944945)
    tr.add_argument("--gamma",         type=float, default=0.999,
                    help="Factor de descuento. Usa 0.999 para episodios de ~15000 pasos.")
    tr.add_argument("--gae-lambda",    type=float, default=0.9896265362770728)
    tr.add_argument("--clip-range",    type=float, default=0.2)
    tr.add_argument("--ent-coef",      type=float, default=0.0013133328618022328)    
    tr.add_argument("--vf-coef",       type=float, default=0.6976369958223515)        
    tr.add_argument("--n-steps",       type=int,   default=512)
    tr.add_argument("--batch-size",    type=int,   default=512)
    tr.add_argument("--n-epochs",      type=int,   default=10)
    tr.add_argument("--max-grad-norm", type=float, default=0.48889263323130994)
    tr.add_argument("--max-steps",     type=int,   default=15000,
                    help="Pasos máximos por episodio. 15000 ≈ 1 órbita completa.")
    tr.add_argument("--lambda-smooth", type=float, default=0.08,
                    help="Peso de penalización de variación de acción (suavizado).")
    tr.add_argument("--gaussian-amplitude", type=float, default=0.05,
                    help="Amplitud del bonus gaussiano (default: 0.05).")
    tr.add_argument("--gaussian-sigma", type=float, default=0.02,
                    help="Ancho de la gaussiana en rad/s (default: 0.02).")
    tr.add_argument("--history-length", type=int, default=0,
                    help="Pasos previos apilados (0=off, entorno 12D clásico). "
                         "Si >0: HistoryWrapper con entorno 9D — compatible con ppo y sac. "
                         "No compatible con recurrent_ppo (el LSTM ya gestiona memoria).")

    va = sub.add_parser("validate", help="Línea base con Bdot clásico")
    va.add_argument("--n-orbits", type=int,   default=3)
    va.add_argument("--k-bdot",   type=float, default=1e6)
    va.add_argument("--dt",       type=float, default=1.0)
    va.add_argument("--no-plot",  action="store_true")
    va.add_argument("--save",     action="store_true")
    va.add_argument("--open-loop", action="store_true",
                    help="Lazo abierto: m=0 (sin magnetopares, sin control Bdot).")

    ev = sub.add_parser("evaluate", help="Evaluar política entrenada")
    ev.add_argument("--algo",        type=str, default="recurrent_ppo",
                    choices=["ppo", "recurrent_ppo", "sac"])
    ev.add_argument("--model",       type=str, required=True)
    ev.add_argument("--omega-thresh", type=float, default=0.0025,
                help="Umbral de éxito en rad/s (0.0025 = 0.14°/s)")
    ev.add_argument("--n-episodes",  type=int, default=5)
    ev.add_argument("--seed",        type=int, default=None)
    ev.add_argument("--no-plot",     action="store_true")
    ev.add_argument("--plot-mode",   type=str, default="full",
                    choices=["full", "comparison"])
    ev.add_argument("--device",      type=str, default="cpu", choices=["cpu", "cuda"])
    ev.add_argument("--history-length", type=int, default=0,
                    help="Debe coincidir con el usado en entrenamiento (0=off).")
    ev.add_argument("--n-orbits", type=int, default=7,
                    help="Órbitas a simular en plot-mode comparison (default: 7).")
    ev.add_argument("--steady-omega", action="store_true",
                    help="Imprime |ω| medio tras cruzar el umbral de 0.3 °/s.")

    ex = sub.add_parser("export", help="Exportar política PPO a ONNX para Simulink")
    ex.add_argument("--model",     type=str, required=True)
    ex.add_argument("--output",    type=str, default=None)
    ex.add_argument("--no-verify", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse()

    if args.mode == "train":
        run_training(
            algo              = args.algo,
            total_timesteps   = args.timesteps,
            n_envs            = args.n_envs,
            run_name          = args.run_name,
            seed              = args.seed,
            resume            = args.resume,
            device            = args.device,
            omega_detumble_thresh = args.omega_thresh,
            lr                = args.lr,
            gamma             = args.gamma,
            gae_lambda        = args.gae_lambda,
            clip_range        = args.clip_range,
            ent_coef          = args.ent_coef,
            vf_coef           = args.vf_coef,
            n_steps           = args.n_steps,
            batch_size        = args.batch_size,
            n_epochs          = args.n_epochs,
            max_grad_norm     = args.max_grad_norm,
            max_steps         = args.max_steps,
            lambda_smooth     = args.lambda_smooth,
            gaussian_amplitude= args.gaussian_amplitude,
            gaussian_sigma    = args.gaussian_sigma,
            history_length    = args.history_length,
        )

    elif args.mode == "validate":
        run_validation(
            n_orbits  = args.n_orbits,
            k_bdot    = args.k_bdot,
            dt        = args.dt,
            plot      = not args.no_plot,
            save      = args.save,
            open_loop = args.open_loop,
        )

    elif args.mode == "evaluate":
        run_evaluation(
            algo           = args.algo,
            model_path     = args.model,
            n_episodes     = args.n_episodes,
            plot           = not args.no_plot,
            plot_mode      = args.plot_mode,
            seed           = args.seed,
            device         = args.device,
            omega_thresh   = args.omega_thresh,
            history_length = args.history_length,
            n_orbits       = args.n_orbits,
            steady_omega   = args.steady_omega,
        )

    elif args.mode == "export":
        export_onnx(
            model_path  = args.model,
            output_path = args.output,
            verify      = not args.no_verify,
        )