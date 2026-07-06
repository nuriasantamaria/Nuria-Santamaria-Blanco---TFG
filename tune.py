from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environment import UPMSat2Env
from environment_ppo import UPMSat2EnvPPO
from wrappers import HistoryWrapper

try:
    from sb3_contrib import RecurrentPPO
    _HAS_RECURRENT = True
except ImportError:
    _HAS_RECURRENT = False
    RecurrentPPO = None

N_EVAL_EPISODES  = 5        
EVAL_FREQ        = 10_000   
N_STARTUP_TRIALS = 5        
N_WARMUP_STEPS   = 3       


ENV_MAX_STEPS = 15_000


GAUSSIAN_AMPLITUDE = 0.15
GAUSSIAN_SIGMA     = 0.02



DB_PATH = "optuna_study.db"


def _make_env_fn(seed: int = 0, history_length: int = 0):
    
    def _init():
        if history_length > 0:
            env = UPMSat2EnvPPO(
                seed=seed,
                max_steps=ENV_MAX_STEPS,
                gaussian_amplitude=GAUSSIAN_AMPLITUDE,
                gaussian_sigma=GAUSSIAN_SIGMA,
            )
            return HistoryWrapper(env, history_length=history_length)
        return UPMSat2Env(
            seed=seed,
            max_steps=ENV_MAX_STEPS,
            gaussian_amplitude=GAUSSIAN_AMPLITUDE,
            gaussian_sigma=GAUSSIAN_SIGMA,
        )
    return _init


def _sample_ppo_params(trial: optuna.Trial,
                       recurrent: bool = False) -> tuple[dict, dict]:
    learning_rate = trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True)
    gae_lambda    = trial.suggest_float("gae_lambda", 0.95, 0.999)
    ent_coef      = trial.suggest_float("ent_coef", 5e-3, 5e-2, log=True)
    vf_coef       = trial.suggest_float("vf_coef", 0.8, 2.0)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 1.0)
    log_std_init  = trial.suggest_float("log_std_init", -0.7, 0.0)  

    gamma      = 0.999   
    n_steps    = 512     
    n_epochs   = 5
    batch_size = 512     
    net_arch   = [256, 256]
    activation = nn.ReLU

    policy_kwargs = dict(
        net_arch=net_arch,
        activation_fn=activation,
        log_std_init=log_std_init,
    )

    if recurrent:
        lstm_hidden   = trial.suggest_categorical("lstm_hidden_size", [64, 128, 256])
        n_lstm_layers = 1
        policy_kwargs["lstm_hidden_size"] = lstm_hidden
        policy_kwargs["n_lstm_layers"]    = n_lstm_layers

    model_params = dict(
        learning_rate = learning_rate,
        n_steps       = n_steps,
        batch_size    = batch_size,
        n_epochs      = n_epochs,
        gamma         = gamma,
        gae_lambda    = gae_lambda,
        clip_range    = 0.2,
        ent_coef      = ent_coef,
        vf_coef       = vf_coef,
        max_grad_norm = max_grad_norm,
        use_sde       = False,
        policy_kwargs = policy_kwargs,
        verbose       = 0,
        seed          = trial.number,
    )
    return model_params, {}


def _sample_sac_params(trial: optuna.Trial) -> dict:
    learning_rate  = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    gamma          = trial.suggest_float("gamma", 0.99, 0.9999, log=True)
    tau            = trial.suggest_float("tau", 0.001, 0.02, log=True)
    batch_size     = trial.suggest_categorical("batch_size", [128, 256, 512])
    train_freq     = trial.suggest_categorical("train_freq", [1, 2, 4, 8])
    gradient_steps = trial.suggest_categorical("gradient_steps", [-1, 1, 2, 4])
    net_arch_key   = trial.suggest_categorical(
        "net_arch", ["[64,64]", "[128,64]", "[128,128]", "[256,128]", "[256,256]"])
    activation     = trial.suggest_categorical("activation", ["tanh", "relu", "elu"])

    net_arch_map = {
        "[64,64]":   [64, 64],
        "[128,64]":  [128, 64],
        "[128,128]": [128, 128],
        "[256,128]": [256, 128],
        "[256,256]": [256, 256],
    }
    activation_map = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}

    net_arch = net_arch_map[net_arch_key]
    policy_kwargs = dict(
        net_arch     = dict(pi=net_arch, qf=net_arch),
        activation_fn= activation_map[activation],
    )

    return dict(
        learning_rate   = learning_rate,
        gamma           = gamma,
        tau             = tau,
        batch_size      = batch_size,
        train_freq      = train_freq,
        gradient_steps  = gradient_steps,
        buffer_size     = 300_000,
        learning_starts = 10_000,
        ent_coef        = "auto",
        policy_kwargs   = policy_kwargs,
        verbose         = 0,
        seed            = trial.number,
    )


class TrialEvalCallback(BaseCallback):
  
    def __init__(self, eval_env, trial, n_eval_episodes=5, eval_freq=10_000,
                 deterministic=True, verbose=0):
        super().__init__(verbose)
        self.eval_env         = eval_env
        self.trial            = trial
        self.n_eval_episodes  = n_eval_episodes
        self.eval_freq        = eval_freq
        self.deterministic    = deterministic
        self.eval_idx         = 0
        self.is_pruned        = False
        self.last_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        if hasattr(self.training_env, "obs_rms"):
            self.eval_env.obs_rms = self.training_env.obs_rms

        mean_reward, _ = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=self.deterministic,
        )
        self.last_mean_reward = mean_reward
        self.eval_idx += 1

        self.trial.report(mean_reward, self.eval_idx)

        if self.verbose > 0:
            print(f"  [Trial {self.trial.number}] "
                  f"eval {self.eval_idx}: mean_reward={mean_reward:.2f}")

        if self.trial.should_prune():
            self.is_pruned = True
            return False

        return True


def _objective(trial: optuna.Trial, algo: str, total_timesteps: int,
               history_length: int = 0) -> float:
    
    seed         = 777
    is_recurrent = algo == "recurrent_ppo"
    is_sac       = algo == "sac"
    n_envs_train = 1 if is_sac else 2

    if history_length > 0 and algo == "recurrent_ppo":
        raise ValueError(
            f"--history-length no es compatible con --algo recurrent_ppo "
            f"(el LSTM ya gestiona la memoria temporal; combinar ambos es redundante).")

    print(f"[Trial {trial.number}] Muestreando hiperparámetros...")
    if is_sac:
        hp = _sample_sac_params(trial)
        env_params = {}
    else:
        hp, env_params = _sample_ppo_params(trial, recurrent=is_recurrent)

    print(f"[Trial {trial.number}] Inicializando entornos "
          f"(seed={seed}, algo={algo}, history_length={history_length}, "
          f"ENV_MAX_STEPS={ENV_MAX_STEPS}, gamma={hp['gamma']:.5f})...")

    try:
        train_env = make_vec_env(
            _make_env_fn(seed=seed, history_length=history_length),
            n_envs=n_envs_train, seed=seed)
        eval_env  = DummyVecEnv([
            _make_env_fn(seed=seed + 9999, history_length=history_length)])
    except Exception as e:
        print(f"[Trial {trial.number}] ERROR al crear entornos: {e}")
        raise optuna.exceptions.TrialPruned()

    gamma = hp["gamma"]

    # normalización consistente con main.py
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True,
                             clip_obs=10.0, clip_reward=15.0, gamma=gamma)
    eval_env  = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                             clip_obs=10.0, training=False, gamma=gamma)
    eval_env.obs_rms = train_env.obs_rms

    if algo == "ppo":
        AlgoClass, policy_name = PPO, "MlpPolicy"
    elif algo == "recurrent_ppo":
        AlgoClass, policy_name = RecurrentPPO, "MlpLstmPolicy"
    else:
        AlgoClass, policy_name = SAC, "MlpPolicy"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Trial {trial.number}] Creando modelo {algo.upper()} (device={device})...")
    try:
        model = AlgoClass(policy=policy_name, env=train_env, device=device, **hp)
    except (torch.AcceleratorError, RuntimeError) as e:
        if "out of memory" in str(e) and device == "cuda":
            print(f"[Trial {trial.number}] CUDA OOM — reintentando en CPU...")
            torch.cuda.empty_cache()
            device = "cpu"
            model = AlgoClass(policy=policy_name, env=train_env, device=device, **hp)
        else:
            raise

    eval_callback = TrialEvalCallback(
        eval_env,
        trial,
        n_eval_episodes = N_EVAL_EPISODES,
        eval_freq       = max(EVAL_FREQ // n_envs_train, 1000),
        deterministic   = True,
        verbose         = 0,
    )

    print(f"[Trial {trial.number}] Entrenando ({total_timesteps:,} pasos)...")
    try:
        model.learn(total_timesteps=total_timesteps, callback=eval_callback,
                    reset_num_timesteps=True)
    except Exception as e:
        print(f"[Trial {trial.number}] ERROR durante learn(): {e}")
        import traceback
        traceback.print_exc()
        train_env.close()
        eval_env.close()
        raise optuna.exceptions.TrialPruned()

    train_env.close()
    eval_env.close()

    if eval_callback.is_pruned:
        print(f"[Trial {trial.number}] Podado por MedianPruner")
        raise optuna.exceptions.TrialPruned()

    print(f"[Trial {trial.number}] Recompensa final: "
          f"{eval_callback.last_mean_reward:.4f}")
    return eval_callback.last_mean_reward


def run_study(algo: str, n_trials: int, timesteps: int, seed: int = 1234,
              history_length: int = 0):
    if algo not in ["ppo", "recurrent_ppo", "sac"]:
        raise ValueError(f"Algoritmo no soportado: {algo}")

    if algo == "recurrent_ppo" and not _HAS_RECURRENT:
        raise ImportError("RecurrentPPO no disponible: pip install sb3-contrib")

    if history_length > 0 and algo == "recurrent_ppo":
        raise ValueError(
            f"--history-length no es compatible con --algo recurrent_ppo "
            f"(el LSTM ya gestiona la memoria temporal; combinar ambos es redundante).")

    
    if history_length > 0:
        study_name = f"optuna_{algo}_hist{history_length}"
    else:
        study_name = f"optuna_{algo}"
    storage    = f"sqlite:///{DB_PATH}"

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS, seed=seed)
    pruner  = MedianPruner(n_startup_trials=N_STARTUP_TRIALS,
                           n_warmup_steps=N_WARMUP_STEPS)

    study = optuna.create_study(
        study_name     = study_name,
        storage        = storage,
        sampler        = sampler,
        pruner         = pruner,
        direction      = "maximize",
        load_if_exists = True,
    )

    print(f"\n{'='*60}")
    print(f"  Estudio Optuna    : {study_name}")
    print(f"  Trials nuevos     : {n_trials}  (+ {len(study.trials)} previos)")
    print(f"  Timesteps/trial   : {timesteps:,}")
    print(f"  history_length    : {history_length}")
    print(f"  ENV_MAX_STEPS     : {ENV_MAX_STEPS}")
    print(f"  lambda_m/smooth   : defaults de environment_ppo.py (0.3 / 0.15)")
    print(f"  gaussian_amplitude: {GAUSSIAN_AMPLITUDE}")
    print(f"  gaussian_sigma    : {GAUSSIAN_SIGMA}")
    print(f"  Storage           : {DB_PATH}")
    print(f"{'='*60}\n")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study.optimize(
            lambda trial: _objective(trial, algo, timesteps, history_length),
            n_trials          = n_trials,
            show_progress_bar = True,
        )

    _print_and_save_results(study, algo)


def _print_and_save_results(study: optuna.Study, algo: str):
    print(f"\n{'='*60}")
    print(f"  RESULTADO FINAL — {algo.upper()}")
    print(f"{'='*60}")

    pruned   = [t for t in study.trials
                if t.state == optuna.trial.TrialState.PRUNED]
    complete = [t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"  Trials completados : {len(complete)}")
    print(f"  Trials podados     : {len(pruned)}")

    best = study.best_trial
    print(f"\n  Mejor trial  : #{best.number}")
    print(f"  Mejor reward : {best.value:.4f}")
    print(f"\n  Hiperparámetros óptimos:")
    for k, v in best.params.items():
        print(f"    {k:40s} = {v}")

    out_path = Path(f"best_hyperparams_{algo}.json")
    with open(out_path, "w") as f:
        json.dump(best.params, f, indent=2)
    print(f"\n  Guardado en: {out_path}")


def show_results(study_name: str):
    storage = f"sqlite:///{DB_PATH}"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        _print_and_save_results(study, study_name.replace("optuna_", ""))
    except Exception as e:
        print(f"Error cargando estudio '{study_name}': {e}")


def _parse_args():
    p = argparse.ArgumentParser(
        description="Búsqueda de hiperparámetros con Optuna — UPMSat-2 RecurrentPPO"
    )
    p.add_argument("--algo",         type=str, default="recurrent_ppo",
                   choices=["ppo", "recurrent_ppo", "sac"],
                   help="Algoritmo a optimizar")
    p.add_argument("--n-trials",     type=int, default=20,
                   help="Número de trials (default: 20)")
    p.add_argument("--timesteps",    type=int, default=300_000,
                   help="Pasos por trial (default: 300000)")
    p.add_argument("--seed",         type=int, default=1234,
                   help="Seed del optimizador TPE (default: 1234)")
    p.add_argument("--history-length", type=int, default=0,
                   help="Pasos previos apilados (0=off, entorno 12D). "
                        "Si >0: HistoryWrapper con entorno 9D. Compatible con ppo y sac, "
                        "no con recurrent_ppo (LSTM ya gestiona la memoria).")
    p.add_argument("--show-results", action="store_true",
                   help="Mostrar resultados del estudio guardado")
    p.add_argument("--study-name",   type=str, default=None,
                   help="Nombre del estudio a mostrar")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.show_results:
        if args.study_name:
            name = args.study_name
        elif args.history_length > 0:
            name = f"optuna_{args.algo}_hist{args.history_length}"
        else:
            name = f"optuna_{args.algo}"
        show_results(name)
    else:
        run_study(
            algo      = args.algo,
            n_trials  = args.n_trials,
            timesteps = args.timesteps,
            seed      = args.seed,
            history_length = args.history_length,
        )