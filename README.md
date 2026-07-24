# Nuria-Santamaria-Blanco---TFG
# Attitude Control via Deep Reinforcement Learning — UPMSat-2

**Undergraduate Thesis (TFG) — BSc in Aerospace Engineering, ETSIAE–UPM**
**Author:** Nuria Santamaría Blanco · **Supervisor:** Ignacio Gómez Pérez
**Grade:** 9.5/10 · Defended July 2026

---

## Overview

This project compares **Deep Reinforcement Learning (DRL)** controllers against the classical **B-dot algorithm** for magnetic detumbling of the **UPMSat-2** microsatellite. A high-fidelity 7-DOF dynamics simulator was built from scratch in Python and wrapped as a custom [Gymnasium](https://gymnasium.farama.org/) environment, on top of which PPO, RecurrentPPO (RPPO), and SAC agents were trained and rigorously benchmarked against B-dot.

**Headline result:** the best DRL policy achieves **74.1% faster detumbling** than the classical B-dot controller, with statistically validated robustness across randomized inertia and orbital conditions.

## Motivation

Magnetic detumbling — reducing a satellite's angular velocity after deployment using magnetorquers — is traditionally handled by B-dot, a simple, robust, but conservative control law. This thesis asks whether a learned policy, trained under domain randomization, can outperform B-dot on detumbling speed without sacrificing robustness or requiring anything beyond the sensors B-dot already uses (magnetometer-only observations, formulated as a POMDP).

## Methods

**Simulator**
- 7-DOF rigid-body dynamics: quaternion kinematics + Euler rotational dynamics
- Orbital propagation with J2 perturbation
- Tilted-dipole (degree-1 IGRF) geomagnetic field model
- Four perturbation torques modeled
- RK4 numerical integration at 1 Hz
- Validated against a Simulink reference model to 10⁻⁹ relative error

**RL environment**
- Custom Gymnasium environment, 12-dimensional observation space
- Teacher–student reward shaping using privileged true angular velocity (ω_real) during training
- Domain randomization over inertia tensor and orbital elements for sim-to-real robustness
- POMDP formulation: magnetometer-only observations (no direct angular velocity measurement), matching real hardware constraints

**Training & evaluation**
- Algorithms: PPO, RecurrentPPO, SAC (Stable-Baselines3)
- Hyperparameter optimization via Optuna
- HistoryWrapper for frame stacking (RPPO)
- Statistical comparison: paired Friedman → Wilcoxon → Holm correction pipeline across 15 seeds per algorithm
- Sensitivity analysis on B-dot gain (k_p = 10⁶ and 2×10⁶)

## Key results

| Controller | Relative detumbling speed |
|---|---|
| B-dot (baseline) | — |
| Best DRL policy (PPO) | **+74.1% faster** |

RPPO and SAC were also trained and compared; SAC underperformed PPO/RPPO, which the thesis analyzes in terms of the POMDP's partial observability and SAC's off-policy assumptions. Full statistical results, ablations, and gain-sensitivity analysis are in the thesis document.

## Repository structure

```
.
├── src/
│   ├── simulator/        # 7-DOF dynamics, orbital propagation, magnetic field model
│   ├── env/               # Gymnasium environment wrapper
│   ├── training/           # PPO / RPPO / SAC training scripts, Optuna configs
│   └── analysis/          # Statistical pipeline (Friedman-Wilcoxon-Holm), plotting
├── results/                # Trained policies, evaluation logs, figures
├── thesis/                 # Full TFG document (PDF)
└── README.md
```

*(Adjust the tree above to match your actual folder layout.)*

## Getting started

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

**Train an agent**
```bash
python src/training/train.py --algo ppo --config configs/ppo_default.yaml
```

**Evaluate against B-dot baseline**
```bash
python src/analysis/compare_to_bdot.py --policy results/ppo_best.zip
```

*(Adjust commands to match your actual scripts and entry points.)*

## Tech stack

Python · Stable-Baselines3 · Gymnasium · NumPy · Optuna · MATLAB/Simulink (validation)

## Acknowledgements

Supervised by Ignacio Gómez Pérez, ETSIAE–UPM.
