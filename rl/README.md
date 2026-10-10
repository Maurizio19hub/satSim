# satSim — Reinforcement Learning

This page explains how the PPO agent is set up: what it sees (observation), what it does (action), how it is rewarded, and how each choice came about. The physics of the simulator is described in the [technical documentation](../TECHNICAL_DOCUMENTATION.md).

**Current reference model:** `rl/pretrained/local_training_v4_seed1@3_best.zip` (v11, 4 M training steps). On the 10 test manoeuvres: final error 0.0004° vs 0.0031° for the PD, 10.4 s vs 12.2 s to get within 1°.

---

## Goal

Bring the CubeSat from a random attitude, 40–80° away from the target, to the target attitude and keep it there, by commanding the four reaction wheels directly. The classical PD controller is the baseline. The agent is successful if, on the 10 test manoeuvres (seeds 100–109):
- its reward is at least as good as the PD's, **and**
- the final pointing error is ≤ 0.01° on every manoeuvre.

## Action: torque *changes*, not torques

At every step (0.05 s) the agent outputs 4 numbers in [−1, 1], one per wheel. Each one is a **change** of that wheel's torque, at most ±0.4 mN·m per step. So going from zero to full torque takes 0.25 s.

Why: when the agent could set the torque directly, it learned to flip the motors from full positive to full negative from one step to the next (*chattering*), which on real hardware means vibrations, current peaks and wear. The engine also applies the motor limits and the 10 °/s² body acceleration limit, for the agent and the PD alike.

## Observation: what the agent sees (14 numbers)

| Part | Values | Why it is there |
|---|---|---|
| **Attitude error**, log scale | 3 | Axis × size of the error. A logarithmic scale lets the network "see" both 60° and 0.01°: with a linear scale an error of 0.45° looked like zero and the agent stopped there (v4 → v5). |
| **Rotation rate** | 3 | How fast the satellite is turning, normalised to ~1. |
| **Current wheel torques** | 4 | Needed because the action is a change: without it the agent would not know what torque it is applying. |
| **Wheel speeds** (fraction of 6000 RPM) | 4 | Lets the agent see how close each wheel is to saturation, and the useless wheel spin described below (v9). |

All values are scaled to roughly [−1, 1], because neural networks learn badly from inputs of very different sizes.

## Reward: what the agent is paid for

At every step the agent receives:

```
reward = − error penalty           (linear + logarithmic in the pointing error)
         − acceleration penalty    (only above 2 °/s², per axis)
         − null-space penalty      (useless wheel spin)
         + bonus if error < 0.1°   (+0.02)
         + bonus if error < 0.01°  (+0.02 more)
```

| Term | What it does | Why it was added |
|---|---|---|
| **Linear error penalty** | Large when the satellite is far from the target: pushes for a fast manoeuvre. Summed over the episode it measures "how long you stayed far away". | Replaced a reward based on progress (v1), which did not care how fast the target was reached or whether the satellite stayed there. |
| **Logarithmic error penalty** | Dominates below ~1°: every gain in precision is still rewarded, even at 0.001°. | Without it the agent stopped just inside the last bonus threshold (0.03° in v6, 0.008° in v7). |
| **Precision bonuses** | Reward every step spent within 0.1° and within 0.01°. | Make "reach the target *and stay there*" explicit (v4, v7). |
| **Acceleration penalty** | Discourages harsh accelerations above 2 °/s². | Smoothness. Since the 10 °/s² hard limit was introduced, it only shapes the behaviour below that. |
| **Null-space penalty** | Penalises wheel spin that does not move the satellite (see below). | Without it the wheels drifted up to saturation (v8–v11). |

Weights and thresholds are in `REWARD_CONFIG` in `rl/adcs_env.py`.

### The null-space problem

With 4 wheels for 3 axes there is one combination, wheels 1 and 3 one way and 2 and 4 the other, whose effects cancel out: the wheels spin, but the satellite does not move. The agent had no reason to avoid it, so small random amounts accumulated until a wheel hit 6000 RPM and the satellite lost control on that axis. The PD never does this, because it always shares the torque among the wheels in the most economical way.

- Penalising the **total** wheel speed (v10) slowed the manoeuvre and fixed the drift only partly.
- Penalising **only the useless component** (v11) solved it: 97 % of the wheel speed in the null space with v8, 0 % with v11, and no loss of speed.

## Training in short

- **Algorithm:** PPO (Stable-Baselines3), small neural network on the CPU, 4 parallel copies of the simulator.
- **Schedule:** learning rate decreasing linearly to zero, 2 M steps (~45 min).
- **Model selection:** every 100 k steps the agent is tested on 10 validation manoeuvres (seeds 200–209), never the test ones, and the best model is kept.
- **Variability:** results vary a lot between training runs: the same code gave −60.8 and −32.7 with two different seeds. Comparisons between versions should use more than one run.

Commands to train and evaluate are in the [main README](../README.md#reproduce-the-training).

## Versions at a glance

Results on the 10 test manoeuvres. The reward changed between versions, so compare each row with its own PD column, not rows with each other.

| Version | Main change | Reward (PD) | Final error | Time to < 1° | Notes |
|---|---|---|---|---|---|
| v3 | Action = torque change | −61.7 (−53.5) | 0.64° | 15.1 s | Stops at a fixed offset |
| v4 | Bonus below 0.1° | −56.9 (−19.5) | 0.45° | 16.2 s | Bonus never reached |
| v5 | Log-scale attitude error | −29.8 (−19.5) | 0.0042° | 12.0 s | Offset gone |
| v6 | Decreasing learning rate, best-model saving | **−17.1** (−19.5) | 0.029° | 9.1 s | Beats PD, not precise enough |
| v7 | Second bonus below 0.01° | **+13.4** (+10.0) | 0.013° | 9.2 s | Stops just inside the threshold |
| v8 | Logarithmic error penalty | **−27.8** (−42.9) | **0.0003°** | 10.8 s | Both criteria met; wheels drift in the null space |
| v9 | Wheel speeds observed | −34.5 (−42.9) | 0.0008° | – | Seeing the wheels is not enough |
| v10 | Penalty on total wheel speed | −67.9 (−43.0) | 0.0011° | 14.4 s | Slower, drift only reduced |
| v11 | Penalty on null-space spin only | **−28.2** (−43.2) | **0.0004°** | **10.4 s** | Both criteria met, no drift (4 M steps) |
| v11 + 10 °/s² limit | Body acceleration limit (same weights) | **−29.0** (−43.2) | **0.0004°** | 10.5 s | Max acceleration 18.0 → 9.8 °/s² |

The detailed log of every experiment follows.

---

## RL development log

### 2026-09-26 — Skeleton
- Created the `rl/` folder with `adcs_env.py`.
- `SatAttitudeEnv(gym.Env)` with empty methods: `__init__`, `reset`, `step`, `_compute_reward`, `_get_obs`.
- PPO configuration (`PPO_CONFIG`) and training (`TRAIN_CONFIG`, `train()`) with Stable-Baselines3.
- Added the dependencies `gymnasium`, `stable-baselines3`, `tensorboard`.

### 2026-09-28 — Environment v1
- Implemented `__init__`, `reset`, `step`, `_get_obs`, `_compute_reward`.

