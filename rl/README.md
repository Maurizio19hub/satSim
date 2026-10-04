# satSim — Reinforcement Learning (PPO)

This document covers **only the Reinforcement Learning logic** of the project: problem formulation, spaces, reward, algorithm and training choices. The physics of the simulator is described in the [technical documentation](../TECHNICAL_DOCUMENTATION.md).

**Status:** v11 of the environment.
- **Action:** change of the wheel torques.
- **Observation:** attitude error on a logarithmic scale + angular velocity + current torque + wheel speeds.
- **Reward:** − attitude error (linear + logarithmic) − accelerations above threshold − wheel speed in the null space + bonuses below 0.1° and 0.01°.
- **Reference model:** v11, `rl/pretrained/local_training_v4_seed1@3_best.zip` (4 M steps): reward −28.2 vs PD −43.2, final error 0.0004°, no wheel drift. Default of `python main.py --compare`.
- The details of every version are in the log (§9).

---

## Contents

1. [Goal](#1-goal)
2. [Folder structure](#2-folder-structure)
3. [MDP formulation](#3-mdp-formulation)
4. [Environment methods](#4-environment-methods)
5. [Reward](#5-reward)
6. [Algorithm: PPO](#6-algorithm-ppo)
7. [Training](#7-training)
8. [Open decisions](#8-open-decisions)
9. [RL development log](#9-rl-development-log)

---

## 1. Goal

Train an agent that brings the CubeSat from its initial attitude to the target attitude `q_target` and keeps it there. The agent directly commands the motor torques of the reaction wheels and replaces the quaternion PD controller (`satsim/controller.py`).

The PD remains the comparison **baseline**: the agent must match or beat it in precision, settling time, energy consumption and wheel saturation margin.

## 2. Folder structure

```
rl/
├── __init__.py
├── adcs_env.py     # Gymnasium environment SatAttitudeEnv (physics only, no SB3 and no GUI)
├── train.py        # PPO training with Stable-Baselines3 (headless)
├── evaluate.py     # comparison of a model with the PD and baselines on the test seeds
├── compare.py      # step-by-step PPO vs PD comparison for the GUI (python main.py --compare)
├── models.py       # portable model loading (load_model) and observation format (env_kwargs_for)
├── pretrained/
│   ├── ppo_adcs_v8.zip   # best v8 model (1.7 M steps), 10-value observation
│   ├── local_training_v4_seed1_best.zip     # v11 model (seed 1, 2 M steps)
│   └── local_training_v4_seed1@3_best.zip   # v11 model (4 M steps), 14-value observation: current reference
└── README.md       # this document
```

Additional dependencies (in `requirements.txt`): `gymnasium`, `stable-baselines3`, `tensorboard`.

Dependencies go one way only: `rl → satsim`. The engine imports nothing from `rl/`.

**Training without GUI.** `rl/` never imports `gui/`, PySide6, pyqtgraph or OpenGL: the graphical simulation starts only with `python main.py`. During training only the physics engine runs. The test `tests/test_rl_env.py::test_training_is_headless` checks that importing `rl.train` does not load any graphical module.

## 3. MDP formulation

| Element | Choice | Status |
|---|---|---|
| **Control step** | One engine step, Δt = `params.dt` (0.05 s). Action held constant over the step (zero-order hold). | decided |
| **Action** | `Box([-1, 1]^N)`: **change** of the motor torque of each wheel. `τ ← clip(τ + action · Δτ_max, ±T_max)` with `Δτ_max = max_torque_rate · Δt` (physical parameter in `satsim/config.py`, = 0.2 · T_max). The physical limits (torque and speed saturation) are still enforced by the engine. | v3 |
| **Observation** | `Box(6 + 2N)`: `[e_log (3) , ω / OMEGA_SCALE (3) , τ / T_max (N) , Ω / Ω_max (N)]`, i.e. attitude error on a logarithmic scale, angular velocity, current motor torque and wheel speeds. `OMEGA_SCALE = 0.1 rad/s`. With `wheel_speed_obs=False` one gets the `Box(6 + N)` format of the models up to v8. | v9 |
| **Initial state** | Random attitude 40–80° from the target (random axis), `ω` uniform in ±0.05 rad/s, wheels at rest, zero torque. Sampled with `self.np_random`. | v3 |
| **Termination** | None: `terminated` is always `False`. | v1 |
| **Truncation** | `max_episode_steps` (default 2000 steps = 100 s). | decided |

### Observation

- **Position = `e_log`** (v5), attitude error on a logarithmic scale:

  $$
  \mathbf e_{log} = \hat{\mathbf n}\;\frac{\ln(1+\theta/\theta_0)}{\ln(1+\pi/\theta_0)},\qquad \theta_0 = 0.1°
  $$

  Here $\hat{\mathbf n}$ is the axis and $\theta = 2\,\mathrm{atan2}(|\mathbf q_{vec}|, q_0)$ the angle of the error quaternion `q_err` (with $q_0 \ge 0$, shortest rotation). It is 0 on the target and its magnitude lies in [0, 1].

  | θ | 0.01° | 0.1° | 0.45° | 1° | 10° | 60° | 180° |
  |---|---|---|---|---|---|---|---|
  | \|e_log\| | 0.013 | 0.092 | 0.23 | 0.32 | 0.62 | 0.85 | 1 |
  | \|q_vec\| (v1–v4) | 0.0001 | 0.0009 | 0.0039 | 0.0087 | 0.087 | 0.5 | 1 |

  Reason: with `q_err` an error of 0.45° entered as ~0.003 and the network could not tell it from zero (see the v4 log). With the logarithmic scale the same error is worth 0.23, without saturating at large rotations. Limitation: near 180° the axis changes abruptly (outside the distribution of initial states, ≤ 80°).
- **Velocity = `ω`** in body axes, divided by `OMEGA_SCALE` to bring it to values of order 1 (the PPO neural network works better with normalised inputs).
- **Current torque = `τ / T_max`**, in [−1, 1]. It is needed because the action is a change: without knowing the current torque the agent would not know which torque it is applying (non-Markovian state).
- **Wheel speeds = `Ω / Ω_max`** (v9), in [−1, 1], where ±1 means saturation.

### Action: torque change (rate limit)

The action is the **change** of the motor torque of each wheel in one step, not the torque itself. The torque is therefore an internal state of the environment (`self._tau`), reset to zero at every `reset`:

$$
\tau_t = \mathrm{clip}\left(\tau_{t-1} + a_t\,\Delta\tau_{max},\ -T_{max},\ T_{max}\right), \qquad \Delta\tau_{max} = \texttt{max\_torque\_rate}\cdot \Delta t
$$

- With `max_torque_rate = 8 mN·m/s` and Δt = 0.05 s, Δτ_max = 0.4 mN·m = 0.2 · T_max: the torque goes from 0 to T_max in 5 steps (0.25 s) and from −T_max to +T_max in 10 steps (0.5 s). Since 2026-10-01 the same limit is enforced by the engine on any controller, PD included.
- Reason: with the absolute torque the agent could switch from +T_max to −T_max in a single step (*chattering*), with vibrations, current peaks and motor wear. Now the change is limited by construction.
- The command `τ` is the one requested from the engine. The engine may apply less if the wheel is close to speed saturation.
- `action = 0` means "keep the current torque", not "zero torque".

### Unobserved wheels (up to v8)

- **Wheel speeds not observed** (deliberate initial choice). Consequence: the state is not fully observable. The wheel momentum `h_rw` enters the dynamics (gyroscopic coupling) and determines saturation. With wheels starting at rest and 100 s episodes the effect was expected to be small (with the PD the wheels stay below ~900 RPM out of 6000). It becomes relevant with strong disturbances or long episodes: in that case `Ω/Ω_max` is added to the observation, which was done in v9 (see the log).

## 4. Environment methods

`SatAttitudeEnv` in `adcs_env.py` follows the Gymnasium API, required by Stable-Baselines3.

| Method | Task |
|---|---|
| `__init__(params, max_episode_steps, render_mode, reward_config, wheel_speed_obs)` | Creates `SatelliteEngine`, defines `action_space` and `observation_space`, reads the reward weights (`REWARD_CONFIG`, overridable) and the torque rate limit. |
| `reset(seed, options)` | Samples the initial conditions, resets the engine, zeroes the torque, stores the initial `ω` as "previous step". Returns `(obs, info)`. |
| `step(action)` | Updates the torque `τ ← clip(τ + action · Δτ_max)`, calls `engine.step(τ)`, computes the acceleration `α = (ω − ω_prev)/Δt` and the reward, updates the previous values. Returns `(obs, reward, terminated, truncated, info)`. |
| `_compute_reward(tel, action)` | Reward of the step (§5). |
| `_get_obs(tel)` | Observation vector (§3). |
| `_info(tel)` | Per-step diagnostics: `att_err_deg`, `alpha_deg`, `tau`, `power`, `wheel_saturation`. |

Note: the method is called `_compute_reward` and not `compute_reward` because Stable-Baselines3 reserves that name for *goal-conditioned* environments (`GoalEnv`) and `check_env` would fail.

## 5. Reward

The reward started from two terms (v2–v3):

$$
r_t = -\,k_{err}\,\theta_t \;-\; k_{acc}\sum_{i\in\{x,y,z\}} \max\!\left(0,\ |\alpha_{i,t}| - \alpha_{max}\right)
$$

and was then extended with the bonuses (v4, v7), the logarithmic error penalty (v8) and the null-space penalty (v11). The full current form is in the docstring of `SatAttitudeEnv._compute_reward` and in the log (§9).

| Symbol | Meaning | Value (`REWARD_CONFIG`) |
|---|---|---|
| $\theta_t$ | attitude error `tel.att_err_deg` [°] | — |
| $\alpha_t$ | angular acceleration $(\omega_t-\omega_{t-1})/\Delta t$ [°/s²] | — |
| $k_{err}$ | penalty per degree of error, per step | `k_err = 0.01` |
| $\alpha_{max}$ | acceleration threshold per axis | `alpha_max_deg = 2.0` °/s² (confirmed) |
| $k_{acc}$ | penalty per °/s² above threshold | `k_accel = 0.01` |
| $\Omega_j/\Omega_{max}$ | speed of wheel j, fraction of 6000 RPM | — |
| $k_{null}$ | weight of the null-space drift penalty (v11) | `k_null = 0.1` |

Since v11 the reward has a further term, $-\,k_{null}\,\dfrac{|(I - A^+A)\,\boldsymbol\Omega|}{\sqrt N\,\Omega_{max}}$.
- $(I - A^+A)$ projects the wheel speeds onto the null space of the pyramid: the $(-,+,-,+)/2$ combination, which produces no momentum on the body.
- Divided by $\sqrt N$ the term equals the drift speed of each wheel as a fraction of $\Omega_{max}$ (all components of the null vector have magnitude 1/2): 0.5 = 3000 RPM of drift on every wheel, penalty 0.05 per step.
- The speed useful for the manoeuvre (range space) is not penalised. The PD, which allocates with the pseudo-inverse, has zero drift by construction.

**Error term.** Linear penalty, always active: at 60° it is −0.6 per step, at 1° it is −0.01. Without bonuses the reward is always ≤ 0 and its maximum (0) is reached only on the target. Over an episode the penalty is proportional to the area under the θ(t) curve. It therefore rewards both **arriving early** and **staying** on the target: this is why it replaced the differential term of v1 (§ limits of v1).

**Acceleration term.** Zero below threshold, linear above. The acceleration is obtained from the difference of `ω` between two steps, i.e. it is the mean acceleration over the step. It also includes the effect of the disturbances, but these are ~1e-6 N·m and negligible compared with the threshold.

**Choice of the threshold.** References measured on the model:

| Quantity | Value |
|---|---|
| Maximum acceleration the wheels can produce, x/y axes | ≈ 10.6 °/s² |
| Maximum acceleration the wheels can produce, z axis | ≈ 53 °/s² (J_zz is 5 times smaller) |
| PD, 99th percentile of \|α\| (20 episodes) | 1.4 °/s² |
| PD, maximum \|α\| | 8.4 °/s² |

Real-world references:

| Source | Value |
|---|---|
| MinXSS-1 (3U CubeSat, BCT XACT ADCS): default operational peak | 1 °/s² acceleration, 6 °/s rate |
| MinXSS-1: hardware capability | ~25 °/s² |
| ST200 star tracker: maximum tolerated rate | 0.3 °/s (tip/tilt), 0.6 °/s (roll) |
| XACT: attitude tracking error unchanged up to | ~1.1 °/s |

There is no universal "safety" threshold on angular acceleration. For a rigid 3U without deployed panels, structural loads are negligible: at 10 °/s² the linear acceleration at the tip of the satellite is ~0.03 m/s². The real limits are operational:
- the maximum angular **rate** that star trackers and gyroscopes tolerate;
- the angular momentum available in the wheels;
- power;
- jitter, and flexible modes if there are appendages.

The simulator's PD was not designed with an acceleration limit: its 8.4 °/s² peak depends on the chosen gains, not on a requirement. Options for $\alpha_{max}$:
- **1 °/s²**: real operational default (MinXSS);
- **2 °/s²**: current value, just above the typical behaviour of the PD (p99 = 1.4);
- **PD peak (~8.4 °/s²)**: "never harsher than the PD", but close to the physical limit of the wheels, so the penalty would almost never be active.

Sources: [MinXSS-1 On-Orbit Pointing and Power Performance (arXiv:1706.06967)](https://arxiv.org/abs/1706.06967); [Nanobob, ST200 (arXiv:1711.01886)](https://arxiv.org/pdf/1711.01886).

**v3 check (episode with seed 1, 2000 steps, action = torque change).**

| Policy | Return | Final error | Max \|α\| |
|---|---|---|---|
| Zero actions (torque always 0, free satellite) | −2539 | 121° | 0.03 °/s² |
| Random actions | −2496 | 95° | 46 °/s² |
| PD, passed through the same rate limit | −76 | 0.003° | 8.1 °/s² |

In v2 (absolute torque) the PD scored −75: the rate limit does not noticeably degrade the control.

### History: v1 (differential reward)

v1 used $r_t = k_{prog}(\theta_{t-1}-\theta_t)$ instead of the error term. It was abandoned for two reasons:
- **Telescoping sum.** Over an episode it equals $k_{prog}(\theta_0-\theta_T)$: only where the satellite ends up counts, not how fast it gets there.
- **No incentive to stay on the target.** Once the target is reached the signal is ~0, so oscillating around it costs nothing.

v1 check (seed 1): zero actions −43.4, PD +76.5.

### Known limitations

1. **Scale of the two terms.** With a random policy the acceleration penalty is of the same order as the error term (~0.1–0.3 per step). If the agent gets stuck in the "small actions" local minimum, `k_accel` is reduced.
2. **Partial observation** (unobserved wheels, up to v8), see §3.

## 6. Algorithm: PPO

**PPO** (*Proximal Policy Optimization*) from Stable-Baselines3 is used, with the `MlpPolicy` policy. PPO is on-policy, supports continuous actions (`Box`) and works with vectorised environments.

Hyperparameters (`PPO_CONFIG` in `train.py`, SB3 defaults, to be tuned):

| Parameter | Value |
|---|---|
| `learning_rate` | 3e-4 → 0 linear (`LR_SCHEDULE` in `rl/models.py`, since v6) |
| `n_steps` | 2048 |
| `batch_size` | 64 |
| `n_epochs` | 10 |
| `gamma` | 0.99 |
| `gae_lambda` | 0.95 |
| `clip_range` | 0.2 |
| `ent_coef` | 0.0 |
| `vf_coef` | 0.5 |
| `max_grad_norm` | 0.5 |
| `device` | `cpu` (with a small network the CPU is faster than the GPU) |

## 7. Training

```bash
pip install -r requirements.txt
python -m rl.train                             # 2 M steps, 4 environments in sequence
python -m rl.train --subproc                   # 4 environments in 4 processes (faster)
python -m rl.train --timesteps 24576           # short run: measures the speed of your PC
python -m rl.train --subproc --resume models/ppo_adcs --timesteps 1000000 --seed 1 --out models/ppo_adcs_2M
                                               # continue an existing training run for another 1 M steps
tensorboard --logdir runs/ppo_adcs             # learning curves
```

`train()` runs, in order:
1. SB3's `check_env` to verify compliance with the Gymnasium API;
2. `make_vec_env` with `n_envs = 4` environments (`DummyVecEnv`, or `SubprocVecEnv` with `--subproc`);
3. `PPO.learn` for `total_timesteps` (default 2 000 000);
4. prints the duration and saves the model to `models/ppo_adcs.zip` (or to the `--out` path).

### Validation and best model (since v6)

- Every `--eval-every` steps (default 100 k) the deterministic policy is evaluated on the **validation** seeds 200–209.
- If the mean reward is the best seen so far, the model is saved to `<out>_best.zip`. At the end of training there are therefore two models: the final one (`<out>.zip`) and the best one (`<out>_best.zip`).
- The **test** seeds 100–109 (`rl/evaluate.py`) are never used to choose the model: they are used only for the final comparison with the PD.
- Cost: ~25 s per evaluation, ~8 min over 2 M steps. The values also go to TensorBoard (`eval/mean_reward`, `eval/final_err_deg`).

### Continuing a training run (`--resume`)

- `--resume <model>` loads the network weights and the optimiser state of a saved model and continues for another `--timesteps` steps. The hyperparameters are those saved in the model.
- The step counter carries on (e.g. from 1 007 616) and TensorBoard continues the same curve.
- The learning rate schedule continues too: resuming a 2 M model for another 2 M steps restarts the learning rate from 1.5e-4 and brings it down to 0.
- It is better to use a `--seed` different from the previous training run: with the same seed the environment would repeat the same sequence of initial conditions.
- `--out` avoids overwriting the starting model.

The `runs/` and `models/` folders are excluded from git.

### Parallel environments (`n_envs = 4`)

- **What they are.** 4 independent copies of the simulator, i.e. 4 satellites with different initial conditions. **There is only one agent**, i.e. a single neural network: at every step it computes the 4 actions at once, one per satellite, and each copy advances by Δt.
- **Rollout.** Each PPO update uses `n_steps × n_envs = 2048 × 4 = 8192` transitions. When an episode ends in one copy, that copy resets itself and the others carry on.
- **Step count.** `total_timesteps` counts the sum of the steps of all copies: 1 M total steps = 250 k per copy.
- **Execution.** With `DummyVecEnv` (default of `make_vec_env`) the 4 copies run **in sequence in the same process**, so there is no real CPU parallelism. The advantage is statistical: the data of each update come from 4 different episodes and are less correlated. To use more cores, use `--subproc` (`SubprocVecEnv`).

Throughput measured in the cloud: ~460 steps/s with 4 environments, i.e. ~1 M steps in ~35 min.

**Short run (40 000 steps).** The pipeline works end to end. `ep_rew_mean` goes from −378 to −366: too few steps (~20 episodes per environment) to learn the manoeuvre. A long training run is needed.

### Training times

Measurements in the cloud container (Intel Xeon 2.8 GHz, 4 vCPU, PyTorch on CPU):

| Configuration | Steps/s | 1 M steps |
|---|---|---|
| Physics only, 1 environment, no PPO | 809 | 21 min |
| PPO, 4 environments in sequence (`DummyVecEnv`) | 411 | ~41 min |
| PPO, 4 environments in 4 processes (`--subproc`) | 591 | ~28 min |

About half of the time is physics: ~1.2 ms per step, with 5 evaluations of the derivatives (4 RK4 stages + 1 for the telemetry). The other half is PPO: network inference and 10 update epochs every 8192 steps. To estimate the time on another PC, run `python -m rl.train --timesteps 24576` and multiply the printed duration by ~40.

## 8. Open decisions

- Tuning of `k_err`, `k_accel` and `k_null`.
- Early termination conditions.
- Distribution of the initial conditions and possible curriculum.
- Environmental disturbances active or not during training.
- Normalisation of observations and reward with `VecNormalize`.
- Acceleration and wheel saturation margin, still worse than the PD (see the latest log entries).

## 9. RL development log

### 2026-09-26 — Skeleton
- Created the `rl/` folder with `adcs_env.py`.
- `SatAttitudeEnv(gym.Env)` with empty methods: `__init__`, `reset`, `step`, `_compute_reward`, `_get_obs`.
- PPO configuration (`PPO_CONFIG`) and training (`TRAIN_CONFIG`, `train()`) with Stable-Baselines3.
- Added the dependencies `gymnasium`, `stable-baselines3`, `tensorboard`.

### 2026-09-28 — Environment v1
- Implemented `__init__`, `reset`, `step`, `_get_obs`, `_compute_reward`.
- Observation: `q_err` + normalised `ω` (7 values).
- Reward: progress towards the target (difference of the error in degrees) − penalty on angular accelerations above 2 °/s².
- `compute_reward` renamed `_compute_reward` (conflict with `GoalEnv` in `check_env`).
- `check_env` passed. 40k-step PPO trial run done.

### 2026-09-28 — Reward v2
- Differential term replaced by a linear error penalty: `−k_err · θ` with `k_err = 0.01`.
- Term on accelerations above threshold unchanged.
- Documented: action as absolute torque (and risk of chattering), real-world references for the acceleration threshold, how the parallel environments work.

### 2026-09-28 — v3: torque change and separate training
- Action = torque change per wheel, limited to `0.2 · T_max` per step. The current torque `τ / T_max` is added to the observation (7 + N values).
- Acceleration threshold confirmed at 2 °/s².
- Training moved to `rl/train.py` (headless, CLI with `--timesteps`, `--n-envs`, `--subproc`, `--seed`). `rl/adcs_env.py` contains only the environment.
- New tests `tests/test_rl_env.py`: training without GUI modules, `check_env`, torque rate limit, zero reward on the target.
- Training times measured (§7).

### 2026-09-28 — First training run (1 M steps) and resuming training
- First training run in the cloud: 1 M steps, `--subproc`, 19 min. `ep_rew_mean` from −2720 to −72.
  - Flat up to ~200 k steps, rises quickly between 250 k and 480 k, then refines slowly and is not yet flat at 1 M.
- Deterministic comparison over 10 episodes (seeds 100–109):

  | | PPO (1 M) | PD |
  |---|---|---|
  | Reward per episode | −61.7 | −53.5 |
  | Time to get below 1° | 15.1 s | 12.2 s |
  | Final error | 0.64° (identical in all episodes) | 0.003° |
  | Max \|α\| | 10.1 °/s² | 6.2 °/s² |
  | Energy | 69.7 J | 61.3 J |

- Added the `--resume` and `--out` options to `rl/train.py` to continue an existing training run.
- Decision: before introducing a bonus near the target, keep training to find the real plateau.

### 2026-09-28 — Resume: from 1 M to 2 M steps
- Resumed from 1 M with `--resume`, seed 1, 18 min.
- `ep_rew_mean` from −72 to −65: still rising, but more slowly. The policy standard deviation dropped from 0.26 to 0.15.
- Deterministic comparison over 10 episodes (seeds 100–109):

  | | PPO 1 M | PPO 2 M | PD |
  |---|---|---|---|
  | Reward per episode | −61.7 | −56.2 | −53.5 |
  | Final error | 0.64° | 0.40° | 0.003° |
  | Time to get below 1° | 15.1 s | 15.0 s | 12.2 s |
  | Max \|α\| | 10.1 °/s² | 8.7 °/s² | 6.2 °/s² |
  | Energy | 69.7 J | 68.9 J | 61.3 J |

- The final error remains a constant offset, identical in all episodes: the policy converges to a fixed point ~0.4° from the target.

### 2026-09-28 — v4: bonus near the target (1 M steps from scratch)
- Reward: `+0.02` for every step with θ < 0.1°. New `rl/evaluate.py` for the comparison on seeds 100–109.
- v4 baseline:

  | | Reward | Final error |
  |---|---|---|
  | PD | −19.5 | 0.0031° |
  | Free satellite | −2300 | 131° |
  | Random actions | −2759 | 130° |

- 18 min training run. The `ep_rew_mean` curve is practically identical to the one without bonus: from −2720 to −64.5.
- PPO evaluation: reward −56.9, final error 0.448° (identical on all seeds), 16.2 s to get below 1°, max \|α\| 10.8 °/s², 65.0 J. Success criteria not met.
- Diagnosis: the bonus is almost never collected, so it does not drive learning.
  - After 30 s the stochastic policy is below 0.1° in only 0.06 % of the steps.
  - At the fixed point the observation is q_vec ≈ (−0.0027, 0.0013, 0.0025) and the deterministic action is exactly 0: the network does not react to a 0.45° error.
  - This confirms the "input signal too small" limitation (§3). The next step is to rescale the attitude error in the observation.
- Note: at the fixed point a residual torque (+,−,+,−)·0.0067·T_max remains. It lies in the null space of the pyramid, so it does not act on the body, but it uses energy.

### 2026-09-28 — v5: attitude error on a logarithmic scale
- `q_err` (4 values) replaced by `e_log` (3 values, §3): the observation goes from 11 to 10 values. Reward unchanged from v4, so the v4 baseline is still valid.
- v5 training (1 M steps from scratch, 18 min).
  - `ep_rew_mean` from −2590 to −73.5. It starts more slowly than v4 (flat up to ~350 k steps) and at 1 M steps is still rising steeply: −108 → −73.5 over the last 80 k.
  - The policy standard deviation is still 0.49 (v4: 0.26).
- Evaluation (seeds 100–109):

  | | PPO v5 | PD |
  |---|---|---|
  | Reward | −29.8 | −19.5 |
  | Final error | **0.0042°** (v4: 0.448°) | 0.0031° |
  | Time to get below 1° | **12.0 s** | 12.2 s |
  | Time to get below 0.1° | **13.7 s** | 15.1 s |
  | Max \|α\| | 10.4 °/s² | 6.2 °/s² |
  | Energy | 71.6 J | 61.3 J |

  - `[--]` reward ≥ PD; `[OK]` final error ≤ 0.01° on all seeds.
- Reward breakdown (mean over seeds):

  | | Error term | Acceleration term | Bonus |
  |---|---|---|---|
  | PPO | −62.1 | −2.1 | +34.5 |
  | PD | −52.6 | −0.9 | +34.0 |

  Almost all of the gap comes from the error term, i.e. from the area under θ(t) during the large manoeuvre. On seeds 101, 103 and 107 the agent gets below 0.1° only after 17–21 s.
- Conclusion: the logarithmic scale removed the offset. The large manoeuvre on some seeds still needs improving.

### 2026-09-28 — v5: resume from 1 M to 2 M steps
- `ep_rew_mean`: rises from −72.7 to −41.7 (1.42 M), then falls back to −66.4 (1.61 M) and climbs again to −47.3 (2.02 M). Unstable trend. The policy standard deviation drops from 0.50 to 0.26.
- Deterministic evaluation (seeds 100–109): **worse** than the 1 M model.

  | | v5 1 M | v5 2 M | PD |
  |---|---|---|---|
  | Reward | −29.8 | −34.4 | −19.5 |
  | Final error | 0.0042° | 0.018° | 0.0031° |
  | Time to get below 1° | 12.0 s | 12.8 s | 12.2 s |

- Reward per seed, PPO 2 M vs PD:

  | Seed | 100 | 101 | 102 | 103 | 104 | 105 | 106 | 107 | 108 | 109 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | PPO 2 M | −2.9 | −58.5 | −26.1 | −82.7 | −4.6 | −30.6 | 4.8 | −110.5 | −11.3 | −21.4 |
  | PD | −3.0 | −25.2 | −16.5 | −45.8 | −7.3 | −24.6 | −5.5 | −37.2 | −14.9 | −14.8 |

  PPO beats the PD on 4 seeds (100, 104, 106, 108) and clearly loses on 101, 103 and 107.
- Hard seeds:
  - 103 and 107 have the largest initial angle (74°, 69°) and an ω0 that moves away from the target (+2.4 °/s along the error axis);
  - 101 has θ0 = 64°.
  - They are hard for the PD too, but there the PPO gap is largest: the tail of the initial distribution (θ0 close to 80°) is learned worse.
- Problems found:
  1. Without checkpoints the best model of the training run (around 1.42 M) was lost.
  2. `ep_rew_mean` (stochastic policy) and the deterministic evaluation do not move in the same direction.
  3. With a constant learning rate, the fine-tuning phase is unstable.

### 2026-09-28 — v6: decreasing learning rate and best model
- Learning rate linear from 3e-4 to 0. Default `total_timesteps` raised to 2 M.
- Periodic evaluation on the validation seeds 200–209, saving the best model (`--eval-every`).
- No change to environment, observation or reward (v4 baseline valid).
- Option postponed: sample the hard initial conditions (θ0 close to 80°) more often.
- v6 training (2 M steps from scratch, 44 min including evaluations).
  - `ep_rew_mean` from −2590 to −30, steady growth without drops: −222 at 811 k, −82 at 1 M, −45 at 1.4 M, −30 at 2 M.
  - Validation (seeds 200–209): reward from −2506 to **−8.6**, improving at almost every evaluation. The best model practically coincides with the final one (2 M).
  - Final error in validation: minimum 0.0024° at 1.1 M, then back up to 0.029°. The reward keeps improving because the agent becomes faster.
- Test (seeds 100–109), best model:

  | | PPO v6 | PD |
  |---|---|---|
  | Reward | **−17.1** | −19.5 |
  | Final error | 0.029° | 0.0031° |
  | Time to get below 1° | **9.1 s** | 12.2 s |
  | Max \|α\| | 11.5 °/s² | 6.2 °/s² |
  | Energy | 68.6 J | 61.3 J |

  - `[OK]` reward ≥ PD; `[--]` final error ≤ 0.01°.
  - Faster than the PD on 9 seeds out of 10. Better reward on 7 seeds out of 10, worse on 103 (−69.5 vs −45.8), 107 (−39.4 vs −37.2) and 108 (−25.0 vs −14.9).
- Observation: below 0.1° the reward is almost indifferent to the error. The linear term at 0.03° is −0.0003 per step and the bonus is already collected. The "≤ 0.01°" criterion is therefore not represented in the reward: the agent traded residual precision for speed.

### 2026-09-29 — v7: second precision bonus
- Reward: on top of the `+0.02` bonus for θ < 0.1° a second `+0.02` bonus is added for θ < 0.01°. Below 0.01° the reward per step is therefore 0.04. Parameters `bonus2`, `bonus2_theta_deg` in `REWARD_CONFIG`.
- Reason: in v6 the reward was almost indifferent to the error below 0.1°, so the "final error ≤ 0.01°" criterion was not rewarded.
- v7 baseline (seeds 100–109):

  | | Reward | Final error |
  |---|---|---|
  | PD | **+10.0** | 0.0031° |
  | Free satellite | −2300 | 131° |
  | Random actions | −2759 | 130° |

  The PD stays below 0.01° for most of the episode and gains +29.5 compared with v6.
- Training: 2 M steps from scratch, same settings as v6.
- v7 training (2 M steps, 46 min).
  - `ep_rew_mean` almost identical to v6: from −2590 to −31. The training reward (stochastic policy) collects little of the precision bonus.
  - Validation: reward from −2506 to **+24.7** (1.8 M steps, best model). Final error 0.003–0.009° from 1.1 M onwards.
  - At the last evaluation (2 M) validation collapses to −8.6 (error 0.0109°): the final model is worse than the best one.
- Test (seeds 100–109):

  | | PPO v7 best | PPO v7 final | PD |
  |---|---|---|---|
  | Reward | **+13.4** | −17.3 | +10.0 |
  | Mean final error | 0.0131° | 0.0107° | 0.0031° |
  | Time to get below 1° | **9.2 s** | 9.3 s | 12.2 s |
  | Max \|α\| | 10.9 °/s² | 10.7 °/s² | 6.2 °/s² |
  | Energy | 71.9 J | 69.2 J | 61.3 J |

  - Best model: `[OK]` reward ≥ PD; `[--]` final error ≤ 0.01° (8 seeds out of 10).
  - Final error per seed: 0.008° on 8 seeds, 0.0145° on seed 104, 0.0526° on seed 109.
  - Better reward than the PD on 7 seeds out of 10, worse on 103 (−45.0 vs −16.7), 107 (−11.7 vs −7.8) and 108 (6.8 vs 14.6).
- Observation: the agent stops just inside the last bonus threshold. In v6 it stopped at 0.03° with a 0.1° threshold, in v7 at 0.008° with a 0.01° threshold. Below the last threshold the reward does not reward further precision, so the residual error is set by where the threshold is.

### 2026-09-29 — v8: logarithmic error penalty
- Reward: `−0.02 · ln(1 + θ/0.01°)` is added to the linear penalty (`k_log`, `log_theta0_deg`). Bonuses and acceleration penalty are unchanged.
  - The linear part dominates at large angles and keeps the push to complete the manoeuvre quickly.
  - The logarithmic part dominates below ~1° and rewards every gain in precision even below the bonus thresholds: from 0.01° to 0.001° it is worth 0.012 per step, against 0.00009 for the linear part alone.
- Reason: in v6 and v7 the agent stopped just inside the last bonus threshold (0.03° with a 0.1° threshold, 0.008° with a 0.01° threshold).
- Total error penalty per step:

  | θ | 60° | 10° | 1° | 0.1° | 0.01° | 0.001° |
  |---|---|---|---|---|---|---|
  | Penalty | 0.77 | 0.24 | 0.10 | 0.049 | 0.014 | 0.002 |

- v8 baseline (seeds 100–109):

  | | Reward | Final error |
  |---|---|---|
  | PD | **−42.9** | 0.0031° |
  | Free satellite | −2670 | 131° |
  | Random actions | −3135 | 130° |
- v8 training (2 M steps, 46 min).
  - `ep_rew_mean` from −2970 to −84.
  - Validation: reward from −2970 to **−11.2** (1.7 M steps, best model). Final error 0.0003–0.002° from 1.2 M steps onwards.
- Test (seeds 100–109):

  | | PPO v8 best | PPO v8 final | PD |
  |---|---|---|---|
  | Reward | **−27.8** | −48.5 | −42.9 |
  | Final error | **0.0003°** (all seeds) | 0.14° (max 1.39°) | 0.0031° |
  | Time to get below 1° | **10.8 s** | 10.6 s | 12.2 s |
  | Max \|α\| | 14.1 °/s² | 14.2 °/s² | 6.2 °/s² |
  | Energy | 69.1 J | 70.6 J | 61.3 J |

  - Best model: **`[OK]` reward ≥ PD; `[OK]` final error ≤ 0.01° on all seeds.** Both success criteria are met.
  - Better reward than the PD on 8 seeds out of 10. On seed 101 it is worse (−56.4 vs −49.2), on 103 practically even (−74.9 vs −74.4).
  - The final error is 10 times smaller than the PD's.
- The final model (2 M) is much worse than the best one: on one seed it stops at 1.39°. This confirms that saving the best model is essential.
- Accelerations (14.1 vs 6.2 °/s²) and energy (69.1 vs 61.3 J) remain worse than the PD.
- The best v8 model is published in `rl/pretrained/ppo_adcs_v8.zip` (156 KB), as an exception to the rule of not versioning models (`models/` stays in `.gitignore`). To check it: `python -m rl.evaluate rl/pretrained/ppo_adcs_v8`. It only works with the v8 observation format (10 values), which is still supported via `wheel_speed_obs=False`.

### 2026-10-01 — Torque limit in the engine and comparison mode
- The torque rate limit moves from the environment to the engine (`max_torque_rate` in `satsim/config.py`): same value (0.2 · T_max per step), now also applied to the GUI PD. `DTAU_MAX_FRAC` removed. Results of the v8 model unchanged (`rl.evaluate`: −27.8 vs −42.9).
- `SatAttitudeEnv` exposes `last_tel` (telemetry of the last step) and `T_external` (external test torque, 0 during training).
- `rl/compare.py` + `python main.py --compare`: visual PPO vs PD comparison. It matches `rl.evaluate` (test).
- **Finding from the visual comparison: the agent's wheels spin in the null space.**
  - On seeds 100–109, at the end of the episode 91–100 % of PPO's wheel speed lies in the null space of the pyramid: (+,−,+,−) combinations that produce no torque on the body. For the PD the share is 0–9 %, because pseudo-inverse allocation is minimum-norm.
  - Peak wheel speed: PPO 1941–6000 RPM, PD 658–2377 RPM. On seeds 101 (at 7.1 s) and 107 (at 11.3 s) PPO drives a wheel to **saturation**: precisely two of the seeds where it does worse than the PD. A saturated wheel loses control authority and produces acceleration peaks.
  - Likely cause: the agent does not observe the wheel speeds and the reward does not penalise them, so the null-space component, which has no effect on the attitude, drifts freely. It is also the main cause of the higher energy consumption compared with the PD.

### 2026-10-02 — Models portable across Python versions
- **Bug:** loading `ppo_adcs_v8` with Python 3.14, on the local PC, caused a *segmentation fault* both in `pytest` and in `main.py --compare`.
  - The learning rate was a Python function (closure `linear_schedule`). Stable-Baselines3 saved it in the `.zip` as Python 3.11 bytecode, the cloud version.
  - Loaded with Python 3.14, that bytecode crashed the interpreter. The crash stack pointed to `rl/train.py, in schedule`.
- **Fix:**
  - `rl/models.py`: `LR_SCHEDULE = LinearSchedule(3e-4 → 0)` from Stable-Baselines3. It is a class, serialised by reference, hence portable. Same values as the previous function.
  - `load_model()`: loads a model replacing the saved learning rate with `LR_SCHEDULE`, without ever deserialising it. Used by `evaluate.py`, `compare.py` and `train.py --resume`.
  - `rl/pretrained/ppo_adcs_v8.zip` re-saved in portable format: same weights, identical results (−27.8 vs −42.9).
- Test `test_saved_models_are_portable`: no serialised functions in the published models or in a freshly saved model.
- Rule: always load models with `rl.models.load_model`, never with `PPO.load` directly.

### 2026-10-02 — v9: wheel speeds in the observation
- **Reason:** the v8 agent accumulates 91–100 % of the wheel speed in the null space, up to saturation on seeds 101 and 107, without being able to see it (log of 2026-10-01).
- **Change:** `_get_obs` adds `tel.wheel_speed / Ω_max`, i.e. the speeds of the 4 wheels taken from the telemetry of the last step. The physical value is in rad/s; divided by Ω_max = 628 rad/s (6000 RPM) it lies between −1 and +1, where ±1 means saturation.
  - The observation grows from 10 to 14 values.
  - Without normalisation an input up to ±628, against the others around ±1, would dominate the network.
- **Reward unchanged:** one change at a time. PD baseline unchanged (−42.9).
- **Compatibility with models up to v8:**
  - option `SatAttitudeEnv(wheel_speed_obs=False)`;
  - `rl.models.env_kwargs_for(model)` picks the format from the number of inputs of the model;
  - used by `rl.evaluate`, by the comparison (`--compare`) and by `rl.train --resume`. The v8 model keeps working everywhere, with unchanged results.
- Tests: `test_wheel_speed_in_observation`, `test_old_models_still_load` (18 tests in total).
- **Expectation:** observing the wheels gives the agent the information, but not an incentive to keep them slow, because the reward does not penalise them. The next planned step is a penalty on the wheel speeds.
- **Results:** training to be done (2 M steps, locally).

### 2026-10-03 — v10: wheel speed penalty
- **v9 result (14-value observation, reward unchanged):** final error 0.0008° and reward −34.5, worse than v8 (−27.8, error 0.0003°) and worse than the PD only on energy/accelerations. Observing the wheels without an incentive to keep them slow is not enough, as expected.
- **Measurement (seeds 100–109, peak |Ω|/Ω_max):** PD 0.11–0.40 (time average 0.04); v8 0.32–1.00 (time average 0.52, saturation on seeds 101 and 107).
- **Change:** reward −`k_wheel · Σ_j max(0, |Ω_j|/Ω_max − 0.3)`, with `k_wheel = 0.05`.
  - Threshold 0.3 (1800 RPM): the PD exceeds it on only 2 seeds out of 10, and only slightly, so the "normal" manoeuvre is not penalised and the agent's precision is not directly affected.
  - Linear hinge, as for the acceleration: the penalty acts only on null-space drift and on approaching saturation. At Ω = Ω_max on one wheel it costs 0.035 per step, comparable with the error penalty at ~1° (0.10).
- **PD baseline with the new reward:** −43.0 (was −42.9). The v8 model, which does not observe the wheels, drops to −127.4: this confirms that the penalty acts on its behaviour.
- Test: `test_wheel_speed_penalty` (19 tests, all passing). No change to the observation: the 14-value format is the v9 one.
- **To do:** 2 M-step training from scratch (`python -m rl.train --subproc`), then `rl.evaluate` on the `_best` model. If the final error gets worse, reduce `k_wheel` (0.02) or raise the threshold; if the wheels still exceed ~0.5, raise `k_wheel`.

### 2026-10-03 — `rl.evaluate`: wheel peak and null space
- **v10 result (reward `k_wheel = 0.05`, threshold 0.3; local training `local_training_v3`):** reward −67.9 vs PD −43.0, final error 0.0011°, t<1° 14.4 s (PD 12.2), max |α| 15.5, energy 67.8 J. The manoeuvre is slower and the reward is worse than v8/v9: to be understood whether the penalty limits the manoeuvre or it is training variability.
- **Change (metric only, no effect on training):** `rl.evaluate` prints two more columns, the peak |Ω| in RPM (mean/max over seeds) and the share of the wheels' kinetic energy in the null space at the end of the episode, `1 − |A⁺AΩ|²/|Ω|²`. Below the table there is the per-seed detail for PPO and PD.
- Check on v8: PPO peak 1941–6000 RPM and 83–100 % in the null space; PD 659–2378 RPM and 0–1 %. It matches the values in the 2026-10-01 log.
- Test: `test_null_space_share` (20 tests).

### 2026-10-03 — v11: null-space drift penalty (replaces v10)
- **Analysis of v10 with the new metrics** (`local_training_v3`): mean peak 3116 RPM (max 5333), null-space share 71 % (v8: 3879 RPM, 97 %; PD: 1250 RPM, 0 %). 9 seeds out of 10 above the 1800 RPM threshold.
  - The penalty on the total speed reduced the drift only partly and made the manoeuvre slower (14.4 s).
  - It mixes two different things: the speed useful for the manoeuvre (range space of A) and the useless drift (null space). The latter is what has to go, and it can be removed without touching the attitude: a torque along (+,−,+,−) stops the drift without acting on the body.
- **Change:** the v10 term `−k_wheel · Σ max(0, |Ω_j|/Ω_max − 0.3)` is replaced by `−k_null · |(I − A⁺A)Ω| / (√N·Ω_max)`, with `k_null = 0.1` (§5). No threshold: the PD has zero drift. `wheel_speed_thr` and `k_wheel` removed. Observation unchanged (14 values, v9).
- **Baseline with the new reward:** PD −43.2 (v10: −43.0). The v8 model drops to −131.9 (~104 points lost to the drift).
- Test: `test_null_space_penalty` replaces `test_wheel_speed_penalty` (20 tests).
- **To do:** 2 M-step training from scratch, then `rl.evaluate`. Expected: null-space share close to 0 % without worsening t<1°. If the manoeuvre slows down again, reduce `k_null`; if the drift stays above ~20 %, increase it.
- **v11 results** (local training `local_training_v4`, 2 M steps, test seeds 100–109):

  | | v8 | v10 | **v11** | PD |
  |---|---|---|---|---|
  | Reward | −27.8 (v8 reward) | −67.9 | −60.8 | −43.2 |
  | Final error | 0.0003° | 0.0011° | **0.0004°** | 0.0031° |
  | t<1° | 10.8 s | 14.4 s | 13.6 s | 12.2 s |
  | Max \|α\| | 14.1 | 15.5 | 18.4 | 6.2 |
  | Energy | 69.1 J | 67.8 J | **63.8 J** | 61.3 J |
  | Wheel peak mean / max | 3879 / 6000 RPM | 3116 / 5333 | **2059 / 3725** | 1250 / 2378 |
  | Null-space share | 97 % | 71 % | **5 %** | 0 % |

  - Null-space drift is solved (0–14 % per seed, no wheel close to saturation) and the energy gets close to the PD's. Precision unchanged.
  - `[--]` reward ≥ PD; `[OK]` final error ≤ 0.01°. Manoeuvre slower than the PD (13.6 vs 12.2 s) and higher max |α|.
  - Hypotheses on the gap, to be checked: torque contended between manoeuvre and drift correction; drift penalty accumulated during the manoeuvre; training variability (v9 was also worse than v8 with the same reward).
  - Proposed next step: per-term reward breakdown in `rl.evaluate`, then a single targeted change.
- **Second v11 training run, `--seed 1`** (`local_training_v4_seed1`, same code):

  | | v11 seed 0 | **v11 seed 1** | PD |
  |---|---|---|---|
  | Reward | −60.8 | **−32.7** | −43.2 |
  | Final error | 0.0004° | **0.0006°** | 0.0031° |
  | t<1° | 13.6 s | **10.8 s** | 12.2 s |
  | Max \|α\| | 18.4 | 16.3 | 6.2 |
  | Energy | 63.8 J | 67.2 J | 61.3 J |
  | Wheel peak mean / max | 2059 / 3725 RPM | 3161 / 5317 RPM | 1250 / 2378 RPM |
  | Null-space share | 5 % | **1 %** | 0 % |

  - **`[OK]` reward ≥ PD; `[OK]` final error ≤ 0.01° on all seeds.** First model to meet both criteria without wheel drift.
  - **Variability between training runs:** with the same code, the seed changes the reward by 28 points (−60.8 vs −32.7), more than the whole gap to the PD. Comparisons between versions based on a single training run (v9 vs v8, v10) are not conclusive.
  - The speed peaks remain high (up to 5317 RPM, test seed 103), but they are useful momentum, not drift (null space ≤ 2 % on all seeds): the manoeuvre is faster than the PD's and needs more momentum. Hence also the slightly higher energy.
  - Open: max |α| ~2.5 times the PD; reduced wheel margin on the seeds with a high peak; variability between training runs (a third seed).
  - In progress: resuming `local_training_v4` for another 2 M steps. Candidate new reference model: v11 seed 1 (to be published in `rl/pretrained/` at the user's choice).

### 2026-10-03 — Fix: validation with `--resume`
- **Bug:** in the validation callback the first check was at `eval_every` absolute steps (100 k). With `--resume` the counter starts from the steps already done (e.g. 2 M), so at start-up the validation was repeated ~20 times in a row on the same model (~8 min lost). The training results were not affected.
- **Fix:** in `_on_training_start` the first validation is set to `steps already done + eval_every`. Test `test_validation_schedule_after_resume` (21 tests).

### 2026-10-03 — v11: resuming seed 1 from 2 M to 4 M steps
- `--resume local_training_v4_seed1 --timesteps 2000000 --seed 3` (learning rate from 1.5e-4 to 0). In validation the reward improves straight away (−25 → −20 after 100 k steps).
- Test (seeds 100–109), best model (`local_training_v4_seed1@3_best`):

  | | v11 seed 1 (2 M) | **v11 seed 1 → 4 M** | PD |
  |---|---|---|---|
  | Reward | −32.7 | **−28.2** | −43.2 |
  | Final error | 0.0006° | **0.0004°** | 0.0031° |
  | t<1° | 10.8 s | **10.4 s** | 12.2 s |
  | Max \|α\| | 16.3 | 18.0 | 6.2 |
  | Energy | 67.2 J | **65.8 J** | 61.3 J |
  | Wheel peak mean / max | 3161 / 5317 RPM | 2604 / 5945 RPM | 1250 / 2378 RPM |
  | Null-space share | 1 % | **0 %** | 0 % |

  - **`[OK]` reward ≥ PD; `[OK]` final error ≤ 0.01° on all seeds.** Best model so far.
  - Max |α| (18.0 °/s²) and the maximum peak get worse: 5945 RPM on test seed 105 (99 % of Ω_max). It is not drift (null space 0 %) but momentum used for the manoeuvre; the margin against saturation is however zero.
  - Proposed as the new reference model.
  - Next points: accelerations, wheel saturation margin (limit on the total momentum), energy.

### 2026-10-03 — v11 model published
- `rl/pretrained/local_training_v4_seed1@3_best.zip` (v11, seed 1, resumed to 4 M with seed 3), trained locally with Python 3.14.
- Checked in the cloud with Python 3.11: `rl.evaluate` reproduces exactly the same values (−28.2 vs −43.2), so the model is portable. `test_saved_models_are_portable` passes.
- It becomes the default model of `python main.py --compare` and `rl/compare.py`. `test_comparison_matches_evaluate` now uses v11 (seed 105, the one with the highest speed peak). v8 remains for `test_old_models_still_load`.

### 2026-10-04 — English translation
- The whole repository (code, comments, GUI and CLI strings, documentation) translated into English. In `rl.evaluate` the baselines are now called `free` and `random` (were `libero` and `casuale`). The technical documentation is now `TECHNICAL_DOCUMENTATION.md`.
