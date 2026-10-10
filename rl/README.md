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

### 2026-10-09 — Body acceleration limit in the engine
- **Research: is there a safety limit on angular acceleration?** No universal value exists. Real limits depend on the mission:
  - structural: negligible for a rigid 3U (at 10 °/s² the tip of the satellite has ~0.03 m/s²); relevant only with deployed flexible appendages (slew profiles that do not excite flexible modes, e.g. JWST);
  - sensors: star trackers limit the angular **rate** (0.3–0.6 °/s for the ST200), not the acceleration;
  - actuators: wheel torque and momentum, already modelled;
  - jitter: set by the payload requirements.
  - Operational reference: MinXSS-1 (3U, BCT XACT) used 1 °/s² and 6 °/s by default, with hardware capable of ~25 °/s². The limit is therefore a design requirement.
- **Measurement on the v11 model (`local_training_v4_seed1@3_best`, test seeds):** the |α| peak (up to 38.9 °/s², mean 18.0) is a single kick in the first 0.2–0.4 s of every episode, when the agent drives the torque to the maximum; above ~10.6 °/s² it can only be the z axis (J_zz 5 times smaller). |α| p99: PPO 3.8, PD 3.1 °/s². The reward penalty (`k_accel = 0.01` above 2 °/s²) is too weak to prevent it (0.37 for the peak step).
- **Energy breakdown (mean per episode):** static 60 J (0.15 W × 4 wheels × 100 s) for everyone; control-dependent part PPO 5.8 J (k1·|T| 1.3 + k2·|T·Ω| 4.5) vs PD 1.3 J. The reducible part is at most ~4.5 J out of 66.
- **Change (constraint, not reward):** new parameter `ReactionWheelParams.max_body_accel_deg = 10.0` and safety filter `SatelliteEngine.accel_limit`, applied by the engine to any controller (PD and agent) after the rate limit.
  - At the start of the step α is affine in the wheel torques; the excess on a violating axis is removed with the body-torque correction J·(α − clip(α)) allocated with A⁺. Axes within the limit and the null-space part are not touched (scaling the whole command would also have slowed down x/y).
  - Alternating projections with the motor limits (rate limit, ±T_max, wheel speed saturation), which keep priority.
  - `SatAttitudeEnv` now reads back the torque actually commanded after the engine limits (`self._tau = engine.tau_cmd`), so the observation shows the filtered torque.
- **Results without retraining (test seeds 100–109):**

  | | v11 without limit | v11 with limit (same weights) | PD |
  |---|---|---|---|
  | Reward | −28.2 | −29.0 | −43.2 |
  | Final error | 0.0004° | 0.0004° | 0.0031° |
  | t<1° | 10.4 s | 10.5 s | 12.2 s |
  | Max \|α\| (mean over seeds) | 18.0 | **9.8** | 6.2 |
  | Energy | 65.8 J | 65.9 J | 61.3 J |

  - As expected the limit costs almost nothing: it cuts the z-axis kicks, while the manoeuvre time is set by the x/y axes. PD unchanged (its peaks are below 10 °/s²).
  - Random actions still exceed 10 °/s² at times: the satellite tumbles at ~50 °/s and the gyroscopic term changes within the 0.05 s step (the limit is respected at the start of the step in 355 out of 362 cases), or a wheel hits saturation.
- Tests: `test_body_acceleration_limit`, `test_body_acceleration_limit_keeps_momentum` (23 tests). `test_torque_rate_limit` and `test_action_is_rate_limited_torque_change` disable the acceleration limit, because equal torques on all wheels are a pure z torque.
- Cost: ~0.74 vs 0.69 ms per step.
- **To do:** retrain with the limit (the agent can learn that kicks are clipped) and compare with v11; then the energy step (penalty on the dynamic power).
