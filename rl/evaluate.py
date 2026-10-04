"""
Evaluation of a trained PPO model against the baselines (PD, free
satellite, random actions) on the same comparison episodes.

The PD issues the same action as the agent (torque change with the same
rate limit), so the comparison is fair. The PPO policy is evaluated
deterministically (mean action, no exploration noise).

Usage:
    python -m rl.evaluate                          # baselines only
    python -m rl.evaluate models/ppo_adcs          # model + baselines
"""
from __future__ import annotations

import argparse

import numpy as np

from rl.adcs_env import SatAttitudeEnv
from satsim.controller import QuaternionPDController

EVAL_SEEDS = range(100, 110)    # 10 comparison episodes
SUCCESS_ERR_DEG = 0.01          # success criterion on the final error [°]


def pd_action(env: SatAttitudeEnv, pd: QuaternionPDController) -> np.ndarray:
    """Action (normalised torque change) that tracks the PD torque."""
    e = env.engine
    tau_pd = e.allocate(pd.compute(e.q, e.omega, e.h_rw, e.q_target))
    tau_pd = np.clip(tau_pd, -env.tau_max, env.tau_max)
    return np.clip((tau_pd - env._tau) / env.dtau_max, -1.0, 1.0)


def null_space_share(proj_range: np.ndarray, omega_w: np.ndarray) -> float:
    """Share (0–1) of the wheels' kinetic energy in the null space of the pyramid,
    i.e. in (+,−,+,−) combinations that produce no momentum on the body: 1 − |P·Ω|²/|Ω|²,
    with P = A⁺A the projector onto the range of Aᵀ."""
    n2 = float(omega_w @ omega_w)
    if n2 < 1e-12:
        return float("nan")
    return 1.0 - float((proj_range @ omega_w) @ (proj_range @ omega_w)) / n2


def run_episode(env: SatAttitudeEnv, seed: int, policy) -> dict:
    obs, _ = env.reset(seed=seed)
    pd = QuaternionPDController(env.engine.J)
    rng = np.random.default_rng(seed)
    ret, t_1deg, alpha_max, omega_peak = 0.0, np.nan, 0.0, 0.0
    for k in range(env.max_episode_steps):
        if policy == "PD":
            a = pd_action(env, pd)
        elif policy == "free":
            a = np.zeros(env.n_wheels)
        elif policy == "random":
            a = rng.uniform(-1.0, 1.0, env.n_wheels)
        else:
            a, _ = policy.predict(obs, deterministic=True)
        obs, r, terminated, truncated, info = env.step(a)
        ret += r
        alpha_max = max(alpha_max, float(np.abs(info["alpha_deg"]).max()))
        omega_peak = max(omega_peak, float(np.abs(env.last_tel.wheel_speed).max()))
        if np.isnan(t_1deg) and info["att_err_deg"] < 1.0:
            t_1deg = (k + 1) * env.dt
        if terminated or truncated:
            break
    return dict(ret=ret, err=info["att_err_deg"], t_1deg=t_1deg,
                alpha_max=alpha_max, energy=float(env.engine.x[-1]),
                rpm_peak=omega_peak * 30.0 / np.pi,
                null_share=null_space_share(env.engine.rw.A_pinv @ env.engine.rw.A,
                                            env.last_tel.wheel_speed))


def evaluate(policy, seeds=EVAL_SEEDS) -> dict:
    kwargs = {}
    if not isinstance(policy, str):          # SB3 model: environment with its observation
        from rl.models import env_kwargs_for
        kwargs = env_kwargs_for(policy)
    env = SatAttitudeEnv(**kwargs)
    rows = [run_episode(env, s, policy) for s in seeds]
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def _nanmean(x: np.ndarray) -> float:
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


def print_table(results: dict):
    print(f"{'':>10} {'reward':>9} {'err fin [°]':>12} {'err max [°]':>12} "
          f"{'t<1° [s]':>9} {'|α|max':>7} {'energy [J]':>11} {'Ω peak [RPM]':>14} {'null space':>13}")
    for name, v in results.items():
        print(f"{name:>10} {v['ret'].mean():>9.1f} {v['err'].mean():>12.4f} "
              f"{v['err'].max():>12.4f} {_nanmean(v['t_1deg']):>9.1f} "
              f"{v['alpha_max'].mean():>7.1f} {v['energy'].mean():>11.1f} "
              f"{v['rpm_peak'].mean():>8.0f}/{v['rpm_peak'].max():<5.0f} "
              f"{_nanmean(v['null_share']):>12.0%}")


def print_wheel_details(results: dict):
    """Per seed: peak |Ω| [RPM] and null-space share at the end of the episode."""
    print("\nWheels per seed (peak RPM / null-space share at the end of the episode):")
    for name in ("PPO", "PD"):
        if name in results:
            v = results[name]
            print(f"{name:>6} " + "  ".join(f"{p:.0f}/{n:.0%}" for p, n in zip(v["rpm_peak"], v["null_share"])))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("model", nargs="?", help="model to evaluate (e.g. models/ppo_adcs)")
    a = ap.parse_args()

    results = {}
    if a.model:
        from rl.models import load_model
        results["PPO"] = evaluate(load_model(a.model))
    for name in ("PD", "free", "random"):
        results[name] = evaluate(name)
    print_table(results)
    print_wheel_details(results)

    if a.model:
        ppo, pd = results["PPO"], results["PD"]
        checks = {
            "reward ≥ PD": ppo["ret"].mean() >= pd["ret"].mean(),
            f"final error ≤ {SUCCESS_ERR_DEG}° on all seeds": bool((ppo["err"] <= SUCCESS_ERR_DEG).all()),
        }
        for k, ok in checks.items():
            print(f"[{'OK' if ok else '--'}] {k}")


if __name__ == "__main__":
    main()
