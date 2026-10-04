"""
PPO training (Stable-Baselines3) of the attitude control agent.

Headless: it only imports `satsim/` (physics) and `rl/adcs_env.py`, never the GUI
(`gui/`, PySide6, pyqtgraph, OpenGL). The graphical simulation is started only
with `python main.py`.

Usage:
    python -m rl.train                              # default parameters
    python -m rl.train --timesteps 200000 --n-envs 4 --subproc
    python -m rl.train --resume models/ppo_adcs --timesteps 1000000   # continue a training run
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from rl.adcs_env import SatAttitudeEnv
from rl.evaluate import evaluate
from rl.models import LR_SCHEDULE, env_kwargs_for, load_model


# PPO hyperparameters (SB3 defaults, to be tuned).
PPO_CONFIG = dict(
    policy="MlpPolicy",
    learning_rate=LR_SCHEDULE,  # 3e-4 → 0 linear (rl/models.py, portable across Python versions)
    n_steps=2048,           # steps collected per environment before each update
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    device="cpu",           # with a small MlpPolicy the CPU is faster than the GPU
    verbose=1,
)

TRAIN_CONFIG = dict(
    total_timesteps=2_000_000,
    n_envs=4,               # simulator copies (see rl/README.md §7)
    subproc=False,          # True: one copy per process (SubprocVecEnv), uses more cores
    seed=0,
    resume=None,            # path of a saved model to continue from (without .zip)
    eval_every=100_000,     # steps between two validation runs (0 = disabled)
    val_seeds=range(200, 210),  # validation episodes (different from the test seeds 100–109)
    log_dir=Path("runs/ppo_adcs"),
    model_path=Path("models/ppo_adcs"),
)


class BestModelCallback(BaseCallback):
    """Periodically evaluates the deterministic policy on the validation seeds
    and saves the model when the mean reward beats the best value seen so far."""

    def __init__(self, eval_every: int, seeds, save_path: Path):
        super().__init__()
        self.eval_every, self.seeds, self.save_path = eval_every, seeds, save_path
        self.next_eval = eval_every
        self.best = -np.inf

    def _on_training_start(self) -> None:
        # With --resume num_timesteps starts from the steps already done: the first
        # validation comes eval_every steps later, not at eval_every absolute.
        self.next_eval = self.num_timesteps + self.eval_every

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_eval:
            self.next_eval += self.eval_every
            res = evaluate(self.model, seeds=self.seeds)
            mean, err = float(res["ret"].mean()), float(res["err"].mean())
            self.logger.record("eval/mean_reward", mean)
            self.logger.record("eval/final_err_deg", err)
            tag = ""
            if mean > self.best:
                self.best = mean
                self.save_path.parent.mkdir(parents=True, exist_ok=True)
                self.model.save(self.save_path)
                tag = "  <- new best, saved"
            print(f"[validation] {self.num_timesteps:>8} steps: reward {mean:8.1f}  "
                  f"final error {err:.4f}°{tag}", flush=True)
        return True


def train(ppo_config: dict = PPO_CONFIG, train_config: dict = TRAIN_CONFIG) -> PPO:
    """Trains PPO on the environment and saves the model.

    With train_config["resume"] it loads an already trained model and continues
    training for another total_timesteps steps: network weights and optimiser
    state are restored, the step counter carries on and TensorBoard continues
    the same curve.

    With train_config["eval_every"] > 0 the policy is evaluated periodically
    on the validation seeds and the best model is saved to <model_path>_best.
    """
    resume = train_config["resume"]
    # With --resume the environment uses the observation of the resumed model (e.g. v8 without wheels).
    env_kwargs = env_kwargs_for(load_model(resume)) if resume else {}
    check_env(SatAttitudeEnv(**env_kwargs), warn=True)      # compliance with the Gymnasium API

    vec_env = make_vec_env(SatAttitudeEnv, n_envs=train_config["n_envs"],
                           seed=train_config["seed"], env_kwargs=env_kwargs,
                           vec_env_cls=SubprocVecEnv if train_config["subproc"] else DummyVecEnv)
    if resume:
        model = load_model(resume, env=vec_env, device=ppo_config.get("device", "auto"),
                           tensorboard_log=str(train_config["log_dir"]))
        print(f"Resumed {resume}: {model.num_timesteps} steps already done")
    else:
        model = PPO(env=vec_env, tensorboard_log=str(train_config["log_dir"]),
                    seed=train_config["seed"], **ppo_config)

    best_path = train_config["model_path"].with_name(train_config["model_path"].name + "_best")
    callback = (BestModelCallback(train_config["eval_every"], train_config["val_seeds"], best_path)
                if train_config["eval_every"] > 0 else None)

    t0 = time.perf_counter()
    model.learn(total_timesteps=train_config["total_timesteps"], callback=callback,
                reset_num_timesteps=not resume, tb_log_name="PPO")
    print(f"Training completed in {(time.perf_counter() - t0) / 60:.1f} min")

    train_config["model_path"].parent.mkdir(parents=True, exist_ok=True)
    model.save(train_config["model_path"])
    print(f"Model saved to {train_config['model_path']}.zip ({model.num_timesteps} total steps)")
    if callback is not None:
        print(f"Best model in {best_path}.zip (validation reward {callback.best:.1f})")
    vec_env.close()
    return model


def parse_args() -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--timesteps", type=int, default=TRAIN_CONFIG["total_timesteps"])
    ap.add_argument("--n-envs", type=int, default=TRAIN_CONFIG["n_envs"])
    ap.add_argument("--subproc", action="store_true", help="one process per environment")
    ap.add_argument("--seed", type=int, default=TRAIN_CONFIG["seed"])
    ap.add_argument("--resume", type=Path, default=None,
                    help="model to continue from (e.g. models/ppo_adcs)")
    ap.add_argument("--eval-every", type=int, default=TRAIN_CONFIG["eval_every"],
                    help="steps between two validation runs (0 = disabled)")
    ap.add_argument("--out", type=Path, default=TRAIN_CONFIG["model_path"],
                    help="where to save the model (default models/ppo_adcs)")
    a = ap.parse_args()
    return {**TRAIN_CONFIG, "total_timesteps": a.timesteps, "n_envs": a.n_envs,
            "subproc": a.subproc, "seed": a.seed, "resume": a.resume, "model_path": a.out,
            "eval_every": a.eval_every}


if __name__ == "__main__":
    # The __main__ guard is also required by SubprocVecEnv on Windows/macOS.
    train(train_config=parse_args())
