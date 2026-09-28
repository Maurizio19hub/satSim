"""
Addestramento PPO (Stable-Baselines3) dell'agente di controllo d'assetto.

Headless: importa solo `satsim/` (fisica) e `rl/adcs_env.py`, mai la GUI
(`gui/`, PySide6, pyqtgraph, OpenGL). La simulazione grafica si avvia solo
con `python main.py`.

Uso:
    python -m rl.train                              # parametri di default
    python -m rl.train --timesteps 200000 --n-envs 4 --subproc
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from rl.adcs_env import SatAttitudeEnv

# Iperparametri di PPO (valori di default di SB3, da tarare).
PPO_CONFIG = dict(
    policy="MlpPolicy",
    learning_rate=3e-4,
    n_steps=2048,           # passi raccolti per ambiente prima di ogni aggiornamento
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    device="cpu",           # con una MlpPolicy piccola la CPU è più veloce della GPU
    verbose=1,
)

TRAIN_CONFIG = dict(
    total_timesteps=1_000_000,
    n_envs=4,               # copie del simulatore (vedi rl/README.md §7)
    subproc=False,          # True: una copia per processo (SubprocVecEnv), usa più core
    seed=0,
    log_dir=Path("runs/ppo_adcs"),
    model_path=Path("models/ppo_adcs"),
)


def train(ppo_config: dict = PPO_CONFIG, train_config: dict = TRAIN_CONFIG) -> PPO:
    """Addestra PPO sull'ambiente e salva il modello."""
    check_env(SatAttitudeEnv(), warn=True)      # verifica la conformità all'API Gymnasium

    vec_env = make_vec_env(SatAttitudeEnv, n_envs=train_config["n_envs"],
                           seed=train_config["seed"],
                           vec_env_cls=SubprocVecEnv if train_config["subproc"] else DummyVecEnv)
    model = PPO(env=vec_env, tensorboard_log=str(train_config["log_dir"]),
                seed=train_config["seed"], **ppo_config)

    t0 = time.perf_counter()
    model.learn(total_timesteps=train_config["total_timesteps"])
    print(f"Addestramento completato in {(time.perf_counter() - t0) / 60:.1f} min")

    train_config["model_path"].parent.mkdir(parents=True, exist_ok=True)
    model.save(train_config["model_path"])
    vec_env.close()
    return model


def parse_args() -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--timesteps", type=int, default=TRAIN_CONFIG["total_timesteps"])
    ap.add_argument("--n-envs", type=int, default=TRAIN_CONFIG["n_envs"])
    ap.add_argument("--subproc", action="store_true", help="un processo per ambiente")
    ap.add_argument("--seed", type=int, default=TRAIN_CONFIG["seed"])
    a = ap.parse_args()
    return {**TRAIN_CONFIG, "total_timesteps": a.timesteps, "n_envs": a.n_envs,
            "subproc": a.subproc, "seed": a.seed}


if __name__ == "__main__":
    # La guardia __main__ è necessaria anche per SubprocVecEnv su Windows/macOS.
    train(train_config=parse_args())
