"""
Addestramento PPO (Stable-Baselines3) dell'agente di controllo d'assetto.

Headless: importa solo `satsim/` (fisica) e `rl/adcs_env.py`, mai la GUI
(`gui/`, PySide6, pyqtgraph, OpenGL). La simulazione grafica si avvia solo
con `python main.py`.

Uso:
    python -m rl.train                              # parametri di default
    python -m rl.train --timesteps 200000 --n-envs 4 --subproc
    python -m rl.train --resume models/ppo_adcs --timesteps 1000000   # continua un training
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


# Iperparametri di PPO (valori di default di SB3, da tarare).
PPO_CONFIG = dict(
    policy="MlpPolicy",
    learning_rate=LR_SCHEDULE,  # 3e-4 → 0 lineare (rl/models.py, portabile tra versioni di Python)
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
    total_timesteps=2_000_000,
    n_envs=4,               # copie del simulatore (vedi rl/README.md §7)
    subproc=False,          # True: una copia per processo (SubprocVecEnv), usa più core
    seed=0,
    resume=None,            # percorso di un modello salvato da cui continuare (senza .zip)
    eval_every=100_000,     # passi tra due valutazioni di validazione (0 = disattivata)
    val_seeds=range(200, 210),  # episodi di validazione (diversi dai seed di test 100–109)
    log_dir=Path("runs/ppo_adcs"),
    model_path=Path("models/ppo_adcs"),
)


class BestModelCallback(BaseCallback):
    """Valuta periodicamente la policy deterministica sui seed di validazione
    e salva il modello quando la reward media supera il miglior valore visto."""

    def __init__(self, eval_every: int, seeds, save_path: Path):
        super().__init__()
        self.eval_every, self.seeds, self.save_path = eval_every, seeds, save_path
        self.next_eval = eval_every
        self.best = -np.inf

    def _on_training_start(self) -> None:
        # Con --resume num_timesteps parte dai passi già fatti: la prima
        # validazione va eval_every passi dopo, non a eval_every assoluti.
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
                tag = "  <- nuovo migliore, salvato"
            print(f"[validazione] {self.num_timesteps:>8} passi: reward {mean:8.1f}  "
                  f"errore finale {err:.4f}°{tag}", flush=True)
        return True


def train(ppo_config: dict = PPO_CONFIG, train_config: dict = TRAIN_CONFIG) -> PPO:
    """Addestra PPO sull'ambiente e salva il modello.

    Con train_config["resume"] carica un modello già addestrato e continua
    l'addestramento per altri total_timesteps passi: pesi della rete e stato
    dell'ottimizzatore vengono ripresi, il contatore dei passi prosegue e
    TensorBoard continua la stessa curva.

    Con train_config["eval_every"] > 0 la policy viene valutata periodicamente
    sui seed di validazione e il modello migliore è salvato in <model_path>_best.
    """
    resume = train_config["resume"]
    # Con --resume l'ambiente usa l'osservazione del modello ripreso (es. v8 senza ruote).
    env_kwargs = env_kwargs_for(load_model(resume)) if resume else {}
    check_env(SatAttitudeEnv(**env_kwargs), warn=True)      # conformità all'API Gymnasium

    vec_env = make_vec_env(SatAttitudeEnv, n_envs=train_config["n_envs"],
                           seed=train_config["seed"], env_kwargs=env_kwargs,
                           vec_env_cls=SubprocVecEnv if train_config["subproc"] else DummyVecEnv)
    if resume:
        model = load_model(resume, env=vec_env, device=ppo_config.get("device", "auto"),
                           tensorboard_log=str(train_config["log_dir"]))
        print(f"Ripreso {resume}: {model.num_timesteps} passi già eseguiti")
    else:
        model = PPO(env=vec_env, tensorboard_log=str(train_config["log_dir"]),
                    seed=train_config["seed"], **ppo_config)

    best_path = train_config["model_path"].with_name(train_config["model_path"].name + "_best")
    callback = (BestModelCallback(train_config["eval_every"], train_config["val_seeds"], best_path)
                if train_config["eval_every"] > 0 else None)

    t0 = time.perf_counter()
    model.learn(total_timesteps=train_config["total_timesteps"], callback=callback,
                reset_num_timesteps=not resume, tb_log_name="PPO")
    print(f"Addestramento completato in {(time.perf_counter() - t0) / 60:.1f} min")

    train_config["model_path"].parent.mkdir(parents=True, exist_ok=True)
    model.save(train_config["model_path"])
    print(f"Modello salvato in {train_config['model_path']}.zip ({model.num_timesteps} passi totali)")
    if callback is not None:
        print(f"Modello migliore in {best_path}.zip (reward di validazione {callback.best:.1f})")
    vec_env.close()
    return model


def parse_args() -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--timesteps", type=int, default=TRAIN_CONFIG["total_timesteps"])
    ap.add_argument("--n-envs", type=int, default=TRAIN_CONFIG["n_envs"])
    ap.add_argument("--subproc", action="store_true", help="un processo per ambiente")
    ap.add_argument("--seed", type=int, default=TRAIN_CONFIG["seed"])
    ap.add_argument("--resume", type=Path, default=None,
                    help="modello da cui continuare (es. models/ppo_adcs)")
    ap.add_argument("--eval-every", type=int, default=TRAIN_CONFIG["eval_every"],
                    help="passi tra due valutazioni di validazione (0 = disattivata)")
    ap.add_argument("--out", type=Path, default=TRAIN_CONFIG["model_path"],
                    help="dove salvare il modello (default models/ppo_adcs)")
    a = ap.parse_args()
    return {**TRAIN_CONFIG, "total_timesteps": a.timesteps, "n_envs": a.n_envs,
            "subproc": a.subproc, "seed": a.seed, "resume": a.resume, "model_path": a.out,
            "eval_every": a.eval_every}


if __name__ == "__main__":
    # La guardia __main__ è necessaria anche per SubprocVecEnv su Windows/macOS.
    train(train_config=parse_args())
