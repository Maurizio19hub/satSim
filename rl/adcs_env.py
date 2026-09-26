"""
Ambiente Gymnasium per l'addestramento di un agente PPO (Stable-Baselines3)
sul controllo d'assetto del CubeSat 3U con ruote di reazione.

L'ambiente avvolge SatelliteEngine (pacchetto `satsim/`) e sostituisce il
controllore PD: ad ogni passo l'agente sceglie direttamente le coppie motore
delle N ruote. La logica RL (spazi, osservazione, reward, terminazione) è
documentata in rl/README.md.

Stato attuale: SCHELETRO. I metodi dell'ambiente sono vuoti e vanno
implementati; la configurazione di PPO e la funzione di training sono già
pronte per l'uso con Stable-Baselines3.

Uso previsto:

    python -m rl.adcs_env            # addestramento PPO con i parametri di default
"""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

from satsim import SatelliteEngine, SimParams, Telemetry

# =============================================================================
# Configurazione
# =============================================================================
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
    verbose=1,
)

TRAIN_CONFIG = dict(
    total_timesteps=1_000_000,
    n_envs=4,               # ambienti paralleli (DummyVecEnv)
    seed=0,
    log_dir=Path("runs/ppo_adcs"),
    model_path=Path("models/ppo_adcs"),
)


# =============================================================================
# Ambiente
# =============================================================================
class SatAttitudeEnv(gym.Env):
    """Ambiente di controllo d'assetto: azione = coppie motore delle ruote."""

    metadata = {"render_modes": []}

    def __init__(self, params: SimParams | None = None, max_episode_steps: int = 2000,
                 render_mode: str | None = None):
        """Inizializzazione dell'ambiente.

        Da implementare:
        - creare SatelliteEngine(params) e salvare dt, numero di ruote, T_max;
        - definire self.action_space (Box normalizzato in [-1, 1]^N);
        - definire self.observation_space (Box dell'osservazione);
        - inizializzare contatori di episodio e pesi della reward.
        """
        super().__init__()
        # TODO: implementare
        raise NotImplementedError

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Inizio di un nuovo episodio.

        Da implementare:
        - chiamare super().reset(seed=seed) per inizializzare self.np_random;
        - campionare le condizioni iniziali (assetto, ω, velocità delle ruote);
        - chiamare engine.reset(...) e azzerare i contatori.

        Ritorna: (observation, info)
        """
        super().reset(seed=seed)
        # TODO: implementare
        raise NotImplementedError

    def step(self, action: np.ndarray):
        """Un passo di controllo (Δt = params.dt).

        Da implementare:
        - denormalizzare l'azione: tau = clip(action, -1, 1) · T_max;
        - avanzare la fisica: tel = engine.step(tau);
        - calcolare osservazione, reward, terminated, truncated, info.

        Ritorna: (observation, reward, terminated, truncated, info)
        """
        # TODO: implementare
        raise NotImplementedError

    def compute_reward(self, tel: Telemetry, action: np.ndarray) -> float:
        """Reward del passo corrente, calcolata dalla telemetria.

        Da implementare (termini candidati, vedi rl/README.md):
        - penalità sull'errore d'assetto (tel.att_err_deg / tel.q_err);
        - penalità sulla velocità angolare (tel.omega);
        - penalità sullo sforzo di controllo / energia (action, tel.power_total);
        - penalità sulla saturazione delle ruote (tel.wheel_saturation).
        """
        # TODO: implementare
        raise NotImplementedError

    def _get_obs(self, tel: Telemetry) -> np.ndarray:
        """Costruisce il vettore d'osservazione dalla telemetria (da implementare)."""
        # TODO: implementare
        raise NotImplementedError


# =============================================================================
# Training con Stable-Baselines3
# =============================================================================
def train(ppo_config: dict = PPO_CONFIG, train_config: dict = TRAIN_CONFIG) -> PPO:
    """Addestra PPO sull'ambiente e salva il modello."""
    check_env(SatAttitudeEnv(), warn=True)      # verifica la conformità all'API Gymnasium

    vec_env = make_vec_env(SatAttitudeEnv, n_envs=train_config["n_envs"],
                           seed=train_config["seed"])
    model = PPO(env=vec_env, tensorboard_log=str(train_config["log_dir"]),
                seed=train_config["seed"], **ppo_config)
    model.learn(total_timesteps=train_config["total_timesteps"])

    train_config["model_path"].parent.mkdir(parents=True, exist_ok=True)
    model.save(train_config["model_path"])
    vec_env.close()
    return model


if __name__ == "__main__":
    train()
