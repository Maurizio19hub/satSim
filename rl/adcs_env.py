"""
Ambiente Gymnasium per l'addestramento di un agente PPO (Stable-Baselines3)
sul controllo d'assetto del CubeSat 3U con ruote di reazione.

L'ambiente avvolge SatelliteEngine (pacchetto `satsim/`) e sostituisce il
controllore PD: ad ogni passo l'agente sceglie direttamente le coppie motore
delle N ruote. La logica RL (spazi, osservazione, reward, terminazione) è
documentata in rl/README.md.

Stato attuale: prima versione. Osservazione = assetto e velocità angolare;
reward = avvicinamento al target e penalità sulle accelerazioni oltre soglia.

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
from satsim.quaternion import quat_from_axis_angle

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
# Scala di normalizzazione di ω nell'osservazione: 0.1 rad/s ≈ 5.7 °/s → ~1.
OMEGA_SCALE = 0.1           # [rad/s]

# Pesi e soglie della reward (vedi rl/README.md §5).
REWARD_CONFIG = dict(
    k_progress=1.0,         # reward per grado di avvicinamento al target in un passo
    alpha_max_deg=2.0,      # soglia di accelerazione angolare per asse [°/s²]
    k_accel=0.01,           # penalità per ogni °/s² oltre la soglia, per asse
)


class SatAttitudeEnv(gym.Env):
    """Ambiente di controllo d'assetto: azione = coppie motore delle ruote.

    Osservazione (7): [ q_err (4) | ω / OMEGA_SCALE (3) ]
    Azione (N):       coppie motore normalizzate in [-1, 1], tau = action · T_max
    """

    metadata = {"render_modes": []}

    def __init__(self, params: SimParams | None = None, max_episode_steps: int = 2000,
                 render_mode: str | None = None, reward_config: dict | None = None):
        """Inizializzazione: engine fisico, spazi di azione e osservazione, parametri."""
        super().__init__()
        self.engine = SatelliteEngine(params)
        self.dt = self.engine.dt
        self.n_wheels = self.engine.rw.n
        self.tau_max = self.engine.rw.tau_max
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.reward_config = {**REWARD_CONFIG, **(reward_config or {})}

        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.n_wheels,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32)

        self._steps = 0
        self._prev_err_deg = 0.0        # errore d'assetto al passo precedente
        self._prev_omega = np.zeros(3)  # ω al passo precedente (per l'accelerazione)
        self._alpha = np.zeros(3)       # accelerazione angolare dell'ultimo passo [rad/s²]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Nuovo episodio: assetto casuale a 40–80° dal target, ω casuale, ruote ferme.

        Ritorna: (observation, info)
        """
        super().reset(seed=seed)
        rng = self.np_random
        axis = rng.normal(size=3)
        q0 = quat_from_axis_angle(axis, np.radians(rng.uniform(40.0, 80.0)))
        omega0 = rng.uniform(-0.05, 0.05, 3)
        tel = self.engine.reset(q0=q0, omega0=omega0)

        self._steps = 0
        self._prev_err_deg = tel.att_err_deg
        self._prev_omega = tel.omega
        self._alpha = np.zeros(3)
        return self._get_obs(tel), self._info(tel)

    def step(self, action: np.ndarray):
        """Un passo di controllo (Δt = params.dt).

        Ritorna: (observation, reward, terminated, truncated, info)
        """
        tau = np.clip(action, -1.0, 1.0) * self.tau_max
        tel = self.engine.step(tau)
        self._steps += 1

        # Accelerazione angolare dalla variazione di velocità tra due passi
        self._alpha = (tel.omega - self._prev_omega) / self.dt
        reward = self._compute_reward(tel, action)

        self._prev_err_deg = tel.att_err_deg
        self._prev_omega = tel.omega

        terminated = False
        truncated = self._steps >= self.max_episode_steps
        return self._get_obs(tel), reward, terminated, truncated, self._info(tel)

    def _compute_reward(self, tel: Telemetry, action: np.ndarray) -> float:
        """Reward del passo corrente.

            r = k_progress · (θ_prev − θ)                       [θ in gradi]
              − k_accel · Σ_assi max(0, |α_i| − α_max)          [α in °/s²]

        Il primo termine è positivo se l'assetto si avvicina al target, negativo
        se si allontana. Il secondo penalizza solo le accelerazioni oltre soglia.
        """
        c = self.reward_config
        progress = self._prev_err_deg - tel.att_err_deg
        alpha_deg = np.degrees(np.abs(self._alpha))
        accel_excess = np.maximum(0.0, alpha_deg - c["alpha_max_deg"]).sum()
        return float(c["k_progress"] * progress - c["k_accel"] * accel_excess)

    def _get_obs(self, tel: Telemetry) -> np.ndarray:
        """Osservazione: quaternione d'errore e velocità angolare normalizzata."""
        return np.concatenate((tel.q_err, tel.omega / OMEGA_SCALE)).astype(np.float32)

    def _info(self, tel: Telemetry) -> dict:
        return dict(att_err_deg=tel.att_err_deg, alpha_deg=np.degrees(self._alpha),
                    power=tel.power_total, wheel_saturation=tel.wheel_saturation.max())


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
