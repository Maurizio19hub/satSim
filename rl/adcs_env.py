"""
Ambiente Gymnasium per il controllo d'assetto del CubeSat 3U con ruote di
reazione, da addestrare con PPO (Stable-Baselines3, vedi rl/train.py).

L'ambiente avvolge SatelliteEngine (pacchetto `satsim/`) e sostituisce il
controllore PD. Non importa nulla della GUI né di Stable-Baselines3: è solo
fisica + interfaccia Gymnasium. La logica RL (spazi, osservazione, reward)
è documentata in rl/README.md.

Stato attuale: v4.
- Azione = variazione della coppia motore di ogni ruota (coppia rate-limited).
- Osservazione = assetto, velocità angolare e coppia motore corrente.
- Reward = penalità lineare sull'errore d'assetto e sulle accelerazioni
  oltre soglia + bonus per ogni passo vicino al target.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from satsim import SatelliteEngine, SimParams, Telemetry
from satsim.quaternion import quat_from_axis_angle

# Scala di normalizzazione di ω nell'osservazione: 0.1 rad/s ≈ 5.7 °/s → ~1.
OMEGA_SCALE = 0.1           # [rad/s]

# Massima variazione di coppia per passo, come frazione di T_max.
# Con 0.2 e Δt = 0.05 s la coppia va da 0 a T_max in 5 passi (0.25 s).
DTAU_MAX_FRAC = 0.2

# Pesi e soglie della reward (vedi rl/README.md §5).
REWARD_CONFIG = dict(
    k_err=0.01,             # penalità per grado di errore d'assetto, per passo
    alpha_max_deg=2.0,      # soglia di accelerazione angolare per asse [°/s²]
    k_accel=0.01,           # penalità per ogni °/s² oltre la soglia, per asse
    bonus=0.02,             # bonus per ogni passo con errore d'assetto sotto soglia
    bonus_theta_deg=0.1,    # soglia d'errore per il bonus [°]
)


class SatAttitudeEnv(gym.Env):
    """Ambiente di controllo d'assetto: azione = variazione delle coppie motore.

    Osservazione (7 + N): [ q_err (4) | ω / OMEGA_SCALE (3) | τ / T_max (N) ]
    Azione (N):           Δτ normalizzata in [-1, 1],
                          τ ← clip(τ + action · DTAU_MAX_FRAC · T_max, ±T_max)
    """

    metadata = {"render_modes": []}

    def __init__(self, params: SimParams | None = None, max_episode_steps: int = 2000,
                 render_mode: str | None = None, reward_config: dict | None = None,
                 dtau_max_frac: float = DTAU_MAX_FRAC):
        """Inizializzazione: engine fisico, spazi di azione e osservazione, parametri."""
        super().__init__()
        self.engine = SatelliteEngine(params)
        self.dt = self.engine.dt
        self.n_wheels = self.engine.rw.n
        self.tau_max = self.engine.rw.tau_max
        self.dtau_max = dtau_max_frac * self.tau_max
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.reward_config = {**REWARD_CONFIG, **(reward_config or {})}

        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.n_wheels,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7 + self.n_wheels,),
                                            dtype=np.float32)

        self._steps = 0
        self._tau = np.zeros(self.n_wheels)     # coppia motore corrente [N·m]
        self._prev_omega = np.zeros(3)          # ω al passo precedente (per l'accelerazione)
        self._alpha = np.zeros(3)               # accelerazione angolare dell'ultimo passo [rad/s²]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Nuovo episodio: assetto casuale a 40–80° dal target, ω casuale,
        ruote ferme e coppia nulla.

        Ritorna: (observation, info)
        """
        super().reset(seed=seed)
        rng = self.np_random
        axis = rng.normal(size=3)
        q0 = quat_from_axis_angle(axis, np.radians(rng.uniform(40.0, 80.0)))
        omega0 = rng.uniform(-0.05, 0.05, 3)
        tel = self.engine.reset(q0=q0, omega0=omega0)

        self._steps = 0
        self._tau = np.zeros(self.n_wheels)
        self._prev_omega = tel.omega
        self._alpha = np.zeros(3)
        return self._get_obs(tel), self._info(tel)

    def step(self, action: np.ndarray):
        """Un passo di controllo (Δt = params.dt).

        Ritorna: (observation, reward, terminated, truncated, info)
        """
        dtau = np.clip(action, -1.0, 1.0) * self.dtau_max
        self._tau = np.clip(self._tau + dtau, -self.tau_max, self.tau_max)
        tel = self.engine.step(self._tau)
        self._steps += 1

        # Accelerazione angolare dalla variazione di velocità tra due passi
        self._alpha = (tel.omega - self._prev_omega) / self.dt
        reward = self._compute_reward(tel, action)
        self._prev_omega = tel.omega

        terminated = False
        truncated = self._steps >= self.max_episode_steps
        return self._get_obs(tel), reward, terminated, truncated, self._info(tel)

    def _compute_reward(self, tel: Telemetry, action: np.ndarray) -> float:
        """Reward del passo corrente.

            r = − k_err · θ                                     [θ in gradi]
                − k_accel · Σ_assi max(0, |α_i| − α_max)        [α in °/s²]
                + bonus · [θ < θ_bonus]

        Il primo termine penalizza linearmente l'errore d'assetto: più il
        satellite resta lontano dal target, e più a lungo, più perde. Il
        secondo penalizza solo le accelerazioni oltre soglia. Il terzo premia
        ogni passo trascorso vicino al target, quindi il restarci.
        """
        c = self.reward_config
        alpha_deg = np.degrees(np.abs(self._alpha))
        accel_excess = np.maximum(0.0, alpha_deg - c["alpha_max_deg"]).sum()
        on_target = tel.att_err_deg < c["bonus_theta_deg"]
        return float(-c["k_err"] * tel.att_err_deg - c["k_accel"] * accel_excess
                     + c["bonus"] * on_target)

    def _get_obs(self, tel: Telemetry) -> np.ndarray:
        """Osservazione: quaternione d'errore, ω normalizzata, coppia corrente normalizzata."""
        return np.concatenate((tel.q_err, tel.omega / OMEGA_SCALE,
                               self._tau / self.tau_max)).astype(np.float32)

    def _info(self, tel: Telemetry) -> dict:
        return dict(att_err_deg=tel.att_err_deg, alpha_deg=np.degrees(self._alpha),
                    tau=self._tau.copy(), power=tel.power_total,
                    wheel_saturation=tel.wheel_saturation.max())
