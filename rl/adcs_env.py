"""
Ambiente Gymnasium per il controllo d'assetto del CubeSat 3U con ruote di
reazione, da addestrare con PPO (Stable-Baselines3, vedi rl/train.py).

L'ambiente avvolge SatelliteEngine (pacchetto `satsim/`) e sostituisce il
controllore PD. Non importa nulla della GUI né di Stable-Baselines3: è solo
fisica + interfaccia Gymnasium. La logica RL (spazi, osservazione, reward)
è documentata in rl/README.md.

Stato attuale: v9.
- Azione = variazione della coppia motore di ogni ruota (coppia rate-limited).
- Osservazione = errore d'assetto in scala logaritmica, velocità angolare,
  coppia motore corrente e velocità delle ruote (v9; disattivabile con
  wheel_speed_obs=False per i modelli fino alla v8).
- Reward = penalità lineare + logaritmica sull'errore d'assetto e sulle accelerazioni
  oltre soglia + due bonus per ogni passo vicino al target (0.1° e 0.01°).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from satsim import SatelliteEngine, SimParams, Telemetry
from satsim.quaternion import quat_from_axis_angle

# Scala di normalizzazione di ω nell'osservazione: 0.1 rad/s ≈ 5.7 °/s → ~1.
OMEGA_SCALE = 0.1           # [rad/s]

# Scala logaritmica dell'errore d'assetto nell'osservazione:
#   e = asse · log(1 + θ/θ0) / log(1 + π/θ0)
# θ0 fissa dove la scala passa da lineare a logaritmica. Con θ0 = 0.1°:
# 0.01° → 0.013, 0.1° → 0.092, 0.45° → 0.23, 10° → 0.62, 60° → 0.85, 180° → 1.
LOG_THETA0 = np.radians(0.1)    # [rad]

# Pesi e soglie della reward (vedi rl/README.md §5).
REWARD_CONFIG = dict(
    k_err=0.01,             # penalità per grado di errore d'assetto, per passo
    k_log=0.02,             # peso della penalità logaritmica sull'errore
    log_theta0_deg=0.01,    # scala della penalità logaritmica [°]
    alpha_max_deg=2.0,      # soglia di accelerazione angolare per asse [°/s²]
    k_accel=0.01,           # penalità per ogni °/s² oltre la soglia, per asse
    bonus=0.02,             # bonus per ogni passo con errore d'assetto sotto soglia
    bonus_theta_deg=0.1,    # soglia d'errore per il bonus [°]
    bonus2=0.02,            # secondo bonus (precisione), si somma al primo
    bonus2_theta_deg=0.01,  # soglia d'errore per il secondo bonus [°]
    k_null=0.1,             # penalità sulla velocità delle ruote nello spazio nullo (frazione di Ω_max)
)


def log_attitude_error(q_err: np.ndarray, theta0: float = LOG_THETA0) -> np.ndarray:
    """Errore d'assetto come vettore asse · g(θ), con g logaritmica in [0, 1].

    q_err ha q0 ≥ 0 (rotazione più breve), quindi θ = 2·atan2(|q_vec|, q0) ∈ [0, π].
    Vicino a zero g(θ) ≈ θ/(θ0·log(1 + π/θ0)): il segnale resta leggibile anche
    per errori di centesimi di grado, senza saturare alle grandi rotazioni.
    """
    q_vec = q_err[1:]
    s = np.linalg.norm(q_vec)
    if s < 1e-12:
        return np.zeros(3)
    theta = 2.0 * np.arctan2(s, q_err[0])
    g = np.log1p(theta / theta0) / np.log1p(np.pi / theta0)
    return q_vec / s * g


class SatAttitudeEnv(gym.Env):
    """Ambiente di controllo d'assetto: azione = variazione delle coppie motore.

    Osservazione (6 + 2N): [ e_log (3) | ω / OMEGA_SCALE (3) | τ / T_max (N) | Ω / Ω_max (N) ]
                  (6 + N con wheel_speed_obs=False, formato dei modelli fino alla v8)
    Azione (N):           Δτ normalizzata in [-1, 1],
                          τ ← clip(τ + action · Δτ_max, ±T_max)
                          con Δτ_max = max_torque_rate · Δt (limite del motore,
                          satsim/config.py: 0.2 · T_max per passo di 0.05 s)
    """

    metadata = {"render_modes": []}

    def __init__(self, params: SimParams | None = None, max_episode_steps: int = 2000,
                 render_mode: str | None = None, reward_config: dict | None = None,
                 wheel_speed_obs: bool = True):
        """Inizializzazione: engine fisico, spazi di azione e osservazione, parametri."""
        super().__init__()
        self.engine = SatelliteEngine(params)
        self.dt = self.engine.dt
        self.n_wheels = self.engine.rw.n
        self.tau_max = self.engine.rw.tau_max
        # Massima variazione di coppia per passo: la stessa che il motore
        # applica a qualunque controllore (engine.rw.rate_limit).
        self.dtau_max = self.engine.rw.tau_rate_max * self.dt
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.reward_config = {**REWARD_CONFIG, **(reward_config or {})}

        self.wheel_speed_obs = wheel_speed_obs
        self.omega_w_max = self.engine.rw.omega_max          # [rad/s] (6000 RPM)
        rw = self.engine.rw
        self.null_proj = np.eye(self.n_wheels) - rw.A_pinv @ rw.A   # proiettore sullo spazio nullo di A

        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.n_wheels,), dtype=np.float32)
        n_obs = 6 + self.n_wheels * (2 if wheel_speed_obs else 1)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n_obs,), dtype=np.float32)

        self._steps = 0
        self._tau = np.zeros(self.n_wheels)     # coppia motore corrente [N·m]
        self.T_external = np.zeros(3)           # coppia esterna di prova (body) [N·m], 0 nel training
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
        self.last_tel = tel                     # ultima telemetria completa (per GUI e analisi)

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
        tel = self.engine.step(self._tau, self.T_external)
        self.last_tel = tel
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

            r = − k_err · θ − k_log · ln(1 + θ/θ0)              [θ in gradi]
                − k_accel · Σ_assi max(0, |α_i| − α_max)        [α in °/s²]
                − k_null · |Ω_null| / (√N · Ω_max)                 [Ω_max = 6000 RPM]
                + bonus · [θ < θ_bonus] + bonus2 · [θ < θ_bonus2]

        Il primo termine penalizza l'errore d'assetto: la parte lineare domina
        agli angoli grandi (spinge a fare in fretta la manovra), quella
        logaritmica agli angoli piccoli (premia ogni miglioramento di
        precisione, anche sotto le soglie dei bonus). Il
        secondo penalizza solo le accelerazioni oltre soglia. I bonus premiano
        ogni passo trascorso vicino al target (0.1°) e molto vicino (0.01°),
        quindi il restarci con la precisione richiesta. L'ultimo termine penalizza
        solo la parte di velocità delle ruote nello spazio nullo della piramide,
        Ω_null = (I − A⁺A)·Ω, cioè le combinazioni (+,−,+,−) che non producono
        momento sul corpo: non limita la velocità utile alla manovra. Diviso per
        √N vale la velocità di deriva di ogni ruota (le componenti del vettore
        nullo hanno tutte modulo 1/2), in [0, 1].
        """
        c = self.reward_config
        alpha_deg = np.degrees(np.abs(self._alpha))
        accel_excess = np.maximum(0.0, alpha_deg - c["alpha_max_deg"]).sum()
        theta = tel.att_err_deg
        bonus = (c["bonus"] * (theta < c["bonus_theta_deg"])
                 + c["bonus2"] * (theta < c["bonus2_theta_deg"]))
        null_drift = np.linalg.norm(self.null_proj @ tel.wheel_speed) / (
            np.sqrt(self.n_wheels) * self.omega_w_max)
        err_pen = c["k_err"] * theta + c["k_log"] * np.log1p(theta / c["log_theta0_deg"])
        return float(-err_pen - c["k_accel"] * accel_excess
                      - c["k_null"] * null_drift + bonus)

    def _get_obs(self, tel: Telemetry) -> np.ndarray:
        """Osservazione: errore d'assetto in scala log, ω normalizzata, coppia corrente
        normalizzata e velocità delle ruote dalla telemetria (rad/s), normalizzata
        con Ω_max: ±1 = ruota in saturazione."""
        parts = [log_attitude_error(tel.q_err), tel.omega / OMEGA_SCALE, self._tau / self.tau_max]
        if self.wheel_speed_obs:
            parts.append(tel.wheel_speed / self.omega_w_max)
        return np.concatenate(parts).astype(np.float32)

    def _info(self, tel: Telemetry) -> dict:
        return dict(att_err_deg=tel.att_err_deg, alpha_deg=np.degrees(self._alpha),
                    tau=self._tau.copy(), power=tel.power_total,
                    wheel_saturation=tel.wheel_saturation.max())
