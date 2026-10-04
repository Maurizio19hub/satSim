"""
Gymnasium environment for the attitude control of the 3U CubeSat with reaction
wheels, to be trained with PPO (Stable-Baselines3, see rl/train.py).

The environment wraps SatelliteEngine (package `satsim/`) and replaces the
PD controller. It imports nothing from the GUI or from Stable-Baselines3: it is just
physics + Gymnasium interface. The RL logic (spaces, observation, reward)
is documented in rl/README.md.

Current state: v11.
- Action = change of the motor torque of each wheel (rate-limited torque).
- Observation = attitude error on a logarithmic scale, angular velocity,
  current motor torque and wheel speeds (v9; can be disabled with
  wheel_speed_obs=False for models up to v8).
- Reward = linear + logarithmic penalty on the attitude error, penalty on the
  accelerations above threshold and on the wheel drift in the null space (v11),
  + two bonuses for each step close to the target (0.1° and 0.01°).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from satsim import SatelliteEngine, SimParams, Telemetry
from satsim.quaternion import quat_from_axis_angle

# Normalisation scale of ω in the observation: 0.1 rad/s ≈ 5.7 °/s → ~1.
OMEGA_SCALE = 0.1           # [rad/s]

# Logarithmic scale of the attitude error in the observation:
#   e = axis · log(1 + θ/θ0) / log(1 + π/θ0)
# θ0 sets where the scale switches from linear to logarithmic. With θ0 = 0.1°:
# 0.01° → 0.013, 0.1° → 0.092, 0.45° → 0.23, 10° → 0.62, 60° → 0.85, 180° → 1.
LOG_THETA0 = np.radians(0.1)    # [rad]

# Reward weights and thresholds (see rl/README.md §5).
REWARD_CONFIG = dict(
    k_err=0.01,             # penalty per degree of attitude error, per step
    k_log=0.02,             # weight of the logarithmic error penalty
    log_theta0_deg=0.01,    # scale of the logarithmic penalty [°]
    alpha_max_deg=2.0,      # angular acceleration threshold per axis [°/s²]
    k_accel=0.01,           # penalty per °/s² above the threshold, per axis
    bonus=0.02,             # bonus for each step with attitude error below threshold
    bonus_theta_deg=0.1,    # error threshold for the bonus [°]
    bonus2=0.02,            # second bonus (precision), added to the first
    bonus2_theta_deg=0.01,  # error threshold for the second bonus [°]
    k_null=0.1,             # penalty on the wheel speed in the null space (fraction of Ω_max)
)


def log_attitude_error(q_err: np.ndarray, theta0: float = LOG_THETA0) -> np.ndarray:
    """Attitude error as the vector axis · g(θ), with g logarithmic in [0, 1].

    q_err has q0 ≥ 0 (shortest rotation), so θ = 2·atan2(|q_vec|, q0) ∈ [0, π].
    Near zero g(θ) ≈ θ/(θ0·log(1 + π/θ0)): the signal stays readable even
    for errors of hundredths of a degree, without saturating at large rotations.
    """
    q_vec = q_err[1:]
    s = np.linalg.norm(q_vec)
    if s < 1e-12:
        return np.zeros(3)
    theta = 2.0 * np.arctan2(s, q_err[0])
    g = np.log1p(theta / theta0) / np.log1p(np.pi / theta0)
    return q_vec / s * g


class SatAttitudeEnv(gym.Env):
    """Attitude control environment: action = change of the motor torques.

    Observation (6 + 2N): [ e_log (3) | ω / OMEGA_SCALE (3) | τ / T_max (N) | Ω / Ω_max (N) ]
                  (6 + N with wheel_speed_obs=False, format of the models up to v8)
    Action (N):           normalised Δτ in [-1, 1],
                          τ ← clip(τ + action · Δτ_max, ±T_max)
                          with Δτ_max = max_torque_rate · Δt (motor limit,
                          satsim/config.py: 0.2 · T_max per 0.05 s step)
    """

    metadata = {"render_modes": []}

    def __init__(self, params: SimParams | None = None, max_episode_steps: int = 2000,
                 render_mode: str | None = None, reward_config: dict | None = None,
                 wheel_speed_obs: bool = True):
        """Initialisation: physics engine, action and observation spaces, parameters."""
        super().__init__()
        self.engine = SatelliteEngine(params)
        self.dt = self.engine.dt
        self.n_wheels = self.engine.rw.n
        self.tau_max = self.engine.rw.tau_max
        # Maximum torque change per step: the same one the motor
        # applies to any controller (engine.rw.rate_limit).
        self.dtau_max = self.engine.rw.tau_rate_max * self.dt
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.reward_config = {**REWARD_CONFIG, **(reward_config or {})}

        self.wheel_speed_obs = wheel_speed_obs
        self.omega_w_max = self.engine.rw.omega_max          # [rad/s] (6000 RPM)
        rw = self.engine.rw
        self.null_proj = np.eye(self.n_wheels) - rw.A_pinv @ rw.A   # projector onto the null space of A

        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.n_wheels,), dtype=np.float32)
        n_obs = 6 + self.n_wheels * (2 if wheel_speed_obs else 1)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n_obs,), dtype=np.float32)

        self._steps = 0
        self._tau = np.zeros(self.n_wheels)     # current motor torque [N·m]
        self.T_external = np.zeros(3)           # external test torque (body) [N·m], 0 during training
        self._prev_omega = np.zeros(3)          # ω at the previous step (for the acceleration)
        self._alpha = np.zeros(3)               # angular acceleration of the last step [rad/s²]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """New episode: random attitude 40–80° from the target, random ω,
        wheels at rest and zero torque.

        Returns: (observation, info)
        """
        super().reset(seed=seed)
        rng = self.np_random
        axis = rng.normal(size=3)
        q0 = quat_from_axis_angle(axis, np.radians(rng.uniform(40.0, 80.0)))
        omega0 = rng.uniform(-0.05, 0.05, 3)
        tel = self.engine.reset(q0=q0, omega0=omega0)
        self.last_tel = tel                     # last full telemetry (for GUI and analysis)

        self._steps = 0
        self._tau = np.zeros(self.n_wheels)
        self._prev_omega = tel.omega
        self._alpha = np.zeros(3)
        return self._get_obs(tel), self._info(tel)

    def step(self, action: np.ndarray):
        """One control step (Δt = params.dt).

        Returns: (observation, reward, terminated, truncated, info)
        """
        dtau = np.clip(action, -1.0, 1.0) * self.dtau_max
        self._tau = np.clip(self._tau + dtau, -self.tau_max, self.tau_max)
        tel = self.engine.step(self._tau, self.T_external)
        self.last_tel = tel
        self._steps += 1

        # Angular acceleration from the change in velocity between two steps
        self._alpha = (tel.omega - self._prev_omega) / self.dt
        reward = self._compute_reward(tel, action)
        self._prev_omega = tel.omega

        terminated = False
        truncated = self._steps >= self.max_episode_steps
        return self._get_obs(tel), reward, terminated, truncated, self._info(tel)

    def _compute_reward(self, tel: Telemetry, action: np.ndarray) -> float:
        """Reward of the current step.

            r = − k_err · θ − k_log · ln(1 + θ/θ0)              [θ in degrees]
                − k_accel · Σ_axes max(0, |α_i| − α_max)        [α in °/s²]
                − k_null · |Ω_null| / (√N · Ω_max)                 [Ω_max = 6000 RPM]
                + bonus · [θ < θ_bonus] + bonus2 · [θ < θ_bonus2]

        The first term penalises the attitude error: the linear part dominates
        at large angles (it pushes for a fast manoeuvre), the logarithmic part
        at small angles (it rewards every gain in precision, even below the
        bonus thresholds). The second term penalises only the accelerations
        above threshold. The bonuses reward every step spent close to the
        target (0.1°) and very close (0.01°), i.e. staying there with the
        required precision. The last term penalises only the part of the wheel
        speeds in the null space of the pyramid, Ω_null = (I − A⁺A)·Ω, i.e. the
        (+,−,+,−) combinations that produce no momentum on the body: it does not
        limit the speed useful for the manoeuvre. Divided by √N it equals the
        drift speed of each wheel (the components of the null vector all have
        magnitude 1/2), in [0, 1].
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
        """Observation: attitude error on a log scale, normalised ω, normalised
        current torque and wheel speeds from the telemetry (rad/s), normalised
        by Ω_max: ±1 = wheel at saturation."""
        parts = [log_attitude_error(tel.q_err), tel.omega / OMEGA_SCALE, self._tau / self.tau_max]
        if self.wheel_speed_obs:
            parts.append(tel.wheel_speed / self.omega_w_max)
        return np.concatenate(parts).astype(np.float32)

    def _info(self, tel: Telemetry) -> dict:
        return dict(att_err_deg=tel.att_err_deg, alpha_deg=np.degrees(self._alpha),
                    tau=self._tau.copy(), power=tel.power_total,
                    wheel_saturation=tel.wheel_saturation.max())
