"""
Confronto passo-passo tra l'agente PPO e il controllore PD sulla stessa
condizione iniziale (stesso seed), senza dipendenze grafiche.

Entrambi i controllori girano in un SatAttitudeEnv: stessa fisica, stessi
disturbi, stesso limite di variazione della coppia (imposto dal motore) e
stessa reward. La GUI di confronto (gui/compare_window.py) si limita a
visualizzare lo stato di questa classe.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from rl.adcs_env import SatAttitudeEnv
from rl.evaluate import pd_action
from satsim.controller import QuaternionPDController

DEFAULT_MODEL = "rl/pretrained/ppo_adcs_v11"
IMPULSE_TORQUE = 5e-3       # [N·m]
IMPULSE_DURATION = 1.0      # [s]


class ControllerRun:
    """Un satellite controllato da `policy` ("PD" oppure un modello SB3)."""

    FIELDS = ("t", "err", "w", "alpha", "rpm", "power")

    def __init__(self, name: str, policy):
        self.name, self.policy = name, policy
        kwargs = {}
        if policy != "PD":                   # ambiente con l'osservazione del modello
            from rl.models import env_kwargs_for
            kwargs = env_kwargs_for(policy)
        self.env = SatAttitudeEnv(**kwargs)
        self.pd = QuaternionPDController(self.env.engine.J)
        self.hist = {k: deque(maxlen=self.env.max_episode_steps + 1) for k in self.FIELDS}

    def reset(self, seed: int):
        self.obs, info = self.env.reset(seed=seed)
        self.tel = self.env.last_tel
        self.ret = 0.0
        self.alpha_max = 0.0
        self.t_1deg = self.t_001deg = None
        self.done = False
        for d in self.hist.values():
            d.clear()
        self._record(info)

    def step(self):
        if self.done:
            return
        if self.policy == "PD":
            a = pd_action(self.env, self.pd)
        else:
            a, _ = self.policy.predict(self.obs, deterministic=True)
        self.obs, r, terminated, truncated, info = self.env.step(a)
        self.ret += r
        self.alpha_max = max(self.alpha_max, float(np.abs(info["alpha_deg"]).max()))
        self.tel = self.env.last_tel
        t = self.tel.t
        if self.t_1deg is None and info["att_err_deg"] < 1.0:
            self.t_1deg = t
        if self.t_001deg is None and info["att_err_deg"] < 0.01:
            self.t_001deg = t
        self.done = terminated or truncated
        self._record(info)

    def _record(self, info: dict):
        h, tel = self.hist, self.tel
        h["t"].append(tel.t)
        h["err"].append(max(tel.att_err_deg, 1e-5))         # > 0 per la scala log
        h["w"].append(np.degrees(np.linalg.norm(tel.omega)))
        h["alpha"].append(float(np.abs(info["alpha_deg"]).max()))
        h["rpm"].append(float(np.abs(tel.wheel_rpm).max()))
        h["power"].append(tel.power_total)

    def arrays(self) -> dict:
        return {k: np.asarray(v) for k, v in self.hist.items()}


class Comparison:
    """Due satelliti (PPO a sinistra, PD a destra) che avanzano insieme."""

    def __init__(self, model_path: str = DEFAULT_MODEL, seed: int = 0):
        from rl.models import load_model
        self.model_path = model_path
        self.runs = [ControllerRun("PPO", load_model(model_path)),
                     ControllerRun("PD", "PD")]
        self.dt = self.runs[0].env.dt
        self._impulse_left = 0.0
        self.reset(seed)

    @property
    def done(self) -> bool:
        return all(r.done for r in self.runs)

    @property
    def t(self) -> float:
        return self.runs[0].tel.t

    def reset(self, seed: int):
        self.seed = seed
        self._impulse_left = 0.0
        for r in self.runs:
            r.env.T_external = np.zeros(3)
            r.reset(seed)
        q, w = self.runs[0].env.engine.q, self.runs[0].env.engine.omega
        self.theta0 = 2 * np.degrees(np.arccos(min(1.0, abs(q[0]))))
        self.omega0 = np.degrees(np.linalg.norm(w))

    def apply_impulse(self, rng: np.random.Generator | None = None):
        """Stessa coppia esterna (direzione casuale, 5 mN·m per 1 s) su entrambi."""
        rng = rng or np.random.default_rng()
        d = rng.normal(size=3)
        T = IMPULSE_TORQUE * d / np.linalg.norm(d)
        for r in self.runs:
            r.env.T_external = T.copy()
        self._impulse_left = IMPULSE_DURATION

    def step(self):
        for r in self.runs:
            r.step()
        if self._impulse_left > 0:
            self._impulse_left -= self.dt
            if self._impulse_left <= 1e-9:
                for r in self.runs:
                    r.env.T_external = np.zeros(3)

    def advance(self, n: int):
        for _ in range(n):
            if self.done:
                break
            self.step()
