"""
Anello chiuso engine + controllore + disturbi manuali, con storico della telemetria.

Non dipende dalla GUI: può essere usato anche da script headless o notebook.
In futuro l'ambiente Gymnasium sostituirà il controllore PD con l'azione
dell'agente RL chiamando direttamente SatelliteEngine.step().
"""
from collections import deque

import numpy as np

from .config import SimParams
from .controller import QuaternionPDController
from .physics_engine import SatelliteEngine, Telemetry
from .quaternion import quat_from_axis_angle, quat_mult, quat_normalize


class TelemetryHistory:
    """Buffer circolare delle grandezze da plottare (finestra temporale fissa)."""
    FIELDS = ("t", "q", "att_err", "omega", "rpm", "power", "energy")

    def __init__(self, window_s: float, dt: float):
        n = int(window_s / dt) + 1
        self.buf = {k: deque(maxlen=n) for k in self.FIELDS}

    def clear(self):
        for d in self.buf.values():
            d.clear()

    def append(self, tel: Telemetry):
        b = self.buf
        b["t"].append(tel.t)
        b["q"].append(tel.q)
        b["att_err"].append(tel.att_err_deg)
        b["omega"].append(np.degrees(tel.omega))
        b["rpm"].append(tel.wheel_rpm)
        b["power"].append(tel.power_total)
        b["energy"].append(tel.energy)

    def arrays(self) -> dict:
        return {k: np.asarray(v) for k, v in self.buf.items()}


class ClosedLoopSimulation:
    def __init__(self, params: SimParams | None = None, history_window_s: float = 60.0,
                 seed: int | None = None):
        self.engine = SatelliteEngine(params)
        self.controller = QuaternionPDController(self.engine.J)
        self.controller_enabled = True
        self.T_manual = np.zeros(3)          # coppia manuale continua (slider)
        self._impulse = np.zeros(3)          # coppia impulsiva temporanea
        self._impulse_left = 0.0
        self.rng = np.random.default_rng(seed)
        self.history = TelemetryHistory(history_window_s, self.engine.dt)
        self.last: Telemetry | None = None
        self.reset()

    # ----------------------------------------------------------- comandi
    def reset(self, tumble: bool = True):
        """Condizione iniziale: assetto casuale a ~60° dal target + rotazione residua (post-rilascio)."""
        if tumble:
            axis = self.rng.normal(size=3)
            q0 = quat_from_axis_angle(axis, np.radians(self.rng.uniform(40, 80)))
            w0 = self.rng.uniform(-0.05, 0.05, 3)
        else:
            q0, w0 = None, None
        self._impulse_left = 0.0
        self.history.clear()
        self.last = self.engine.reset(q0=q0, omega0=w0)
        self.history.append(self.last)

    def kick_attitude(self, angle_deg: float = 45.0):
        """Ruota istantaneamente l'assetto di un angolo casuale (test di stabilizzazione)."""
        dq = quat_from_axis_angle(self.rng.normal(size=3), np.radians(angle_deg))
        self.engine.set_attitude(quat_normalize(quat_mult(self.engine.q, dq)))

    def apply_impulse(self, torque_body, duration_s: float = 1.0):
        self._impulse = np.asarray(torque_body, float)
        self._impulse_left = duration_s

    # ---------------------------------------------------------- avanzamento
    def step(self) -> Telemetry:
        e = self.engine
        if self.controller_enabled:
            T_cmd = self.controller.compute(e.q, e.omega, e.h_rw, e.q_target)
            tau = e.allocate(T_cmd)
        else:
            tau = np.zeros(e.rw.n)

        T_ext = self.T_manual.copy()
        if self._impulse_left > 0:
            T_ext += self._impulse
            self._impulse_left -= e.dt

        self.last = e.step(tau, T_ext)
        self.history.append(self.last)
        return self.last

    def advance(self, n_steps: int) -> Telemetry:
        for _ in range(n_steps):
            self.step()
        return self.last
