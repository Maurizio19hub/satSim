"""
Engine fisico del simulatore ADCS — completamente indipendente dalla GUI.

Vettore di stato (dimensione 7 + N + 1):

    x = [ q0 q1 q2 q3 | ωx ωy ωz | Ω1 … ΩN | E ]

    q  : quaternione d'assetto body → inerziale (scalare per primo)
    ω  : velocità angolare del corpo, espressa in body [rad/s]
    Ω  : velocità di rotazione assiale delle N ruote [rad/s]
    E  : energia elettrica consumata dalle ruote [J]

Equazioni implementate (vedi README.md per la derivazione):

    [1] J  (tensore d'inerzia 3x3 del CubeSat 3U)
    [2] dΩ/dt = T_rw / I_rw                     (+ saturazione a ±Ω_max)
        P     = k1|T_rw| + k2|T_rw·Ω| + P_static
    [3] J dω/dt = T_tot − ω × (J ω + h_rw),   h_rw = A · I_rw · Ω
    [4] dq/dt = ½ q ⊗ [0, ω]                    (+ normalizzazione ad ogni passo)
    [5] T_tot = T_rw_azione + T_gg + T_srp + T_mag (+ T_manuale)

L'integrazione è RK4 a passo fisso; il comando di coppia delle ruote è
mantenuto costante durante il passo (zero-order hold), come in un
calcolatore di bordo reale e come in un ambiente Gymnasium.
"""
from dataclasses import dataclass

import numpy as np

from .config import SimParams
from .environment import OrbitalEnvironment
from .quaternion import (cross3, quat_angle_deg, quat_error, quat_mult,
                         quat_normalize, quat_to_dcm)

RPM_TO_RADS = 2 * np.pi / 60.0


# =============================================================================
# [1] Matrice d'inerzia
# =============================================================================
def cubesat_inertia(mass: float, size, products=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Tensore d'inerzia di un parallelepipedo omogeneo rispetto al baricentro.

        Jxx = m/12 (b² + c²),  Jyy = m/12 (a² + c²),  Jzz = m/12 (a² + b²)

    più eventuali prodotti d'inerzia (convenzione J_ij = −∫ x_i x_j dm).
    """
    a, b, c = size
    jxy, jxz, jyz = products
    J = np.array([
        [mass / 12 * (b**2 + c**2), jxy, jxz],
        [jxy, mass / 12 * (a**2 + c**2), jyz],
        [jxz, jyz, mass / 12 * (a**2 + b**2)],
    ])
    # Verifica di consistenza fisica: definita positiva
    assert np.all(np.linalg.eigvalsh(J) > 0), "J deve essere definita positiva"
    return J


# =============================================================================
# [2] Array di ruote di reazione
# =============================================================================
class ReactionWheelArray:
    def __init__(self, p):
        self.p = p
        self.A = self._spin_axes(p.configuration, np.radians(p.pyramid_beta_deg))  # 3xN
        self.n = self.A.shape[1]
        self.A_pinv = np.linalg.pinv(self.A)                                          # Nx3
        self.I_rw = p.inertia
        self.omega_max = p.max_rpm * RPM_TO_RADS
        self.tau_max = p.max_torque

    @staticmethod
    def _spin_axes(config: str, beta: float) -> np.ndarray:
        """Matrice di distribuzione A: colonna i = asse di rotazione della ruota i in body."""
        if config == "orthogonal3":
            return np.eye(3)
        if config == "pyramid4":
            phis = np.radians([45, 135, 225, 315])
            return np.array([[np.sin(beta) * np.cos(f), np.sin(beta) * np.sin(f), np.cos(beta)]
                             for f in phis]).T
        raise ValueError(f"Configurazione ruote sconosciuta: {config}")

    def limit_torque(self, tau_cmd: np.ndarray, omega_w: np.ndarray, dt: float) -> np.ndarray:
        """Applica i limiti fisici al comando di coppia motore.

        1) |T_rw| ≤ T_max
        2) saturazione in velocità: poiché dΩ/dt = T/I_rw e T è costante sul
           passo, Ω(t+dt) = Ω + T·dt/I_rw esattamente. Si limita quindi T
           in modo che |Ω(t+dt)| ≤ Ω_max: la ruota non supera mai il limite
           e il momento angolare totale resta conservato (niente clipping).
        """
        tau = np.clip(tau_cmd, -self.tau_max, self.tau_max)
        tau_hi = (self.omega_max - omega_w) * self.I_rw / dt
        tau_lo = (-self.omega_max - omega_w) * self.I_rw / dt
        return np.clip(tau, tau_lo, tau_hi)

    def power(self, tau: np.ndarray, omega_w: np.ndarray) -> np.ndarray:
        """P_i = k1|T_i| + k2|T_i·Ω_i| + P_static   [W] per ogni ruota."""
        p = self.p
        return p.k1 * np.abs(tau) + p.k2 * np.abs(tau * omega_w) + p.p_static

    def momentum_body(self, omega_w: np.ndarray) -> np.ndarray:
        """h_rw = A · I_rw · Ω  [N·m·s] nel riferimento body."""
        return self.A @ (self.I_rw * omega_w)


# =============================================================================
# Telemetria restituita ad ogni passo
# =============================================================================
@dataclass
class Telemetry:
    t: float
    q: np.ndarray
    q_dot: np.ndarray
    q_err: np.ndarray
    att_err_deg: float
    omega: np.ndarray
    omega_dot: np.ndarray
    wheel_speed: np.ndarray       # [rad/s]
    wheel_rpm: np.ndarray
    wheel_saturation: np.ndarray  # frazione 0..1 di Ω_max
    wheel_torque_cmd: np.ndarray  # comando richiesto [N·m]
    wheel_torque: np.ndarray      # coppia effettivamente applicata [N·m]
    wheel_accel: np.ndarray       # dΩ/dt [rad/s²]
    h_rw: np.ndarray              # momento angolare ruote (body)
    H_inertial: np.ndarray        # momento angolare totale (inerziale)
    T_rw_body: np.ndarray         # coppia di reazione sul corpo
    T_gg: np.ndarray
    T_srp: np.ndarray
    T_mag: np.ndarray
    T_manual: np.ndarray
    T_total: np.ndarray
    power_wheels: np.ndarray      # [W] per ruota
    power_total: float            # [W]
    energy: float                 # [J]
    eclipse: bool
    r_inertial: np.ndarray        # posizione orbitale [m]


# =============================================================================
# Engine
# =============================================================================
class SatelliteEngine:
    def __init__(self, params: SimParams | None = None):
        self.params = params or SimParams()
        sc = self.params.spacecraft
        self.dt = self.params.dt
        self.J = cubesat_inertia(sc.mass, sc.size, sc.products_of_inertia)
        self.J_inv = np.linalg.inv(self.J)
        self.rw = ReactionWheelArray(self.params.wheels)
        self.env = OrbitalEnvironment(self.params.orbit, self.params.disturbances, sc)
        self.q_target = np.array([1.0, 0.0, 0.0, 0.0])
        self.n_state = 7 + self.rw.n + 1
        self.reset()

    # ------------------------------------------------------------- stato
    def reset(self, q0=None, omega0=None, wheel_speed0=None) -> Telemetry:
        self.t = 0.0
        x = np.zeros(self.n_state)
        x[0:4] = quat_normalize(np.asarray(q0, float)) if q0 is not None else [1, 0, 0, 0]
        x[4:7] = omega0 if omega0 is not None else 0.0
        x[7:7 + self.rw.n] = wheel_speed0 if wheel_speed0 is not None else 0.0
        self.x = x
        zeros = np.zeros(self.rw.n)
        return self._telemetry(zeros, zeros, np.zeros(3))

    @property
    def q(self):
        return self.x[0:4].copy()

    @property
    def omega(self):
        return self.x[4:7].copy()

    @property
    def wheel_speed(self):
        return self.x[7:7 + self.rw.n].copy()

    @property
    def h_rw(self):
        return self.rw.momentum_body(self.wheel_speed)

    def set_attitude(self, q):
        self.x[0:4] = quat_normalize(np.asarray(q, float))

    # ------------------------------------------------------ allocazione
    def allocate(self, T_body_cmd: np.ndarray) -> np.ndarray:
        """Converte una coppia desiderata sul corpo in coppie motore delle ruote.

        La reazione sul corpo è T_body = −A·T_rw, quindi T_rw = −A⁺·T_body
        (pseudo-inversa di Moore-Penrose: soluzione a minima norma, utile
        con 4 ruote ridondanti).
        """
        return -self.rw.A_pinv @ T_body_cmd

    # ------------------------------------------------------- dinamica
    def _derivatives(self, t, x, tau_w, T_manual):
        """f(t, x, u) — secondo membro del sistema di ODE."""
        n = self.rw.n
        q = x[0:4]
        w = x[4:7]
        Om = x[7:7 + n]
        qn = q / np.linalg.norm(q)          # le coppie usano l'assetto normalizzato

        # [5] Disturbi ambientali (valutati ad ogni stadio RK4)
        R_bi = quat_to_dcm(qn)
        d = self.env.disturbance_torques(t, R_bi, self.J)
        T_rw_body = -self.rw.A @ tau_w
        T_total = T_rw_body + d["gg"] + d["srp"] + d["mag"] + T_manual

        # [2] Ruote: dΩ/dt = T_rw / I_rw
        Om_dot = tau_w / self.rw.I_rw

        # [3] Eulero con accoppiamento giroscopico
        h = self.rw.A @ (self.rw.I_rw * Om)
        w_dot = self.J_inv @ (T_total - cross3(w, self.J @ w + h))

        # [4] Cinematica: dq/dt = ½ q ⊗ [0, ω]
        q_dot = 0.5 * quat_mult(q, np.concatenate(([0.0], w)))

        # Potenza elettrica → dE/dt = P
        P = self.rw.power(tau_w, Om).sum()

        xdot = np.concatenate((q_dot, w_dot, Om_dot, [P]))
        aux = dict(d, T_rw_body=T_rw_body, T_total=T_total, h=h)
        return xdot, aux

    def step(self, tau_wheel_cmd=None, T_manual=None) -> Telemetry:
        """Avanza la simulazione di un passo dt con integrazione RK4.

        tau_wheel_cmd : coppie motore richieste alle N ruote [N·m]
        T_manual      : coppia di disturbo esterna aggiuntiva (body) [N·m]
        """
        n = self.rw.n
        tau_cmd = np.zeros(n) if tau_wheel_cmd is None else np.asarray(tau_wheel_cmd, float)
        T_manual = np.zeros(3) if T_manual is None else np.asarray(T_manual, float)
        tau = self.rw.limit_torque(tau_cmd, self.x[7:7 + n], self.dt)

        dt, t, x = self.dt, self.t, self.x
        k1, _ = self._derivatives(t, x, tau, T_manual)
        k2, _ = self._derivatives(t + dt / 2, x + dt / 2 * k1, tau, T_manual)
        k3, _ = self._derivatives(t + dt / 2, x + dt / 2 * k2, tau, T_manual)
        k4, _ = self._derivatives(t + dt, x + dt * k3, tau, T_manual)
        x_new = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # Normalizzazione del quaternione (vincolo ‖q‖ = 1)
        x_new[0:4] = quat_normalize(x_new[0:4])
        # Rete di sicurezza numerica (limit_torque rende Ω esatto già entro Ω_max)
        x_new[7:7 + n] = np.clip(x_new[7:7 + n], -self.rw.omega_max, self.rw.omega_max)

        self.x = x_new
        self.t = t + dt
        return self._telemetry(tau_cmd, tau, T_manual)

    # ----------------------------------------------------- telemetria
    def _telemetry(self, tau_cmd, tau, T_manual) -> Telemetry:
        xdot, aux = self._derivatives(self.t, self.x, tau, T_manual)
        q, w, Om = self.q, self.omega, self.wheel_speed
        n = self.rw.n
        P = self.rw.power(tau, Om)
        q_err = quat_error(self.q_target, q)
        H_body = self.J @ w + aux["h"]
        return Telemetry(
            t=self.t, q=q, q_dot=xdot[0:4], q_err=q_err, att_err_deg=quat_angle_deg(q_err),
            omega=w, omega_dot=xdot[4:7],
            wheel_speed=Om, wheel_rpm=Om / RPM_TO_RADS,
            wheel_saturation=np.abs(Om) / self.rw.omega_max,
            wheel_torque_cmd=np.asarray(tau_cmd, float), wheel_torque=np.asarray(tau, float),
            wheel_accel=xdot[7:7 + n],
            h_rw=aux["h"], H_inertial=quat_to_dcm(q) @ H_body,
            T_rw_body=aux["T_rw_body"], T_gg=aux["gg"], T_srp=aux["srp"], T_mag=aux["mag"],
            T_manual=T_manual, T_total=aux["T_total"],
            power_wheels=P, power_total=float(P.sum()), energy=float(self.x[-1]),
            eclipse=aux["eclipse"], r_inertial=aux["r_I"],
        )
