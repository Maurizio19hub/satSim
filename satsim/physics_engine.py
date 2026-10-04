"""
Physics engine of the ADCS simulator — fully independent of the GUI.

State vector (size 7 + N + 1):

    x = [ q0 q1 q2 q3 | ωx ωy ωz | Ω1 … ΩN | E ]

    q  : attitude quaternion body → inertial (scalar first)
    ω  : body angular velocity, expressed in body axes [rad/s]
    Ω  : axial spin rates of the N wheels [rad/s]
    E  : electrical energy consumed by the wheels [J]

Implemented equations (see TECHNICAL_DOCUMENTATION.md for the derivation):

    [1] J  (3x3 inertia tensor of the 3U CubeSat)
    [2] dΩ/dt = T_rw / I_rw                     (+ saturation at ±Ω_max,
                                                 torque rate limit)
        P     = k1|T_rw| + k2|T_rw·Ω| + P_static
    [3] J dω/dt = T_tot − ω × (J ω + h_rw),   h_rw = A · I_rw · Ω
    [4] dq/dt = ½ q ⊗ [0, ω]                    (+ normalisation at every step)
    [5] T_tot = T_rw_action + T_gg + T_srp + T_mag (+ T_manual)

Integration is fixed-step RK4; the wheel torque command is held constant
during the step (zero-order hold), as in a real on-board computer and as in
a Gymnasium environment.
"""
from dataclasses import dataclass

import numpy as np

from .config import SimParams
from .environment import OrbitalEnvironment
from .quaternion import (cross3, quat_angle_deg, quat_error, quat_mult,
                         quat_normalize, quat_to_dcm)

RPM_TO_RADS = 2 * np.pi / 60.0


# =============================================================================
# [1] Inertia matrix
# =============================================================================
def cubesat_inertia(mass: float, size, products=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Inertia tensor of a homogeneous box about its centre of mass.

        Jxx = m/12 (b² + c²),  Jyy = m/12 (a² + c²),  Jzz = m/12 (a² + b²)

    plus optional products of inertia (convention J_ij = −∫ x_i x_j dm).
    """
    a, b, c = size
    jxy, jxz, jyz = products
    J = np.array([
        [mass / 12 * (b**2 + c**2), jxy, jxz],
        [jxy, mass / 12 * (a**2 + c**2), jyz],
        [jxz, jyz, mass / 12 * (a**2 + b**2)],
    ])
    # Physical consistency check: positive definite
    assert np.all(np.linalg.eigvalsh(J) > 0), "J must be positive definite"
    return J


# =============================================================================
# [2] Reaction wheel array
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
        self.tau_rate_max = p.max_torque_rate

    @staticmethod
    def _spin_axes(config: str, beta: float) -> np.ndarray:
        """Distribution matrix A: column i = spin axis of wheel i in body axes."""
        if config == "orthogonal3":
            return np.eye(3)
        if config == "pyramid4":
            phis = np.radians([45, 135, 225, 315])
            return np.array([[np.sin(beta) * np.cos(f), np.sin(beta) * np.sin(f), np.cos(beta)]
                             for f in phis]).T
        raise ValueError(f"Unknown wheel configuration: {config}")

    def rate_limit(self, tau_prev: np.ndarray, tau_cmd: np.ndarray, dt: float) -> np.ndarray:
        """Limits the change of the commanded torque relative to the previous step.

            |T(t+dt) − T(t)| ≤ dT_max = max_torque_rate · dt     (per wheel)

        Models the motor driver, which cannot change the current (∝ torque)
        instantaneously. The command is first limited to ±T_max.
        """
        tau = np.clip(tau_cmd, -self.tau_max, self.tau_max)
        d_max = self.tau_rate_max * dt
        return tau_prev + np.clip(tau - tau_prev, -d_max, d_max)

    def limit_torque(self, tau_cmd: np.ndarray, omega_w: np.ndarray, dt: float) -> np.ndarray:
        """Applies the physical limits to the motor torque command.

        1) |T_rw| ≤ T_max
        2) speed saturation: since dΩ/dt = T/I_rw and T is constant over the
           step, Ω(t+dt) = Ω + T·dt/I_rw exactly. T is therefore limited so
           that |Ω(t+dt)| ≤ Ω_max: the wheel never exceeds the limit and the
           total angular momentum is conserved (no clipping).
        """
        tau = np.clip(tau_cmd, -self.tau_max, self.tau_max)
        tau_hi = (self.omega_max - omega_w) * self.I_rw / dt
        tau_lo = (-self.omega_max - omega_w) * self.I_rw / dt
        return np.clip(tau, tau_lo, tau_hi)

    def power(self, tau: np.ndarray, omega_w: np.ndarray) -> np.ndarray:
        """P_i = k1|T_i| + k2|T_i·Ω_i| + P_static   [W] for each wheel."""
        p = self.p
        return p.k1 * np.abs(tau) + p.k2 * np.abs(tau * omega_w) + p.p_static

    def momentum_body(self, omega_w: np.ndarray) -> np.ndarray:
        """h_rw = A · I_rw · Ω  [N·m·s] in the body frame."""
        return self.A @ (self.I_rw * omega_w)


# =============================================================================
# Telemetry returned at every step
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
    wheel_saturation: np.ndarray  # fraction 0..1 of Ω_max
    wheel_torque_cmd: np.ndarray  # requested command [N·m]
    wheel_torque: np.ndarray      # torque actually applied [N·m]
    wheel_accel: np.ndarray       # dΩ/dt [rad/s²]
    h_rw: np.ndarray              # wheel angular momentum (body)
    H_inertial: np.ndarray        # total angular momentum (inertial)
    T_rw_body: np.ndarray         # reaction torque on the body
    T_gg: np.ndarray
    T_srp: np.ndarray
    T_mag: np.ndarray
    T_manual: np.ndarray
    T_total: np.ndarray
    power_wheels: np.ndarray      # [W] per wheel
    power_total: float            # [W]
    energy: float                 # [J]
    eclipse: bool
    r_inertial: np.ndarray        # orbital position [m]


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

    # ------------------------------------------------------------- state
    def reset(self, q0=None, omega0=None, wheel_speed0=None) -> Telemetry:
        self.t = 0.0
        x = np.zeros(self.n_state)
        x[0:4] = quat_normalize(np.asarray(q0, float)) if q0 is not None else [1, 0, 0, 0]
        x[4:7] = omega0 if omega0 is not None else 0.0
        x[7:7 + self.rw.n] = wheel_speed0 if wheel_speed0 is not None else 0.0
        self.x = x
        self.tau_cmd = np.zeros(self.rw.n)     # last commanded torque (for the rate limit)
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

    # ------------------------------------------------------- allocation
    def allocate(self, T_body_cmd: np.ndarray) -> np.ndarray:
        """Converts a desired body torque into wheel motor torques.

        The reaction on the body is T_body = −A·T_rw, hence T_rw = −A⁺·T_body
        (Moore-Penrose pseudo-inverse: minimum-norm solution, useful with
        4 redundant wheels).
        """
        return -self.rw.A_pinv @ T_body_cmd

    # --------------------------------------------------------- dynamics
    def _derivatives(self, t, x, tau_w, T_manual):
        """f(t, x, u) — right-hand side of the ODE system."""
        n = self.rw.n
        q = x[0:4]
        w = x[4:7]
        Om = x[7:7 + n]
        qn = q / np.linalg.norm(q)          # the torques use the normalised attitude

        # [5] Environmental disturbances (evaluated at every RK4 stage)
        R_bi = quat_to_dcm(qn)
        d = self.env.disturbance_torques(t, R_bi, self.J)
        T_rw_body = -self.rw.A @ tau_w
        T_total = T_rw_body + d["gg"] + d["srp"] + d["mag"] + T_manual

        # [2] Wheels: dΩ/dt = T_rw / I_rw
        Om_dot = tau_w / self.rw.I_rw

        # [3] Euler with gyroscopic coupling
        h = self.rw.A @ (self.rw.I_rw * Om)
        w_dot = self.J_inv @ (T_total - cross3(w, self.J @ w + h))

        # [4] Kinematics: dq/dt = ½ q ⊗ [0, ω]
        q_dot = 0.5 * quat_mult(q, np.concatenate(([0.0], w)))

        # Electrical power → dE/dt = P
        P = self.rw.power(tau_w, Om).sum()

        xdot = np.concatenate((q_dot, w_dot, Om_dot, [P]))
        aux = dict(d, T_rw_body=T_rw_body, T_total=T_total, h=h)
        return xdot, aux

    def step(self, tau_wheel_cmd=None, T_manual=None) -> Telemetry:
        """Advances the simulation by one step dt with RK4 integration.

        tau_wheel_cmd : motor torques requested from the N wheels [N·m]
        T_manual      : additional external disturbance torque (body) [N·m]

        The requested torque goes through three limits, in this order:
        |T| ≤ T_max, change ≤ max_torque_rate·dt relative to the previous
        step, wheel speed saturation.
        """
        n = self.rw.n
        tau_cmd = np.zeros(n) if tau_wheel_cmd is None else np.asarray(tau_wheel_cmd, float)
        T_manual = np.zeros(3) if T_manual is None else np.asarray(T_manual, float)
        self.tau_cmd = self.rw.rate_limit(self.tau_cmd, tau_cmd, self.dt)
        tau = self.rw.limit_torque(self.tau_cmd, self.x[7:7 + n], self.dt)

        dt, t, x = self.dt, self.t, self.x
        k1, _ = self._derivatives(t, x, tau, T_manual)
        k2, _ = self._derivatives(t + dt / 2, x + dt / 2 * k1, tau, T_manual)
        k3, _ = self._derivatives(t + dt / 2, x + dt / 2 * k2, tau, T_manual)
        k4, _ = self._derivatives(t + dt, x + dt * k3, tau, T_manual)
        x_new = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # Quaternion normalisation (constraint ‖q‖ = 1)
        x_new[0:4] = quat_normalize(x_new[0:4])
        # Numerical safety net (limit_torque already keeps Ω exactly within Ω_max)
        x_new[7:7 + n] = np.clip(x_new[7:7 + n], -self.rw.omega_max, self.rw.omega_max)

        self.x = x_new
        self.t = t + dt
        return self._telemetry(tau_cmd, tau, T_manual)

    # -------------------------------------------------------- telemetry
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
