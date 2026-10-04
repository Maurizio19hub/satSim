"""
Quaternion PD controller (placeholder until the RL agent replaces it).

Control law (desired body torque, body frame):

    q_e   = q_target* ⊗ q                    (with q_e0 ≥ 0: shortest rotation)
    T_cmd = −J·(Kp·q_e,vec + Kd·ω) + ω × (J·ω + h_rw)

The first term is an "inertia-normalised" PD: for small angles
q_e,vec ≈ θ/2, so each axis behaves like a second-order oscillator
θ̈ + Kd·θ̇ + (Kp/2)·θ = 0, with

    Kp = 2·ωn²,   Kd = 2·ζ·ωn

The second (optional) term compensates the gyroscopic coupling.
"""
import numpy as np

from .quaternion import cross3, quat_error


class QuaternionPDController:
    def __init__(self, J: np.ndarray, omega_n: float = 0.4, zeta: float = 0.9,
                 gyro_compensation: bool = True):
        self.J = J
        self.gyro_compensation = gyro_compensation
        self.set_bandwidth(omega_n, zeta)

    def set_bandwidth(self, omega_n: float, zeta: float):
        self.omega_n, self.zeta = omega_n, zeta
        self.Kp = 2 * omega_n**2
        self.Kd = 2 * zeta * omega_n

    def compute(self, q: np.ndarray, omega: np.ndarray, h_rw: np.ndarray,
                q_target: np.ndarray) -> np.ndarray:
        q_e = quat_error(q_target, q)
        T = -self.J @ (self.Kp * q_e[1:] + self.Kd * omega)
        if self.gyro_compensation:
            T += cross3(omega, self.J @ omega + h_rw)
        return T
