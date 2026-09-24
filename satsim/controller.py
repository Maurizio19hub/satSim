"""
Controllore PD sui quaternioni (placeholder in attesa dell'agente RL).

Legge di controllo (coppia desiderata sul corpo, riferimento body):

    q_e   = q_target* ⊗ q                    (con q_e0 ≥ 0: rotazione più breve)
    T_cmd = −J·(Kp·q_e,vec + Kd·ω) + ω × (J·ω + h_rw)

Il primo termine è un PD "normalizzato per l'inerzia": per piccoli angoli
q_e,vec ≈ θ/2, quindi ogni asse si comporta come un oscillatore del secondo
ordine θ̈ + Kd·θ̇ + (Kp/2)·θ = 0, con

    Kp = 2·ωn²,   Kd = 2·ζ·ωn

Il secondo termine (opzionale) compensa l'accoppiamento giroscopico.
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
