"""
Algebra dei quaternioni.

Convenzione: scalare per primo, q = [q0, q1, q2, q3] = [cos(θ/2), sin(θ/2)·e],
prodotto di Hamilton. Il quaternione d'assetto q ruota vettori dal
riferimento body (B) al riferimento inerziale (I):

    v_I = q ⊗ [0, v_B] ⊗ q*      ⇔      v_I = R(q) · v_B
"""
import numpy as np


def cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Prodotto vettoriale a × b per vettori 3D o righe di matrici Nx3.

    Equivalente a np.cross ma molto più veloce su array piccoli (np.cross ha
    un overhead elevato che domina il costo di un passo RK4).
    """
    a1, a2, a3 = a[..., 0], a[..., 1], a[..., 2]
    b1, b2, b3 = b[..., 0], b[..., 1], b[..., 2]
    return np.stack((a2 * b3 - a3 * b2, a3 * b1 - a1 * b3, a1 * b2 - a2 * b1), axis=-1)


def quat_mult(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Prodotto di Hamilton p ⊗ q."""
    p0, p1, p2, p3 = p
    q0, q1, q2, q3 = q
    return np.array([
        p0 * q0 - p1 * q1 - p2 * q2 - p3 * q3,
        p0 * q1 + p1 * q0 + p2 * q3 - p3 * q2,
        p0 * q2 - p1 * q3 + p2 * q0 + p3 * q1,
        p0 * q3 + p1 * q2 - p2 * q1 + p3 * q0,
    ])


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)


def quat_from_axis_angle(axis, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    return np.concatenate(([np.cos(angle_rad / 2)], np.sin(angle_rad / 2) * axis))


def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    """Matrice di rotazione R(q) body → inerziale."""
    q0, q1, q2, q3 = q
    return np.array([
        [1 - 2 * (q2**2 + q3**2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
        [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1**2 + q3**2), 2 * (q2 * q3 - q0 * q1)],
        [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1**2 + q2**2)],
    ])


def quat_error(q_target: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Quaternione d'errore q_e = q_target* ⊗ q (rotazione dal target all'assetto attuale).

    Viene restituito con q_e0 >= 0 per rappresentare sempre la rotazione più
    breve (q e -q descrivono lo stesso assetto: doppia copertura di SO(3)).
    """
    qe = quat_mult(quat_conj(q_target), q)
    return qe if qe[0] >= 0 else -qe


def quat_angle_deg(q_err: np.ndarray) -> float:
    """Angolo di rotazione equivalente (asse-angolo) in gradi."""
    return np.degrees(2.0 * np.arccos(np.clip(abs(q_err[0]), -1.0, 1.0)))


def quat_to_euler_zyx_deg(q: np.ndarray) -> np.ndarray:
    """Angoli di Eulero (roll, pitch, yaw) sequenza Z-Y-X, solo per visualizzazione."""
    q0, q1, q2, q3 = q
    roll = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1**2 + q2**2))
    pitch = np.arcsin(np.clip(2 * (q0 * q2 - q3 * q1), -1.0, 1.0))
    yaw = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))
    return np.degrees([roll, pitch, yaw])


def random_quaternion(rng: np.random.Generator) -> np.ndarray:
    """Quaternione uniformemente distribuito su SO(3) (metodo di Shoemake)."""
    u1, u2, u3 = rng.random(3)
    return np.array([
        np.sqrt(u1) * np.cos(2 * np.pi * u3),
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
    ])
