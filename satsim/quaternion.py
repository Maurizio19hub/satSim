"""
Quaternion algebra.

Convention: scalar first, q = [q0, q1, q2, q3] = [cos(θ/2), sin(θ/2)·e],
Hamilton product. The attitude quaternion q rotates vectors from the
body frame (B) to the inertial frame (I):

    v_I = q ⊗ [0, v_B] ⊗ q*      ⇔      v_I = R(q) · v_B
"""
import numpy as np


def cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross product a × b for 3D vectors or rows of Nx3 matrices.

    Equivalent to np.cross but much faster on small arrays (np.cross has
    a large overhead that dominates the cost of an RK4 step).
    """
    a1, a2, a3 = a[..., 0], a[..., 1], a[..., 2]
    b1, b2, b3 = b[..., 0], b[..., 1], b[..., 2]
    return np.stack((a2 * b3 - a3 * b2, a3 * b1 - a1 * b3, a1 * b2 - a2 * b1), axis=-1)


def quat_mult(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hamilton product p ⊗ q."""
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
    """Rotation matrix R(q) body → inertial."""
    q0, q1, q2, q3 = q
    return np.array([
        [1 - 2 * (q2**2 + q3**2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
        [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1**2 + q3**2), 2 * (q2 * q3 - q0 * q1)],
        [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1**2 + q2**2)],
    ])


def quat_error(q_target: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Error quaternion q_e = q_target* ⊗ q (rotation from the target to the current attitude).

    It is returned with q_e0 >= 0 so that it always represents the shortest
    rotation (q and -q describe the same attitude: double cover of SO(3)).
    """
    qe = quat_mult(quat_conj(q_target), q)
    return qe if qe[0] >= 0 else -qe


def quat_angle_deg(q_err: np.ndarray) -> float:
    """Equivalent rotation angle (axis-angle) in degrees."""
    return np.degrees(2.0 * np.arccos(np.clip(abs(q_err[0]), -1.0, 1.0)))


def quat_to_euler_zyx_deg(q: np.ndarray) -> np.ndarray:
    """Euler angles (roll, pitch, yaw), Z-Y-X sequence, for display only."""
    q0, q1, q2, q3 = q
    roll = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1**2 + q2**2))
    pitch = np.arcsin(np.clip(2 * (q0 * q2 - q3 * q1), -1.0, 1.0))
    yaw = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))
    return np.degrees([roll, pitch, yaw])


def random_quaternion(rng: np.random.Generator) -> np.ndarray:
    """Quaternion uniformly distributed on SO(3) (Shoemake's method)."""
    u1, u2, u3 = rng.random(3)
    return np.array([
        np.sqrt(u1) * np.cos(2 * np.pi * u3),
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
    ])
