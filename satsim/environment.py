"""
Modello d'ambiente orbitale e coppie di disturbo.

Fornisce, in funzione del tempo t e dell'assetto q, le tre coppie di disturbo
espresse nel riferimento body:

    T_gg  : gradiente di gravità
    T_srp : pressione di radiazione solare (modello a 6 facce + ombra cilindrica)
    T_mag : coppia magnetica residua (campo terrestre a dipolo)
"""
import numpy as np

from .config import (B0_EARTH, DIPOLE_AXIS_INERTIAL, MU_EARTH, R_EARTH,
                     SOLAR_PRESSURE, DisturbanceParams, OrbitParams,
                     SpacecraftParams)
from .quaternion import cross3


class OrbitalEnvironment:
    def __init__(self, orbit: OrbitParams, dist: DisturbanceParams, sc: SpacecraftParams):
        self.orbit = orbit
        self.dist = dist

        self.radius = R_EARTH + orbit.altitude
        self.mean_motion = np.sqrt(MU_EARTH / self.radius**3)      # n [rad/s]
        self.period = 2 * np.pi / self.mean_motion

        inc = np.radians(orbit.inclination_deg)
        raan = np.radians(orbit.raan_deg)
        # Matrice di rotazione dal piano orbitale al riferimento inerziale (ECI):
        # R3(-RAAN) · R1(-i)
        c, s = np.cos(raan), np.sin(raan)
        R3 = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        c, s = np.cos(inc), np.sin(inc)
        R1 = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        self._orbit_to_eci = R3 @ R1
        self._u0 = np.radians(orbit.initial_arg_lat_deg)

        sun = np.asarray(dist.sun_direction_inertial, dtype=float)
        self.sun_inertial = sun / np.linalg.norm(sun)
        self.residual_dipole = np.asarray(dist.residual_dipole, dtype=float)

        self._build_faces(sc)

    # ------------------------------------------------------------------ orbita
    def position(self, t: float) -> np.ndarray:
        """Posizione del satellite in ECI [m] su orbita circolare."""
        u = self._u0 + self.mean_motion * t
        return self._orbit_to_eci @ (self.radius * np.array([np.cos(u), np.sin(u), 0.0]))

    def in_eclipse(self, r: np.ndarray) -> bool:
        """Modello d'ombra cilindrico: la Terra proietta un cilindro di raggio R_E."""
        proj = r @ self.sun_inertial
        if proj >= 0:
            return False
        return np.linalg.norm(r - proj * self.sun_inertial) < R_EARTH

    def magnetic_field_inertial(self, r: np.ndarray) -> np.ndarray:
        """Campo terrestre a dipolo: B = B0 (R_E/r)³ [3(m̂·r̂)r̂ − m̂]  [T]."""
        rn = np.linalg.norm(r)
        r_hat = r / rn
        m_hat = DIPOLE_AXIS_INERTIAL
        return B0_EARTH * (R_EARTH / rn) ** 3 * (3 * (m_hat @ r_hat) * r_hat - m_hat)

    # ------------------------------------------------------------------- SRP
    def _build_faces(self, sc: SpacecraftParams):
        a, b, c = sc.size
        cm = np.asarray(sc.cm_offset, dtype=float)
        normals, areas, centers = [], [], []
        for axis, (half, area) in enumerate([(a / 2, b * c), (b / 2, a * c), (c / 2, a * b)]):
            for sign in (+1, -1):
                n = np.zeros(3)
                n[axis] = sign
                normals.append(n)
                areas.append(area)
                centers.append(n * half - cm)   # braccio rispetto al centro di massa
        self.face_normals = np.array(normals)
        self.face_areas = np.array(areas)
        self.face_arms = np.array(centers)

    def srp_torque(self, sun_body: np.ndarray) -> np.ndarray:
        """Somma dei contributi delle facce illuminate.

        Per ogni faccia con cosθ = n̂·ŝ > 0:
            F = −P·A·cosθ·[(1−ρs)·ŝ + 2(ρs·cosθ + ρd/3)·n̂]
            T = r_cp × F
        """
        rs, rd = self.dist.rho_spec, self.dist.rho_diff
        cos_t = self.face_normals @ sun_body
        lit = cos_t > 0
        if not np.any(lit):
            return np.zeros(3)
        ct = cos_t[lit][:, None]
        n = self.face_normals[lit]
        A = self.face_areas[lit][:, None]
        F = -SOLAR_PRESSURE * A * ct * ((1 - rs) * sun_body + 2 * (rs * ct + rd / 3) * n)
        return cross3(self.face_arms[lit], F).sum(axis=0)

    # ------------------------------------------------------------ disturbi
    def disturbance_torques(self, t: float, R_bi: np.ndarray, J: np.ndarray) -> dict:
        """Calcola T_gg, T_srp, T_mag nel riferimento body.

        R_bi: matrice di rotazione body → inerziale, R(q).
        """
        r_I = self.position(t)
        R_ib = R_bi.T
        out = {"r_I": r_I, "eclipse": False,
               "gg": np.zeros(3), "srp": np.zeros(3), "mag": np.zeros(3)}

        if self.dist.enable_gravity_gradient:
            rn = np.linalg.norm(r_I)
            r_b = R_ib @ (r_I / rn)
            # T_gg = 3μ/|r|³ · r̂_B × (J · r̂_B)
            out["gg"] = 3 * MU_EARTH / rn**3 * cross3(r_b, J @ r_b)

        if self.dist.enable_srp:
            eclipse = self.in_eclipse(r_I)
            out["eclipse"] = eclipse
            if not eclipse:
                out["srp"] = self.srp_torque(R_ib @ self.sun_inertial)

        if self.dist.enable_magnetic:
            B_b = R_ib @ self.magnetic_field_inertial(r_I)
            # T_mag = m_res × B_B
            out["mag"] = cross3(self.residual_dipole, B_b)

        return out
