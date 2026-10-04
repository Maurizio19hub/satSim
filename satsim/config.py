"""
Configuration parameters of the ADCS simulator.

All quantities are in SI units (kg, m, s, N·m, A·m², T, W, J)
unless explicitly stated otherwise (e.g. RPM).
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SpacecraftParams:
    """3U CubeSat modelled as a homogeneous rectangular box."""
    mass: float = 3.0                                   # [kg]
    size: tuple = (0.10, 0.10, 0.30)                    # [m] sides along x_B, y_B, z_B
    # Residual products of inertia (misalignments, non-uniform distribution
    # of the components). They make J non-diagonal as in a real satellite.
    products_of_inertia: tuple = (2.0e-5, -1.0e-5, 1.5e-5)   # (Jxy, Jxz, Jyz) [kg·m²]
    # Offset of the centre of mass from the geometric centre: it creates the
    # lever arm for the solar radiation pressure torque.
    cm_offset: tuple = (0.002, -0.001, 0.010)           # [m]


@dataclass
class ReactionWheelParams:
    """Reaction wheel array (typical values for CubeSat wheels)."""
    configuration: str = "pyramid4"     # "orthogonal3" or "pyramid4"
    pyramid_beta_deg: float = 54.7356   # angle between wheel axis and z_B axis (pyramid config.)
    inertia: float = 1.5e-5             # I_rw, axial inertia of the rotor [kg·m²]
    max_rpm: float = 6000.0             # speed saturation [RPM]
    max_torque: float = 2.0e-3          # maximum motor torque [N·m]
    # Maximum rate of change of the commanded torque (motor driver):
    # 8e-3 N·m/s → from 0 to T_max in 0.25 s. Avoids abrupt torque jumps
    # (vibrations, current peaks, wear). float("inf") = no limit.
    max_torque_rate: float = 8.0e-3     # [N·m/s]
    # Power model: P = k1*|T| + k2*|T*Omega| + P_static  (per wheel)
    k1: float = 50.0                    # [W/(N·m)]  resistive losses ~ current ∝ torque
    k2: float = 1.2                     # [-]        inverse of the mechanical efficiency
    p_static: float = 0.15              # [W]        control electronics, always on


@dataclass
class OrbitParams:
    """Circular Keplerian orbit (LEO)."""
    altitude: float = 500e3             # [m]
    inclination_deg: float = 51.6       # [deg]
    raan_deg: float = 0.0               # [deg]
    initial_arg_lat_deg: float = 0.0    # [deg] initial argument of latitude


@dataclass
class DisturbanceParams:
    enable_gravity_gradient: bool = True
    enable_srp: bool = True
    enable_magnetic: bool = True
    # Residual magnetic dipole of the satellite (stray currents, magnets) [A·m²]
    residual_dipole: tuple = (0.005, -0.003, 0.010)
    # Optical coefficients of the surfaces (specular / diffuse reflection)
    rho_spec: float = 0.1
    rho_diff: float = 0.3
    # Sun direction in the inertial frame (assumed constant over the time
    # scale of the simulation)
    sun_direction_inertial: tuple = (1.0, 0.4, 0.2)


@dataclass
class SimParams:
    dt: float = 0.05                    # fixed RK4 integration step [s]
    spacecraft: SpacecraftParams = field(default_factory=SpacecraftParams)
    wheels: ReactionWheelParams = field(default_factory=ReactionWheelParams)
    orbit: OrbitParams = field(default_factory=OrbitParams)
    disturbances: DisturbanceParams = field(default_factory=DisturbanceParams)


# --- Physical constants -----------------------------------------------------
MU_EARTH = 3.986004418e14       # Earth's gravitational parameter [m³/s²]
R_EARTH = 6.371e6               # mean Earth radius [m]
B0_EARTH = 3.12e-5              # equatorial magnetic field at the surface [T]
SOLAR_PRESSURE = 4.56e-6        # solar radiation pressure at 1 AU [N/m²]
DIPOLE_AXIS_INERTIAL = np.array([0.0, 0.0, -1.0])   # Earth's dipole (≈ anti-aligned with z)
