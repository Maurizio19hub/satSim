"""
Parametri di configurazione del simulatore ADCS.

Tutte le grandezze sono in unità SI (kg, m, s, N·m, A·m², T, W, J)
salvo dove indicato esplicitamente (es. RPM).
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SpacecraftParams:
    """CubeSat 3U modellato come parallelepipedo omogeneo."""
    mass: float = 3.0                                   # [kg]
    size: tuple = (0.10, 0.10, 0.30)                    # [m] lati lungo x_B, y_B, z_B
    # Prodotti d'inerzia residui (disallineamenti, distribuzione non uniforme
    # dei componenti). Rendono J non diagonale come in un satellite reale.
    products_of_inertia: tuple = (2.0e-5, -1.0e-5, 1.5e-5)   # (Jxy, Jxz, Jyz) [kg·m²]
    # Offset del centro di massa rispetto al centro geometrico: genera il
    # braccio di leva per la coppia da pressione solare.
    cm_offset: tuple = (0.002, -0.001, 0.010)           # [m]


@dataclass
class ReactionWheelParams:
    """Array di ruote di reazione (valori tipici per ruote CubeSat)."""
    configuration: str = "pyramid4"     # "orthogonal3" oppure "pyramid4"
    pyramid_beta_deg: float = 54.7356   # angolo tra asse ruota e asse z_B (config. piramidale)
    inertia: float = 1.5e-5             # I_rw, inerzia assiale del rotore [kg·m²]
    max_rpm: float = 6000.0             # saturazione in velocità [RPM]
    max_torque: float = 2.0e-3          # coppia massima del motore [N·m]
    # Massima velocità di variazione della coppia comandata (driver del motore):
    # 8e-3 N·m/s → da 0 a T_max in 0.25 s. Evita salti bruschi di coppia
    # (vibrazioni, picchi di corrente, usura). float("inf") = nessun limite.
    max_torque_rate: float = 8.0e-3     # [N·m/s]
    # Modello di potenza: P = k1*|T| + k2*|T*Omega| + P_static  (per ruota)
    k1: float = 50.0                    # [W/(N·m)]  perdite resistive ~ corrente ∝ coppia
    k2: float = 1.2                     # [-]        inverso efficienza meccanica
    p_static: float = 0.15              # [W]        elettronica di controllo sempre attiva


@dataclass
class OrbitParams:
    """Orbita circolare kepleriana (LEO)."""
    altitude: float = 500e3             # [m]
    inclination_deg: float = 51.6       # [deg]
    raan_deg: float = 0.0               # [deg]
    initial_arg_lat_deg: float = 0.0    # [deg] argomento di latitudine iniziale


@dataclass
class DisturbanceParams:
    enable_gravity_gradient: bool = True
    enable_srp: bool = True
    enable_magnetic: bool = True
    # Dipolo magnetico residuo del satellite (correnti parassite, magneti) [A·m²]
    residual_dipole: tuple = (0.005, -0.003, 0.010)
    # Coefficienti ottici delle superfici (riflessione speculare / diffusa)
    rho_spec: float = 0.1
    rho_diff: float = 0.3
    # Direzione del Sole nel riferimento inerziale (assunta costante sulla scala
    # temporale della simulazione)
    sun_direction_inertial: tuple = (1.0, 0.4, 0.2)


@dataclass
class SimParams:
    dt: float = 0.05                    # passo fisso di integrazione RK4 [s]
    spacecraft: SpacecraftParams = field(default_factory=SpacecraftParams)
    wheels: ReactionWheelParams = field(default_factory=ReactionWheelParams)
    orbit: OrbitParams = field(default_factory=OrbitParams)
    disturbances: DisturbanceParams = field(default_factory=DisturbanceParams)


# --- Costanti fisiche -------------------------------------------------------
MU_EARTH = 3.986004418e14       # parametro gravitazionale terrestre [m³/s²]
R_EARTH = 6.371e6               # raggio medio terrestre [m]
B0_EARTH = 3.12e-5              # campo magnetico equatoriale al suolo [T]
SOLAR_PRESSURE = 4.56e-6        # pressione di radiazione solare a 1 UA [N/m²]
DIPOLE_AXIS_INERTIAL = np.array([0.0, 0.0, -1.0])   # dipolo terrestre (≈ antiallineato con z)
