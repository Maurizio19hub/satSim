"""satSim — simulatore di controllo d'assetto (ADCS) per CubeSat con ruote di reazione."""
from .config import SimParams
from .controller import QuaternionPDController
from .physics_engine import SatelliteEngine, Telemetry

__version__ = "0.1.0"
__all__ = ["SimParams", "SatelliteEngine", "Telemetry", "QuaternionPDController"]
