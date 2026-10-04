"""satSim — attitude determination and control (ADCS) simulator for a CubeSat with reaction wheels."""
from .config import SimParams
from .controller import QuaternionPDController
from .physics_engine import SatelliteEngine, Telemetry

__version__ = "0.1.0"
__all__ = ["SimParams", "SatelliteEngine", "Telemetry", "QuaternionPDController"]
