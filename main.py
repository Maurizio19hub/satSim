#!/usr/bin/env python3
"""
satSim — visual ADCS simulator for a 3U CubeSat with reaction wheels.

Usage:
    python main.py                       # 4 wheels in a pyramid, real time
    python main.py --wheels 3 --speed 5  # 3 orthogonal wheels, 5x real time
    python main.py --compare --seed 101  # PPO (left) vs PD (right) comparison
"""
import argparse
import sys

import pyqtgraph as pg
from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow, apply_dark_palette
from satsim import SimParams
from satsim.simulation import ClosedLoopSimulation


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--wheels", type=int, choices=(3, 4), default=4,
                    help="3 = orthogonal, 4 = pyramid (default)")
    ap.add_argument("--dt", type=float, default=0.05, help="RK4 step [s] (default 0.05)")
    ap.add_argument("--speed", type=float, default=1.0, help="initial real-time factor")
    ap.add_argument("--seed", type=int, default=None, help="seed for the initial conditions")
    ap.add_argument("--compare", action="store_true",
                    help="PPO vs PD comparison from the same initial condition\n"
                         "(requires gymnasium and stable-baselines3)")
    ap.add_argument("--model", default="rl/pretrained/local_training_v4_seed1@3_best",
                    help="PPO model for --compare (default rl/pretrained/local_training_v4_seed1@3_best)")
    return ap.parse_args()


def main():
    args = parse_args()
    params = SimParams(dt=args.dt)
    params.wheels.configuration = "pyramid4" if args.wheels == 4 else "orthogonal3"

    pg.setConfigOptions(antialias=True)
    app = QApplication(sys.argv)
    apply_dark_palette(app)

    if args.compare:
        # Imported only in this mode: the normal GUI does not need the RL libraries.
        from gui.compare_window import CompareWindow
        win = CompareWindow(args.model, seed=args.seed or 0, speed=args.speed)
    else:
        sim = ClosedLoopSimulation(params, seed=args.seed)
        win = MainWindow(sim, speed=args.speed)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
