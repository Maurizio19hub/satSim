#!/usr/bin/env python3
"""
satSim — simulatore visuale ADCS per CubeSat 3U con ruote di reazione.

Uso:
    python main.py                       # 4 ruote in piramide, tempo reale
    python main.py --wheels 3 --speed 5  # 3 ruote ortogonali, 5x tempo reale
    python main.py --compare --seed 101  # confronto PPO (sinistra) vs PD (destra)
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
                    help="3 = ortogonali, 4 = piramide (default)")
    ap.add_argument("--dt", type=float, default=0.05, help="passo RK4 [s] (default 0.05)")
    ap.add_argument("--speed", type=float, default=1.0, help="fattore tempo reale iniziale")
    ap.add_argument("--seed", type=int, default=None, help="seed per le condizioni iniziali")
    ap.add_argument("--compare", action="store_true",
                    help="confronto PPO vs PD sulla stessa condizione iniziale\n"
                         "(richiede gymnasium e stable-baselines3)")
    ap.add_argument("--model", default="rl/pretrained/ppo_adcs_v8",
                    help="modello PPO per --compare (default rl/pretrained/ppo_adcs_v8)")
    return ap.parse_args()


def main():
    args = parse_args()
    params = SimParams(dt=args.dt)
    params.wheels.configuration = "pyramid4" if args.wheels == 4 else "orthogonal3"

    pg.setConfigOptions(antialias=True)
    app = QApplication(sys.argv)
    apply_dark_palette(app)

    if args.compare:
        # Import solo in questa modalità: la GUI normale non richiede le librerie RL.
        from gui.compare_window import CompareWindow
        win = CompareWindow(args.model, seed=args.seed or 0, speed=args.speed)
    else:
        sim = ClosedLoopSimulation(params, seed=args.seed)
        win = MainWindow(sim, speed=args.speed)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
