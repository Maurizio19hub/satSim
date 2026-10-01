"""
Finestra di confronto PPO vs PD (python main.py --compare).

Due satelliti partono dalla stessa condizione iniziale (stesso seed) e
avanzano insieme: a sinistra l'agente PPO addestrato, a destra il PD. I
grafici sovrappongono le curve dei due controllori. La logica sta in
rl/compare.py; questa finestra visualizza soltanto.
"""
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QElapsedTimer, Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QComboBox, QGridLayout, QGroupBox, QHBoxLayout,
                               QLabel, QMainWindow, QPushButton, QSpinBox,
                               QSplitter, QVBoxLayout, QWidget)

from rl.compare import IMPULSE_DURATION, IMPULSE_TORQUE, Comparison

from .dashboard import MONO
from .view3d import AttitudeView

FRAME_MS = 33
MAX_STEPS_PER_FRAME = 200
COLORS = {"PPO": (80, 170, 255), "PD": (255, 160, 60)}
SEED_MAX = 2**31 - 1


def _dash(color=(200, 80, 80)):
    return pg.mkPen(color, width=1, style=Qt.DashLine)


class ComparePlots(pg.GraphicsLayoutWidget):
    """Grafici con le curve di PPO e PD sovrapposte."""

    SPECS = [  # (chiave storico, titolo, unità, scala log)
        ("err", "Errore d'assetto (scala log)", "deg", True),
        ("w", "Velocità angolare |ω|", "deg/s", False),
        ("alpha", "Accelerazione angolare max_i |α_i|", "deg/s²", False),
        ("rpm", "Velocità ruote max |Ω|", "RPM", False),
        ("power", "Potenza elettrica ruote", "W", False),
    ]

    def __init__(self, alpha_max_deg: float, max_rpm: float, parent=None):
        super().__init__(parent)
        self.setBackground((16, 18, 24))
        self.curves = {}
        first = None
        for row, (key, title, units, log) in enumerate(self.SPECS):
            p = self.addPlot(row=row, col=0, title=title)
            p.showGrid(x=True, y=True, alpha=0.25)
            p.setLabel("left", units)
            p.addLegend(offset=(5, 5), labelTextSize="8pt", brush=(20, 20, 30, 160))
            if log:
                p.setLogMode(y=True)
                for th in (1.0, 0.01):          # 1°, soglia del criterio di successo
                    p.addItem(pg.InfiniteLine(pos=np.log10(th), angle=0, pen=_dash((120, 120, 140))))
            if key == "alpha":
                p.addItem(pg.InfiniteLine(pos=alpha_max_deg, angle=0, pen=_dash()))
            if key == "rpm":
                p.addItem(pg.InfiniteLine(pos=max_rpm, angle=0, pen=_dash()))
            if first is None:
                first = p
            else:
                p.setXLink(first)
            self.curves[key] = {name: p.plot(pen=pg.mkPen(c, width=1.8), name=name)
                                for name, c in COLORS.items()}
        p.setLabel("bottom", "tempo simulato", "s")

    def update_curves(self, runs):
        for r in runs:
            h = r.arrays()
            if h["t"].size < 2:
                continue
            for key, c in self.curves.items():
                c[r.name].setData(h["t"], h[key])


class CompareWindow(QMainWindow):
    def __init__(self, model_path: str, seed: int = 0, speed: float = 1.0):
        super().__init__()
        self.cmp = Comparison(model_path, seed)
        self.paused = False
        self.speed = speed
        self._acc = 0.0
        self.rng = np.random.default_rng()

        self.setWindowTitle("satSim — Confronto PPO vs PD")
        self.resize(1760, 1000)

        env = self.cmp.runs[0].env
        size, A = env.engine.params.spacecraft.size, env.engine.rw.A
        views = QWidget()
        hv = QHBoxLayout(views)
        hv.setContentsMargins(0, 0, 0, 0)
        self.views = []
        for r in self.cmp.runs:
            box = QGroupBox(f"{r.name}  —  {'agente ' + model_path if r.name == 'PPO' else 'controllore PD'}")
            c = COLORS[r.name]
            box.setStyleSheet(f"QGroupBox {{ color: rgb{c}; font-weight: bold; }}")
            v = AttitudeView(size, A)
            QVBoxLayout(box).addWidget(v)
            hv.addWidget(box)
            self.views.append(v)

        self.table = QLabel()
        self.table.setFont(MONO)
        self.table.setTextFormat(Qt.PlainText)
        self.table.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        table_box = QGroupBox("Confronto")
        QVBoxLayout(table_box).addWidget(self.table)

        bottom = QWidget()
        hb = QHBoxLayout(bottom)
        hb.setContentsMargins(0, 0, 0, 0)
        hb.addWidget(self._build_controls(), stretch=1)
        hb.addWidget(table_box, stretch=2)

        left = QSplitter(Qt.Vertical)
        left.addWidget(views)
        left.addWidget(bottom)
        left.setSizes([640, 340])

        self.plots = ComparePlots(env.reward_config["alpha_max_deg"], env.engine.rw.p.max_rpm)
        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(self.plots)
        split.setSizes([1100, 660])
        self.setCentralWidget(split)

        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self.toggle_pause)
        self.clock = QElapsedTimer()
        self.clock.start()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(FRAME_MS)
        self._refresh()

    # ----------------------------------------------------------- controlli
    def _build_controls(self) -> QWidget:
        box = QGroupBox("Comandi")
        g = QGridLayout(box)

        g.addWidget(QLabel("Seed:"), 0, 0)
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, SEED_MAX)
        self.spin_seed.setValue(self.cmp.seed)
        g.addWidget(self.spin_seed, 0, 1)
        btn_start = QPushButton("▶ Avvia con questo seed")
        btn_start.clicked.connect(lambda: self._start(self.spin_seed.value()))
        g.addWidget(btn_start, 1, 0, 1, 2)
        btn_rand = QPushButton("🎲 Seed casuale")
        btn_rand.clicked.connect(self._random_seed)
        g.addWidget(btn_rand, 2, 0, 1, 2)

        self.btn_pause = QPushButton("⏸ Pausa  [Spazio]")
        self.btn_pause.clicked.connect(self.toggle_pause)
        g.addWidget(self.btn_pause, 3, 0, 1, 2)
        btn_imp = QPushButton(f"⚡ Impulso {IMPULSE_TORQUE * 1e3:g} mN·m × {IMPULSE_DURATION:g} s (su entrambi)")
        btn_imp.clicked.connect(lambda: self.cmp.apply_impulse(self.rng))
        g.addWidget(btn_imp, 4, 0, 1, 2)

        g.addWidget(QLabel("Velocità:"), 5, 0)
        self.cmb_speed = QComboBox()
        speeds = [0.25, 0.5, 1, 2, 5, 10]
        for s in speeds:
            self.cmb_speed.addItem(f"{s}×", s)
        self.cmb_speed.setCurrentIndex(speeds.index(self.speed) if self.speed in speeds else 2)
        self.cmb_speed.currentIndexChanged.connect(
            lambda _: setattr(self, "speed", self.cmb_speed.currentData()))
        g.addWidget(self.cmb_speed, 5, 1)
        g.setRowStretch(6, 1)
        return box

    def _start(self, seed: int):
        self.cmp.reset(seed)
        self.spin_seed.setValue(seed)
        self._acc = 0.0
        if self.paused:
            self.toggle_pause()
        self._refresh()

    def _random_seed(self):
        self._start(int(self.rng.integers(0, 100_000)))

    def toggle_pause(self):
        self.paused = not self.paused
        self.btn_pause.setText("▶ Riprendi  [Spazio]" if self.paused else "⏸ Pausa  [Spazio]")

    # ------------------------------------------------------------- loop
    def _tick(self):
        elapsed = self.clock.restart() / 1000.0
        if not self.paused and not self.cmp.done:
            self._acc += elapsed * self.speed
            n = int(self._acc / self.cmp.dt)
            if n > MAX_STEPS_PER_FRAME:          # la CPU non tiene il passo: rallenta
                n, self._acc = MAX_STEPS_PER_FRAME, 0.0
            else:
                self._acc -= n * self.cmp.dt
            self.cmp.advance(n)
        self._refresh()

    def _refresh(self):
        c = self.cmp
        for v, r in zip(self.views, c.runs):
            tel = r.tel
            v.update_state(tel.q, r.env.engine.env.sun_inertial, tel.r_inertial, tel.eclipse)
        self.plots.update_curves(c.runs)
        self.table.setText(self._table_text())
        state = "EPISODIO CONCLUSO" if c.done else ("IN PAUSA" if self.paused else "IN ESECUZIONE")
        self.statusBar().showMessage(
            f"seed {c.seed}   |   θ0 = {c.theta0:.1f}°, |ω0| = {c.omega0:.2f} °/s   |   "
            f"t_sim = {c.t:6.2f} / {c.runs[0].env.max_episode_steps * c.dt:.0f} s   |   {state}")

    def _table_text(self) -> str:
        ppo, pd = self.cmp.runs
        f_t = lambda x: f"{x:9.2f} s" if x is not None else "        –  "

        def row(label, a, b, fmt):
            return f"{label:<26}{format(a, fmt):>14}{format(b, fmt):>14}"

        lines = [
            f"seed {self.cmp.seed}:  θ0 = {self.cmp.theta0:.1f}°   |ω0| = {self.cmp.omega0:.2f} °/s",
            "",
            f"{'':<26}{'PPO':>14}{'PD':>14}",
            row("Errore d'assetto [°]", ppo.tel.att_err_deg, pd.tel.att_err_deg, ".4f"),
            row("|ω| [°/s]", np.degrees(np.linalg.norm(ppo.tel.omega)),
                np.degrees(np.linalg.norm(pd.tel.omega)), ".4f"),
            row("|α| max finora [°/s²]", ppo.alpha_max, pd.alpha_max, ".2f"),
            row("Max |Ω| ruote [RPM]", np.abs(ppo.tel.wheel_rpm).max(),
                np.abs(pd.tel.wheel_rpm).max(), ".0f"),
            row("Energia [J]", ppo.tel.energy, pd.tel.energy, ".1f"),
            row("Reward accumulata", ppo.ret, pd.ret, ".1f"),
            f"{'Tempo per scendere < 1°':<26}{f_t(ppo.t_1deg):>14}{f_t(pd.t_1deg):>14}",
            f"{'Tempo per scendere < 0.01°':<26}{f_t(ppo.t_001deg):>14}{f_t(pd.t_001deg):>14}",
        ]
        if self.cmp.done:
            better = "PPO" if ppo.ret > pd.ret else "PD"
            lines += ["", f"Episodio concluso (100 s): reward migliore → {better} "
                          f"({max(ppo.ret, pd.ret):.1f} vs {min(ppo.ret, pd.ret):.1f})"]
        return "\n".join(lines)
