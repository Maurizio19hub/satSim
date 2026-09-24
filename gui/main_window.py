"""
Finestra principale: collega la simulazione (satsim) ai widget di visualizzazione.

La GUI non contiene fisica: legge solo la telemetria e invia comandi
(pausa, coppie manuali, abilitazione disturbi/controllore).
"""
import numpy as np
from PySide6.QtCore import QElapsedTimer, Qt, QTimer
from PySide6.QtGui import QColor, QKeySequence, QPalette, QShortcut
from PySide6.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QGroupBox,
                               QHBoxLayout, QLabel, QMainWindow, QPushButton,
                               QScrollArea, QSlider, QSplitter, QVBoxLayout,
                               QWidget)

from satsim.simulation import ClosedLoopSimulation

from .dashboard import ModelPanel, TelemetryPanel, TelemetryPlots, format_model_text
from .view3d import AttitudeView

FRAME_MS = 33                 # ~30 fps
MAX_STEPS_PER_FRAME = 400     # evita il blocco della GUI a velocità elevate
MANUAL_TORQUE_MAX_MNM = 5.0   # fondo scala degli slider [mN·m]


class MainWindow(QMainWindow):
    def __init__(self, sim: ClosedLoopSimulation, speed: float = 1.0):
        super().__init__()
        self.sim = sim
        self.engine = sim.engine
        self.paused = False
        self.speed = speed
        self._acc = 0.0
        self._frame = 0

        self.setWindowTitle("satSim — ADCS CubeSat 3U · Ruote di reazione")
        self.resize(1720, 980)

        rw = self.engine.rw
        self.view = AttitudeView(self.engine.params.spacecraft.size, rw.A)
        self.plots = TelemetryPlots(rw.n, rw.p.max_rpm)
        self.panel = TelemetryPanel(rw.n)
        self.model = ModelPanel()

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(4, 4, 4, 4)
        lv.addWidget(self.view, stretch=1)
        lv.addWidget(self._build_controls())

        panel_scroll = QScrollArea()
        panel_scroll.setWidgetResizable(True)
        panel_scroll.setWidget(self.panel)

        right = QSplitter(Qt.Vertical)
        right.addWidget(panel_scroll)
        model_box = QGroupBox("Modello matematico attivo")
        QVBoxLayout(model_box).addWidget(self.model)
        right.addWidget(model_box)
        right.setSizes([560, 420])

        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(self.plots)
        split.addWidget(right)
        split.setSizes([560, 600, 560])
        self.setCentralWidget(split)

        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self.toggle_pause)

        self.clock = QElapsedTimer()
        self.clock.start()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(FRAME_MS)
        self._refresh(force_model=True)

    # ----------------------------------------------------------- controlli
    def _build_controls(self) -> QWidget:
        box = QGroupBox("Comandi")
        g = QGridLayout(box)

        self.btn_pause = QPushButton("⏸ Pausa  [Spazio]")
        self.btn_pause.clicked.connect(self.toggle_pause)
        btn_reset = QPushButton("⟲ Reset (tumbling casuale)")
        btn_reset.clicked.connect(self._reset)
        btn_kick = QPushButton("↻ Perturba assetto 45°")
        btn_kick.clicked.connect(lambda: self.sim.kick_attitude(45.0))
        btn_imp = QPushButton("⚡ Impulso 5 mN·m × 1 s")
        btn_imp.clicked.connect(self._impulse)
        g.addWidget(self.btn_pause, 0, 0)
        g.addWidget(btn_reset, 0, 1)
        g.addWidget(btn_kick, 1, 0)
        g.addWidget(btn_imp, 1, 1)

        row = QHBoxLayout()
        row.addWidget(QLabel("Velocità simulazione:"))
        self.cmb_speed = QComboBox()
        speeds = [0.25, 0.5, 1, 2, 5, 10, 20]
        for s in speeds:
            self.cmb_speed.addItem(f"{s}×", s)
        self.cmb_speed.setCurrentIndex(speeds.index(self.speed) if self.speed in speeds else 2)
        self.cmb_speed.currentIndexChanged.connect(
            lambda _: setattr(self, "speed", self.cmb_speed.currentData()))
        row.addWidget(self.cmb_speed)
        row.addStretch()
        g.addLayout(row, 2, 0, 1, 2)

        d = self.engine.params.disturbances
        checks = [
            ("Controllore PD", self.sim.controller_enabled,
             lambda v: setattr(self.sim, "controller_enabled", v)),
            ("Gradiente di gravità", d.enable_gravity_gradient,
             lambda v: setattr(d, "enable_gravity_gradient", v)),
            ("Pressione solare", d.enable_srp, lambda v: setattr(d, "enable_srp", v)),
            ("Dipolo magnetico", d.enable_magnetic, lambda v: setattr(d, "enable_magnetic", v)),
        ]
        crow = QHBoxLayout()
        for label, state, cb in checks:
            c = QCheckBox(label)
            c.setChecked(state)
            c.toggled.connect(cb)
            crow.addWidget(c)
        g.addLayout(crow, 3, 0, 1, 2)

        tbox = QGroupBox(f"Coppia di disturbo manuale (body) ±{MANUAL_TORQUE_MAX_MNM:g} mN·m")
        tg = QGridLayout(tbox)
        self.sliders, self.slider_labels = [], []
        for i, ax in enumerate("xyz"):
            s = QSlider(Qt.Horizontal)
            s.setRange(-50, 50)
            s.setValue(0)
            s.valueChanged.connect(self._manual_torque_changed)
            lab = QLabel("+0.0 mN·m")
            lab.setMinimumWidth(80)
            tg.addWidget(QLabel(f"T_{ax}"), i, 0)
            tg.addWidget(s, i, 1)
            tg.addWidget(lab, i, 2)
            self.sliders.append(s)
            self.slider_labels.append(lab)
        btn_zero = QPushButton("Azzera coppie manuali")
        btn_zero.clicked.connect(lambda: [s.setValue(0) for s in self.sliders])
        tg.addWidget(btn_zero, 3, 0, 1, 3)
        g.addWidget(tbox, 4, 0, 1, 2)
        return box

    def _manual_torque_changed(self):
        scale = MANUAL_TORQUE_MAX_MNM / 50.0
        vals = np.array([s.value() * scale for s in self.sliders])     # mN·m
        for lab, v in zip(self.slider_labels, vals):
            lab.setText(f"{v:+.1f} mN·m")
        self.sim.T_manual = vals * 1e-3

    def _impulse(self):
        d = self.sim.rng.normal(size=3)
        self.sim.apply_impulse(5e-3 * d / np.linalg.norm(d), duration_s=1.0)

    def _reset(self):
        self.sim.reset()
        self._acc = 0.0
        self._refresh(force_model=True)

    def toggle_pause(self):
        self.paused = not self.paused
        self.btn_pause.setText("▶ Riprendi  [Spazio]" if self.paused else "⏸ Pausa  [Spazio]")

    # ------------------------------------------------------------- loop
    def _tick(self):
        elapsed = self.clock.restart() / 1000.0
        n = 0
        if not self.paused:
            self._acc += elapsed * self.speed
            n = int(self._acc / self.engine.dt)
            if n > MAX_STEPS_PER_FRAME:          # la CPU non tiene il passo: rallenta
                n, self._acc = MAX_STEPS_PER_FRAME, 0.0
            else:
                self._acc -= n * self.engine.dt
            self.sim.advance(n)
        self._frame += 1
        self._refresh(force_model=False)
        rtf = n * self.engine.dt / elapsed if elapsed > 0 else 0
        self.statusBar().showMessage(
            f"t_sim = {self.sim.last.t:8.2f} s   |   fattore tempo reale ≈ {rtf:5.2f}×   |   "
            f"{'IN PAUSA' if self.paused else 'IN ESECUZIONE'}   |   "
            f"orbita {self.engine.params.orbit.altitude / 1e3:.0f} km, "
            f"T = {self.engine.env.period / 60:.1f} min")

    def _refresh(self, force_model: bool):
        tel = self.sim.last
        self.view.update_state(tel.q, self.engine.env.sun_inertial, tel.r_inertial, tel.eclipse)
        self.plots.update_curves(self.sim.history.arrays())
        self.panel.update_values(tel)
        if force_model or self._frame % 8 == 0:
            self.model.update_text(format_model_text(
                self.engine, tel, self.sim.controller, self.sim.controller_enabled))


def apply_dark_palette(app):
    app.setStyle("Fusion")
    pal = QPalette()
    bg, base, text = QColor(30, 32, 40), QColor(20, 22, 28), QColor(220, 222, 230)
    pal.setColor(QPalette.Window, bg)
    pal.setColor(QPalette.WindowText, text)
    pal.setColor(QPalette.Base, base)
    pal.setColor(QPalette.AlternateBase, bg)
    pal.setColor(QPalette.Text, text)
    pal.setColor(QPalette.Button, QColor(45, 48, 58))
    pal.setColor(QPalette.ButtonText, text)
    pal.setColor(QPalette.Highlight, QColor(60, 120, 200))
    pal.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    pal.setColor(QPalette.ToolTipBase, base)
    pal.setColor(QPalette.ToolTipText, text)
    app.setPalette(pal)
