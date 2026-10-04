"""
Main window: connects the simulation (satsim) to the display widgets.

The GUI contains no physics: it only reads the telemetry and sends commands
(pause, manual torques, enabling disturbances/controller).
"""
import numpy as np
from PySide6.QtCore import QElapsedTimer, QSize, Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QGroupBox,
                               QHBoxLayout, QLabel, QMainWindow, QPushButton,
                               QScrollArea, QSlider, QSplitter, QVBoxLayout,
                               QWidget)

from satsim.simulation import ClosedLoopSimulation

from . import theme
from .dashboard import ModelPanel, TelemetryPanel, TelemetryPlots, format_model_text
from .view3d import AttitudeView

FRAME_MS = 33                 # ~30 fps
MAX_STEPS_PER_FRAME = 400     # keeps the GUI responsive at high speeds
MANUAL_TORQUE_MAX_MNM = 5.0   # full scale of the sliders [mN·m]


class MainWindow(QMainWindow):
    def __init__(self, sim: ClosedLoopSimulation, speed: float = 1.0):
        super().__init__()
        self.sim = sim
        self.engine = sim.engine
        self.paused = False
        self.speed = speed
        self._acc = 0.0
        self._frame = 0

        self.setWindowTitle("satSim — 3U CubeSat ADCS · Reaction wheels")
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
        model_box = QGroupBox("Active mathematical model")
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

    # ------------------------------------------------------------ controls
    def _build_controls(self) -> QWidget:
        box = QGroupBox("Controls")
        g = QGridLayout(box)

        def button(text, icon_name, slot, variant=None, icon_color=theme.TEXT):
            b = QPushButton(text)
            b.setIcon(theme.icon(icon_name, icon_color))
            b.setIconSize(QSize(16, 16))
            b.setCursor(Qt.PointingHandCursor)
            if variant:
                b.setProperty("variant", variant)
            b.clicked.connect(slot)
            return b

        self.btn_pause = button("Pause  [Space]", "mdi6.pause", self.toggle_pause, "accent", "white")
        btn_reset = button("Reset (random tumbling)", "mdi6.restore", self._reset)
        btn_kick = button("Perturb attitude 45°", "mdi6.rotate-3d-variant",
                          lambda: self.sim.kick_attitude(45.0))
        btn_imp = button("Impulse 5 mN·m × 1 s", "mdi6.flash", self._impulse, "warning", theme.WARNING)
        g.addWidget(self.btn_pause, 0, 0)
        g.addWidget(btn_reset, 0, 1)
        g.addWidget(btn_kick, 1, 0)
        g.addWidget(btn_imp, 1, 1)

        row = QHBoxLayout()
        row.addWidget(QLabel("Simulation speed:"))
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
            ("PD controller", self.sim.controller_enabled,
             lambda v: setattr(self.sim, "controller_enabled", v)),
            ("Gravity gradient", d.enable_gravity_gradient,
             lambda v: setattr(d, "enable_gravity_gradient", v)),
            ("Solar pressure", d.enable_srp, lambda v: setattr(d, "enable_srp", v)),
            ("Magnetic dipole", d.enable_magnetic, lambda v: setattr(d, "enable_magnetic", v)),
        ]
        crow = QHBoxLayout()
        for label, state, cb in checks:
            c = QCheckBox(label)
            c.setChecked(state)
            c.toggled.connect(cb)
            crow.addWidget(c)
        g.addLayout(crow, 3, 0, 1, 2)

        tbox = QGroupBox(f"Manual disturbance torque (body) ±{MANUAL_TORQUE_MAX_MNM:g} mN·m")
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
        btn_zero = QPushButton("Reset manual torques")
        btn_zero.setIcon(theme.icon("mdi6.restore"))
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
        self.btn_pause.setText("Resume  [Space]" if self.paused else "Pause  [Space]")
        self.btn_pause.setIcon(theme.icon("mdi6.play" if self.paused else "mdi6.pause", "white"))

    # ------------------------------------------------------------- loop
    def _tick(self):
        elapsed = self.clock.restart() / 1000.0
        n = 0
        if not self.paused:
            self._acc += elapsed * self.speed
            n = int(self._acc / self.engine.dt)
            if n > MAX_STEPS_PER_FRAME:          # the CPU cannot keep up: slow down
                n, self._acc = MAX_STEPS_PER_FRAME, 0.0
            else:
                self._acc -= n * self.engine.dt
            self.sim.advance(n)
        self._frame += 1
        self._refresh(force_model=False)
        rtf = n * self.engine.dt / elapsed if elapsed > 0 else 0
        self.statusBar().showMessage(
            f"t_sim = {self.sim.last.t:8.2f} s   |   real-time factor ≈ {rtf:5.2f}×   |   "
            f"{'PAUSED' if self.paused else 'RUNNING'}   |   "
            f"orbit {self.engine.params.orbit.altitude / 1e3:.0f} km, "
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
    """Compatibility: the theme is now defined in gui/theme.py."""
    theme.apply_theme(app)
