"""
PPO vs PD comparison window (python main.py --compare).

Two satellites start from the same initial condition (same seed) and
advance together: the trained PPO agent on the left, the PD on the right. The
plots overlay the curves of the two controllers. The logic lives in
rl/compare.py; this window only displays it.
"""
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QElapsedTimer, QSize, Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QButtonGroup, QFrame, QGridLayout, QHBoxLayout,
                               QLabel, QMainWindow, QProgressBar, QPushButton,
                               QSizePolicy, QSpinBox, QSplitter, QVBoxLayout,
                               QWidget)

from rl.compare import IMPULSE_DURATION, IMPULSE_TORQUE, Comparison

from . import theme
from .view3d import AttitudeView

FRAME_MS = 33
MAX_STEPS_PER_FRAME = 200
COLORS = {"PPO": theme.PPO_COLOR, "PD": theme.PD_COLOR}
SPEEDS = [0.5, 1, 2, 5, 10]
SEED_MAX = 2**31 - 1


def card(parent=None) -> QFrame:
    f = QFrame(parent)
    f.setObjectName("card")
    return f


def _dash(color=theme.DANGER):
    return pg.mkPen(color, width=1, style=Qt.DashLine)


# =============================================================================
class ComparePlots(pg.GraphicsLayoutWidget):
    """Plots with the PPO and PD curves overlaid."""

    SPECS = [  # (history key, title, units, log scale)
        ("err", "Attitude error", "deg (log)", True),
        ("w", "Angular velocity |ω|", "deg/s", False),
        ("alpha", "Angular acceleration  max |α_i|", "deg/s²", False),
        ("rpm", "Wheel speed  max |Ω|", "RPM", False),
        ("power", "Wheel electrical power", "W", False),
    ]

    def __init__(self, alpha_max_deg: float, max_rpm: float, parent=None):
        super().__init__(parent)
        self.setBackground(theme.SURFACE)
        self.ci.setContentsMargins(6, 6, 10, 6)
        self.ci.setSpacing(4)
        self.curves = {}
        first = None
        for row, (key, title, units, log) in enumerate(self.SPECS):
            p = self.addPlot(row=row, col=0)
            theme.style_plot(p, title, units)
            if log:
                p.setLogMode(y=True)
                for th in (1.0, 0.01):          # 1°, success-criterion threshold
                    p.addItem(pg.InfiniteLine(pos=np.log10(th), angle=0, pen=_dash(theme.MUTED)))
            if key == "alpha":
                p.addItem(pg.InfiniteLine(pos=alpha_max_deg, angle=0, pen=_dash()))
            if key == "rpm":
                p.addItem(pg.InfiniteLine(pos=max_rpm, angle=0, pen=_dash()))
            if first is None:
                first = p
            else:
                p.setXLink(first)
            self.curves[key] = {name: p.plot(pen=pg.mkPen(c, width=2)) for name, c in COLORS.items()}
        p.setLabel("bottom", "simulated time [s]", color=theme.MUTED)

    def update_curves(self, runs):
        for r in runs:
            h = r.arrays()
            if h["t"].size < 2:
                continue
            for key, c in self.curves.items():
                c[r.name].setData(h["t"], h[key])


# =============================================================================
class MetricCard(QFrame):
    """One quantity with the PPO and PD values side by side; the better one highlighted."""

    def __init__(self, title: str, unit: str, better: str = "low", fmt: str = ".2f"):
        super().__init__()
        self.setObjectName("card")
        self.better, self.fmt, self.unit = better, fmt, unit
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(2)
        t = QLabel(title)
        t.setObjectName("cardTitle")
        lay.addWidget(t)
        row = QHBoxLayout()
        row.setSpacing(12)
        self.vals = {}
        for name, color in COLORS.items():
            col = QVBoxLayout()
            col.setSpacing(0)
            tag = QLabel(name)
            tag.setStyleSheet(f"color: {color}; font-size: 8pt; font-weight: 600;")
            v = QLabel("–")
            v.setStyleSheet("font-size: 14pt; font-weight: 600;")
            col.addWidget(tag)
            col.addWidget(v)
            row.addLayout(col)
            self.vals[name] = v
        row.addStretch()
        lay.addLayout(row)

    def set_values(self, ppo, pd):
        vals = {"PPO": ppo, "PD": pd}
        best = None
        if ppo is not None and pd is not None and self.better and abs(ppo - pd) > 1e-12:
            best = ("PPO" if ppo < pd else "PD") if self.better == "low" else ("PPO" if ppo > pd else "PD")
        for name, lab in self.vals.items():
            x = vals[name]
            txt = "–" if x is None else f"{format(x, self.fmt)} {self.unit}".strip()
            lab.setText(txt)
            weight = 700 if name == best else 500
            color = theme.SUCCESS if name == best else theme.TEXT
            lab.setStyleSheet(f"font-size: 14pt; font-weight: {weight}; color: {color};")


# =============================================================================
class CompareWindow(QMainWindow):
    def __init__(self, model_path: str, seed: int = 0, speed: float = 1.0):
        super().__init__()
        self.cmp = Comparison(model_path, seed)
        self.paused = False
        self.speed = speed if speed in SPEEDS else 1
        self._acc = 0.0
        self.rng = np.random.default_rng()
        self.episode_s = self.cmp.runs[0].env.max_episode_steps * self.cmp.dt

        self.setWindowTitle("satSim — PPO vs PD comparison")
        self.resize(1760, 1020)

        root = QWidget()
        rv = QVBoxLayout(root)
        rv.setContentsMargins(12, 10, 12, 12)
        rv.setSpacing(10)
        rv.addWidget(self._build_header())
        rv.addWidget(self._build_toolbar())

        env = self.cmp.runs[0].env
        views = QWidget()
        hv = QHBoxLayout(views)
        hv.setContentsMargins(0, 0, 0, 0)
        hv.setSpacing(10)
        self.views = []
        for r in self.cmp.runs:
            c = card()
            cv = QVBoxLayout(c)
            cv.setContentsMargins(10, 8, 10, 10)
            head = QHBoxLayout()
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {COLORS[r.name]}; font-size: 12pt;")
            name = QLabel(r.name)
            name.setStyleSheet("font-weight: 700; font-size: 11pt;")
            sub = QLabel(f"agent  {model_path}" if r.name == "PPO" else "quaternion PD controller")
            sub.setObjectName("muted")
            for w in (dot, name, sub):
                head.addWidget(w)
            head.addStretch()
            cv.addLayout(head)
            v = AttitudeView(env.engine.params.spacecraft.size, env.engine.rw.A)
            cv.addWidget(v, stretch=1)
            hv.addWidget(c)
            self.views.append(v)

        metrics = QWidget()
        mg = QGridLayout(metrics)
        mg.setContentsMargins(0, 0, 0, 0)
        mg.setSpacing(10)
        self.cards = {
            "err": MetricCard("Attitude error", "°", "low", ".4f"),
            "w": MetricCard("Angular velocity |ω|", "°/s", "low", ".4f"),
            "alpha": MetricCard("Max |α|", "°/s²", "low", ".2f"),
            "rpm": MetricCard("Max wheel speed", "RPM", "low", ".0f"),
            "energy": MetricCard("Energy", "J", "low", ".1f"),
            "ret": MetricCard("Cumulative reward", "", "high", ".1f"),
            "t1": MetricCard("Time to < 1°", "s", "low", ".2f"),
            "t001": MetricCard("Time to < 0.01°", "s", "low", ".2f"),
        }
        for i, c in enumerate(self.cards.values()):
            mg.addWidget(c, i // 4, i % 4)

        left = QSplitter(Qt.Vertical)
        left.addWidget(views)
        left.addWidget(metrics)
        left.setSizes([620, 220])
        left.setChildrenCollapsible(False)

        plots_card = card()
        pv = QVBoxLayout(plots_card)
        pv.setContentsMargins(6, 6, 6, 6)
        self.plots = ComparePlots(env.reward_config["alpha_max_deg"], env.engine.rw.p.max_rpm)
        pv.addWidget(self.plots)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(plots_card)
        split.setSizes([1080, 660])
        split.setChildrenCollapsible(False)
        rv.addWidget(split, stretch=1)
        self.setCentralWidget(root)

        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self.toggle_pause)
        self.clock = QElapsedTimer()
        self.clock.start()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(FRAME_MS)
        self._refresh()

    # --------------------------------------------------------------- header
    def _build_header(self) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(4, 0, 4, 0)
        h.setSpacing(12)
        ic = QLabel()
        ic.setPixmap(theme.icon("mdi6.satellite-variant", theme.ACCENT).pixmap(26, 26))
        title = QLabel("PPO vs PD comparison")
        title.setObjectName("h1")
        self.lbl_info = QLabel()
        self.lbl_info.setObjectName("muted")
        self.chip = QLabel()
        self.chip.setObjectName("chip")
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setFixedWidth(260)
        self.progress.setTextVisible(False)
        self.lbl_time = QLabel()
        self.lbl_time.setObjectName("muted")
        for x in (ic, title, self.lbl_info):
            h.addWidget(x)
        h.addStretch()
        for x in (self.lbl_time, self.progress, self.chip):
            h.addWidget(x)
        return w

    # -------------------------------------------------------------- toolbar
    def _build_toolbar(self) -> QWidget:
        bar = card()
        h = QHBoxLayout(bar)
        h.setContentsMargins(12, 8, 12, 8)
        h.setSpacing(8)

        def button(text, icon_name, slot, variant=None, icon_color=theme.TEXT):
            b = QPushButton(text)
            b.setIcon(theme.icon(icon_name, icon_color))
            b.setIconSize(QSize(16, 16))
            b.setCursor(Qt.PointingHandCursor)
            if variant:
                b.setProperty("variant", variant)
            b.clicked.connect(slot)
            return b

        def sep():
            s = QFrame()
            s.setFixedWidth(1)
            s.setStyleSheet(f"background: {theme.BORDER};")
            s.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
            return s

        lab = QLabel("Seed")
        lab.setObjectName("muted")
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, SEED_MAX)
        self.spin_seed.setValue(self.cmp.seed)
        self.spin_seed.setFixedWidth(120)
        self.spin_seed.setButtonSymbols(QSpinBox.NoButtons)
        h.addWidget(lab)
        h.addWidget(self.spin_seed)
        h.addWidget(button("Start", "mdi6.play", lambda: self._start(self.spin_seed.value()),
                           "accent", "white"))
        h.addWidget(button("Random", "mdi6.dice-5", self._random_seed))
        h.addWidget(sep())

        self.btn_pause = button("Pause", "mdi6.pause", self.toggle_pause)
        self.btn_pause.setToolTip("Pause / resume  [Space]")
        h.addWidget(self.btn_pause)
        h.addWidget(sep())

        sp = QLabel("Speed")
        sp.setObjectName("muted")
        h.addWidget(sp)
        seg = QHBoxLayout()
        seg.setSpacing(0)
        self.speed_group = QButtonGroup(self)
        for i, s in enumerate(SPEEDS):
            b = QPushButton(f"{s:g}×")
            b.setCheckable(True)
            b.setChecked(s == self.speed)
            b.setCursor(Qt.PointingHandCursor)
            radius = ("border-top-left-radius: 8px; border-bottom-left-radius: 8px;" if i == 0 else
                      "border-top-right-radius: 8px; border-bottom-right-radius: 8px;"
                      if i == len(SPEEDS) - 1 else "")
            b.setStyleSheet(f"QPushButton {{ border-radius: 0; padding: 6px 10px; {radius} }}")
            b.clicked.connect(lambda _, v=s: setattr(self, "speed", v))
            self.speed_group.addButton(b)
            seg.addWidget(b)
        h.addLayout(seg)
        h.addWidget(sep())

        imp = button(f"Impulse {IMPULSE_TORQUE * 1e3:g} mN·m × {IMPULSE_DURATION:g} s",
                     "mdi6.flash", lambda: self.cmp.apply_impulse(self.rng), "warning", theme.WARNING)
        imp.setToolTip("Same external torque, random direction, on both satellites")
        h.addWidget(imp)
        h.addStretch()

        legend = QLabel(f"<span style='color:{theme.PPO_COLOR}'>━━</span> PPO &nbsp;&nbsp;"
                        f"<span style='color:{theme.PD_COLOR}'>━━</span> PD")
        legend.setObjectName("muted")
        h.addWidget(legend)
        return bar

    # -------------------------------------------------------------- actions
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
        self.btn_pause.setText("Resume" if self.paused else "Pause")
        self.btn_pause.setIcon(theme.icon("mdi6.play" if self.paused else "mdi6.pause"))

    # ------------------------------------------------------------- loop
    def _tick(self):
        elapsed = self.clock.restart() / 1000.0
        if not self.paused and not self.cmp.done:
            self._acc += elapsed * self.speed
            n = int(self._acc / self.cmp.dt)
            if n > MAX_STEPS_PER_FRAME:          # the CPU cannot keep up: slow down
                n, self._acc = MAX_STEPS_PER_FRAME, 0.0
            else:
                self._acc -= n * self.cmp.dt
            self.cmp.advance(n)
        self._refresh()

    def _refresh(self):
        c = self.cmp
        ppo, pd = c.runs
        for v, r in zip(self.views, c.runs):
            v.update_state(r.tel.q, r.env.engine.env.sun_inertial, r.tel.r_inertial, r.tel.eclipse)
        self.plots.update_curves(c.runs)

        cards = self.cards
        cards["err"].set_values(ppo.tel.att_err_deg, pd.tel.att_err_deg)
        cards["w"].set_values(*(np.degrees(np.linalg.norm(r.tel.omega)) for r in c.runs))
        cards["alpha"].set_values(ppo.alpha_max, pd.alpha_max)
        cards["rpm"].set_values(*(float(np.abs(r.tel.wheel_rpm).max()) for r in c.runs))
        cards["energy"].set_values(ppo.tel.energy, pd.tel.energy)
        cards["ret"].set_values(ppo.ret, pd.ret)
        cards["t1"].set_values(ppo.t_1deg, pd.t_1deg)
        cards["t001"].set_values(ppo.t_001deg, pd.t_001deg)

        self.lbl_info.setText(f"seed {c.seed}   ·   θ0 = {c.theta0:.1f}°   ·   |ω0| = {c.omega0:.2f} °/s")
        self.lbl_time.setText(f"{c.t:6.1f} / {self.episode_s:.0f} s")
        self.progress.setValue(int(1000 * c.t / self.episode_s))
        if c.done:
            winner = "PPO" if ppo.ret > pd.ret else "PD"
            state, text = "done", f"FINISHED · best reward: {winner}"
        elif self.paused:
            state, text = "pause", "PAUSED"
        else:
            state, text = "run", "RUNNING"
        if self.chip.property("state") != state:
            self.chip.setProperty("state", state)
            self.chip.style().unpolish(self.chip)
            self.chip.style().polish(self.chip)
        self.chip.setText(text)
