"""
Telemetry dashboard panels:

    TelemetryPlots : time plots (pyqtgraph)
    TelemetryPanel : instantaneous numerical values + wheel saturation bars
    ModelPanel     : matrix J and the 5 model equations with their current values
"""
import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import (QGridLayout, QGroupBox, QLabel, QPlainTextEdit,
                               QProgressBar, QVBoxLayout, QWidget)

from satsim.quaternion import quat_to_euler_zyx_deg

from . import theme

MONO = theme.mono_font(9)

PEN_XYZ = [pg.mkPen((255, 90, 90), width=1.5), pg.mkPen((90, 230, 90), width=1.5),
           pg.mkPen((100, 150, 255), width=1.5)]
PEN_Q = [pg.mkPen((230, 230, 230), width=1.5)] + PEN_XYZ
PEN_WHEELS = [pg.mkPen(c, width=1.5) for c in
              [(255, 170, 60), (60, 200, 255), (220, 100, 255), (160, 255, 120)]]


# =============================================================================
class TelemetryPlots(pg.GraphicsLayoutWidget):
    def __init__(self, n_wheels: int, max_rpm: float, parent=None):
        super().__init__(parent)
        self.setBackground(theme.SURFACE)
        self.ci.setContentsMargins(6, 6, 10, 6)
        self.plots = []

        def add_plot(title, units, row):
            p = self.addPlot(row=row, col=0)
            theme.style_plot(p, title, units)
            p.addLegend(offset=(5, 5), labelTextSize="8pt", brush=(25, 28, 35, 200))
            if self.plots:
                p.setXLink(self.plots[0])
            self.plots.append(p)
            return p

        p = add_plot("Attitude quaternion q (body → inertial)", "-", 0)
        self.c_q = [p.plot(pen=PEN_Q[i], name=f"q{i}") for i in range(4)]
        p.setYRange(-1.05, 1.05)

        p = add_plot("Attitude error relative to the target", "deg", 1)
        self.c_err = p.plot(pen=pg.mkPen((255, 200, 60), width=2), name="θ_err")

        p = add_plot("Angular velocity ω (body)", "deg/s", 2)
        self.c_w = [p.plot(pen=PEN_XYZ[i], name=f"ω{'xyz'[i]}") for i in range(3)]

        p = add_plot("Reaction wheel speeds", "RPM", 3)
        self.c_rpm = [p.plot(pen=PEN_WHEELS[i], name=f"RW{i + 1}") for i in range(n_wheels)]
        dash = pg.mkPen((255, 60, 60), width=1, style=pg.QtCore.Qt.DashLine)
        p.addItem(pg.InfiniteLine(pos=max_rpm, angle=0, pen=dash))
        p.addItem(pg.InfiniteLine(pos=-max_rpm, angle=0, pen=dash))

        p = add_plot("Wheel electrical power", "W", 4)
        self.c_pow = p.plot(pen=pg.mkPen((255, 120, 200), width=1.5), name="P_tot")
        self.plots[-1].setLabel("bottom", "simulated time [s]", color=theme.MUTED)

    def update_curves(self, h: dict):
        t = h["t"]
        if t.size < 2:
            return
        for i, c in enumerate(self.c_q):
            c.setData(t, h["q"][:, i])
        self.c_err.setData(t, h["att_err"])
        for i, c in enumerate(self.c_w):
            c.setData(t, h["omega"][:, i])
        for i, c in enumerate(self.c_rpm):
            c.setData(t, h["rpm"][:, i])
        self.c_pow.setData(t, h["power"])


# =============================================================================
class TelemetryPanel(QWidget):
    def __init__(self, n_wheels: int, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        def group(title):
            g = QGroupBox(title)
            lab = QLabel()
            lab.setFont(MONO)
            lab.setTextFormat(pg.QtCore.Qt.PlainText)
            QVBoxLayout(g).addWidget(lab)
            lay.addWidget(g)
            return g, lab

        _, self.lab_att = group("Attitude")
        _, self.lab_w = group("Angular velocity")

        g = QGroupBox("Reaction wheels — saturation")
        grid = QGridLayout(g)
        self.bars = []
        for i in range(n_wheels):
            lbl = QLabel(f"RW{i + 1}")
            lbl.setFont(MONO)
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setFont(MONO)
            bar.setTextVisible(True)
            grid.addWidget(lbl, i, 0)
            grid.addWidget(bar, i, 1)
            self.bars.append(bar)
        lay.addWidget(g)

        _, self.lab_pow = group("Energy")
        _, self.lab_dist = group("Disturbance torques")

    def update_values(self, tel):
        rpy = quat_to_euler_zyx_deg(tel.q)
        q, qe = tel.q, tel.q_err
        self.lab_att.setText(
            f"q     = [{q[0]:+.4f} {q[1]:+.4f} {q[2]:+.4f} {q[3]:+.4f}]\n"
            f"q_err = [{qe[0]:+.4f} {qe[1]:+.4f} {qe[2]:+.4f} {qe[3]:+.4f}]\n"
            f"θ_err = {tel.att_err_deg:9.4f} deg\n"
            f"RPY   = [{rpy[0]:+7.2f} {rpy[1]:+7.2f} {rpy[2]:+7.2f}] deg")
        w = np.degrees(tel.omega)
        self.lab_w.setText(
            f"ω   = [{w[0]:+8.4f} {w[1]:+8.4f} {w[2]:+8.4f}] deg/s\n"
            f"|ω| = {np.linalg.norm(w):8.4f} deg/s")
        for i, bar in enumerate(self.bars):
            sat = tel.wheel_saturation[i]
            bar.setValue(int(sat * 1000))
            bar.setFormat(f"{tel.wheel_rpm[i]:+7.0f} RPM  ({sat * 100:5.1f}%)")
            color = theme.SUCCESS if sat < 0.7 else (theme.WARNING if sat < 0.95 else theme.DANGER)
            bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {color}; }}")
        pw = " ".join(f"{p:.3f}" for p in tel.power_wheels)
        self.lab_pow.setText(
            f"P_tot = {tel.power_total:8.3f} W\n"
            f"P_rw  = [{pw}] W\n"
            f"E     = {tel.energy:8.2f} J  ({tel.energy / 3600:.4f} Wh)")
        f = lambda v: f"{np.linalg.norm(v):9.3e}"
        self.lab_dist.setText(
            f"|T_gg|  = {f(tel.T_gg)} N·m\n"
            f"|T_srp| = {f(tel.T_srp)} N·m {'(eclipse)' if tel.eclipse else ''}\n"
            f"|T_mag| = {f(tel.T_mag)} N·m\n"
            f"|T_man| = {f(tel.T_manual)} N·m")


# =============================================================================
def _v(v, fmt="+.3e"):
    return "[" + " ".join(format(x, fmt) for x in v) + "]"


def format_model_text(engine, tel, controller, controller_on: bool) -> str:
    J = engine.J
    rw = engine.rw
    p = rw.p
    A = rw.A
    dist = engine.params.disturbances
    on = lambda b: "ON " if b else "OFF"
    lines = [
        f"t = {tel.t:.2f} s    dt = {engine.dt} s    integrator: fixed-step RK4",
        "",
        "[1] INERTIA MATRIX  J [kg·m²]  (3U CubeSat, m = "
        f"{engine.params.spacecraft.mass} kg, {engine.params.spacecraft.size} m)",
        *[f"      | {J[i, 0]:+.4e} {J[i, 1]:+.4e} {J[i, 2]:+.4e} |" for i in range(3)],
        "",
        f"[2] REACTION WHEELS  ({p.configuration}, N = {rw.n})",
        "      dΩ/dt = T_rw / I_rw         I_rw = "
        f"{rw.I_rw:.2e} kg·m²,  |Ω| ≤ {p.max_rpm:.0f} RPM,  |T_rw| ≤ {p.max_torque * 1e3:.1f} mN·m",
        "      P     = k1|T_rw| + k2|T_rw·Ω| + P_s   "
        f"(k1={p.k1}, k2={p.k2}, P_s={p.p_static} W)",
        "      A (wheel axes in body, columns):",
        *[f"        {_v(A[i], '+.3f')}" for i in range(3)],
        f"      T_rw  = {_v(tel.wheel_torque)} N·m",
        f"      Ω     = {_v(tel.wheel_speed, '+9.2f')} rad/s",
        f"      dΩ/dt = {_v(tel.wheel_accel, '+9.2f')} rad/s²",
        f"      P     = {_v(tel.power_wheels, '.4f')} W   Σ = {tel.power_total:.4f} W",
        "",
        "[3] EULER EQUATIONS (gyroscopic coupling)",
        "      J·dω/dt = T_tot − ω × (J·ω + h_rw),     h_rw = A·I_rw·Ω",
        f"      ω       = {_v(tel.omega)} rad/s",
        f"      h_rw    = {_v(tel.h_rw)} N·m·s",
        f"      dω/dt   = {_v(tel.omega_dot)} rad/s²",
        f"      H_I     = {_v(tel.H_inertial)} N·m·s  (total momentum, inertial)",
        "",
        "[4] QUATERNION KINEMATICS",
        "      dq/dt = ½ · q ⊗ [0, ωx, ωy, ωz]ᵀ ,   q ← q/‖q‖ at every step",
        f"      q     = {_v(tel.q, '+.6f')}",
        f"      dq/dt = {_v(tel.q_dot)}",
        "",
        "[5] TOTAL TORQUES  T_tot = T_rw_action + T_gg + T_srp + T_mag (+ T_man)",
        f"      T_rw_ac = −A·T_rw        = {_v(tel.T_rw_body)}",
        f"  {on(dist.enable_gravity_gradient)} T_gg = 3μ/r³ · r̂_B × (J·r̂_B) = {_v(tel.T_gg)}",
        f"  {on(dist.enable_srp)} T_srp = Σ r_cp × F_face   = {_v(tel.T_srp)}"
        + ("  [ECLIPSE]" if tel.eclipse else ""),
        f"  {on(dist.enable_magnetic)} T_mag = m_res × B_B       = {_v(tel.T_mag)}",
        f"      T_man                    = {_v(tel.T_manual)}",
        f"      T_tot                    = {_v(tel.T_total)}   [N·m]",
        "",
        f"PD CONTROLLER {'ACTIVE' if controller_on else 'DISABLED'}",
        "      T_cmd = −J·(Kp·q_e,vec + Kd·ω) + ω×(J·ω + h_rw)   T_rw = −A⁺·T_cmd",
        f"      Kp = 2ωn² = {controller.Kp:.4f}   Kd = 2ζωn = {controller.Kd:.4f}"
        f"   (ωn = {controller.omega_n} rad/s, ζ = {controller.zeta})",
    ]
    return "\n".join(lines)


class ModelPanel(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(MONO)
        self.setLineWrapMode(QPlainTextEdit.NoWrap)

    def update_text(self, text: str):
        sb = self.verticalScrollBar()
        pos = sb.value()
        self.setPlainText(text)
        sb.setValue(pos)
