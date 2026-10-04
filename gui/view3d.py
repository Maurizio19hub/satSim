"""
OpenGL 3D view: satellite, fixed inertial/target frame and body-fixed frame.

Graphical scale: 1 unit = 1 dm (the 3U CubeSat appears as a 1 x 1 x 3 box).
"""
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PySide6.QtGui import QColor

from satsim.quaternion import quat_to_dcm

SCALE = 10.0   # metres → graphical units

AXIS_COLORS_BODY = [(1.0, 0.25, 0.25, 1.0), (0.3, 1.0, 0.3, 1.0), (0.35, 0.55, 1.0, 1.0)]
AXIS_COLORS_INERTIAL = [(0.6, 0.2, 0.2, 0.9), (0.2, 0.6, 0.2, 0.9), (0.2, 0.3, 0.7, 0.9)]


def box_meshdata(a: float, b: float, c: float) -> gl.MeshData:
    """Box centred at the origin with per-face colours (+Z gold)."""
    x, y, z = a / 2, b / 2, c / 2
    v = np.array([[-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],
                  [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]])
    faces = np.array([
        [0, 2, 1], [0, 3, 2],   # -Z
        [4, 5, 6], [4, 6, 7],   # +Z
        [0, 1, 5], [0, 5, 4],   # -Y
        [2, 3, 7], [2, 7, 6],   # +Y
        [1, 2, 6], [1, 6, 5],   # +X
        [0, 4, 7], [0, 7, 3],   # -X
    ])
    panel = (0.20, 0.28, 0.55, 1.0)     # side solar panels
    colors = np.array([
        (0.45, 0.45, 0.50, 1.0), (0.45, 0.45, 0.50, 1.0),   # -Z
        (0.95, 0.75, 0.20, 1.0), (0.95, 0.75, 0.20, 1.0),   # +Z (payload face)
        panel, panel, panel, panel,
        (0.30, 0.20, 0.50, 1.0), (0.30, 0.20, 0.50, 1.0),   # +X (marker)
        panel, panel,
    ])
    return gl.MeshData(vertexes=v, faces=faces, faceColors=colors)


def rotation_transform(R: np.ndarray) -> pg.Transform3D:
    return pg.Transform3D(R[0, 0], R[0, 1], R[0, 2], 0,
                          R[1, 0], R[1, 1], R[1, 2], 0,
                          R[2, 0], R[2, 1], R[2, 2], 0,
                          0, 0, 0, 1)


class AttitudeView(gl.GLViewWidget):
    def __init__(self, size_m, wheel_axes: np.ndarray, parent=None):
        super().__init__(parent)
        self.setBackgroundColor((16, 18, 24))
        self.setCameraPosition(distance=10, elevation=20, azimuth=40)
        self.setMinimumSize(420, 360)

        grid = gl.GLGridItem()
        grid.setSize(16, 16)
        grid.setSpacing(1, 1)
        grid.setColor((90, 90, 110, 70))
        grid.translate(0, 0, -4.5)
        self.addItem(grid)

        # --- Inertial / target frame (fixed) ------------------------------
        L_i = 4.2
        for i, lab in enumerate(("X_I", "Y_I", "Z_I")):
            e = np.zeros(3)
            e[i] = 1
            self.addItem(gl.GLLinePlotItem(pos=np.array([-e * 0.3 * L_i, e * L_i]),
                                           color=AXIS_COLORS_INERTIAL[i], width=2, antialias=True))
            self.addItem(gl.GLTextItem(pos=e * (L_i + 0.3), text=lab,
                                       color=QColor.fromRgbF(*AXIS_COLORS_INERTIAL[i])))

        # --- Satellite (mesh) + body frame + wheel axes as children ------
        a, b, c = (s * SCALE for s in size_m)
        self.body = gl.GLMeshItem(meshdata=box_meshdata(a, b, c), smooth=False,
                                  shader="shaded", drawEdges=True,
                                  edgeColor=(0.9, 0.9, 0.95, 0.6), glOptions="opaque")
        self.addItem(self.body)

        L_b = 2.6
        for i, lab in enumerate(("x_B", "y_B", "z_B")):
            e = np.zeros(3)
            e[i] = 1
            line = gl.GLLinePlotItem(pos=np.array([np.zeros(3), e * L_b]),
                                     color=AXIS_COLORS_BODY[i], width=4, antialias=True,
                                     glOptions="additive")
            line.setParentItem(self.body)
            txt = gl.GLTextItem(pos=e * (L_b + 0.25), text=lab,
                                color=QColor.fromRgbF(*AXIS_COLORS_BODY[i]))
            txt.setParentItem(self.body)

        # Wheel spin axes (visible through the box)
        for i in range(wheel_axes.shape[1]):
            ax = wheel_axes[:, i] * 1.2
            w = gl.GLLinePlotItem(pos=np.array([np.zeros(3), ax]), color=(0.9, 0.9, 0.9, 0.8),
                                  width=2, glOptions="additive")
            w.setParentItem(self.body)
            t = gl.GLTextItem(pos=ax * 1.1, text=f"RW{i + 1}", color=QColor(200, 200, 200))
            t.setParentItem(self.body)

        # --- Environmental directions (inertial): Sun and nadir ---------
        self.sun_line = gl.GLLinePlotItem(color=(1.0, 0.9, 0.2, 0.9), width=2, antialias=True)
        self.nadir_line = gl.GLLinePlotItem(color=(0.2, 0.9, 0.9, 0.9), width=2, antialias=True)
        self.sun_label = gl.GLTextItem(text="Sun", color=QColor(255, 230, 60))
        self.nadir_label = gl.GLTextItem(text="Nadir", color=QColor(60, 230, 230))
        for it in (self.sun_line, self.nadir_line, self.sun_label, self.nadir_label):
            self.addItem(it)

    def update_state(self, q: np.ndarray, sun_I: np.ndarray, r_I: np.ndarray, eclipse: bool):
        self.body.setTransform(rotation_transform(quat_to_dcm(q)))

        L = 5.5
        s = sun_I * L
        self.sun_line.setData(pos=np.array([np.zeros(3), s]),
                              color=(0.5, 0.45, 0.1, 0.6) if eclipse else (1.0, 0.9, 0.2, 0.9))
        self.sun_label.setData(pos=s * 1.05, text="Sun (eclipse)" if eclipse else "Sun")
        n = -r_I / np.linalg.norm(r_I) * L
        self.nadir_line.setData(pos=np.array([np.zeros(3), n]))
        self.nadir_label.setData(pos=n * 1.05)
