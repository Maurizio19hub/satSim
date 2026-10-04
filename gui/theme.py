"""
Graphical theme shared by the GUI windows (simulation and comparison).

A single place for colours, the Qt style sheet (QSS), icons and the style of
the pyqtgraph plots: the windows only use apply_theme(), icon() and
style_plot().
"""
import pyqtgraph as pg
from PySide6.QtGui import QColor, QFont, QIcon, QPalette

# --- Palette ----------------------------------------------------------------
BG = "#111318"          # window background
SURFACE = "#191c23"     # cards / panels
SURFACE_2 = "#222632"   # controls (buttons, fields)
SURFACE_3 = "#2b3040"   # hover
BORDER = "#2c3140"
TEXT = "#e6e8ee"
MUTED = "#8b93a7"
ACCENT = "#4f8cff"
SUCCESS = "#3ccf91"
WARNING = "#f5b942"
DANGER = "#ff5c5c"
PPO_COLOR = "#50aaff"
PD_COLOR = "#ffa03c"

FONT_FAMILY = "Inter, Segoe UI, SF Pro Text, Helvetica Neue, Ubuntu, Noto Sans, sans-serif"

QSS = f"""
QMainWindow, QWidget {{ background: {BG}; color: {TEXT};
    font-family: {FONT_FAMILY}; font-size: 10pt; }}
QToolTip {{ background: {SURFACE_2}; color: {TEXT}; border: 1px solid {BORDER};
    border-radius: 6px; padding: 4px 8px; }}

/* Card */
QFrame#card, QGroupBox {{ background: {SURFACE}; border: 1px solid {BORDER};
    border-radius: 10px; }}
QGroupBox {{ margin-top: 14px; padding: 14px 10px 10px 10px; font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 4px;
    color: {MUTED}; }}
QFrame#card QLabel, QGroupBox QLabel {{ background: transparent; }}
QLabel#cardTitle {{ color: {MUTED}; font-size: 9pt; font-weight: 600; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#h1 {{ font-size: 14pt; font-weight: 700; }}

/* Buttons */
QPushButton {{ background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 8px;
    padding: 6px 14px; color: {TEXT}; }}
QPushButton:hover {{ background: {SURFACE_3}; }}
QPushButton:pressed {{ background: {BORDER}; }}
QPushButton[variant="accent"] {{ background: {ACCENT}; border-color: {ACCENT}; color: white;
    font-weight: 600; }}
QPushButton[variant="accent"]:hover {{ background: #6a9dff; }}
QPushButton[variant="warning"] {{ border-color: {WARNING}; color: {WARNING}; }}
QPushButton[segment="true"] {{ border-radius: 0; padding: 6px 10px; margin: 0; }}
QPushButton:checked {{ background: {ACCENT}; border-color: {ACCENT}; color: white; }}

/* Fields */
QSpinBox, QComboBox {{ background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 8px;
    padding: 5px 8px; min-height: 18px; }}
QSpinBox:focus, QComboBox:focus {{ border-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{ background: {SURFACE_2}; border: 1px solid {BORDER};
    selection-background-color: {ACCENT}; }}
QCheckBox {{ spacing: 6px; background: transparent; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px;
    border: 1px solid {BORDER}; background: {SURFACE_2}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QSlider::groove:horizontal {{ height: 4px; background: {SURFACE_3}; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 14px; margin: -6px 0; border-radius: 7px; background: {ACCENT}; }}

/* Bars */
QProgressBar {{ background: {SURFACE_2}; border: none; border-radius: 5px; height: 10px;
    text-align: center; color: {TEXT}; font-size: 8pt; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 5px; }}

/* Status chips */
QLabel#chip {{ border-radius: 10px; padding: 3px 10px; font-weight: 600; font-size: 9pt; }}
QLabel#chip[state="run"] {{ background: rgba(60,207,145,0.15); color: {SUCCESS}; }}
QLabel#chip[state="pause"] {{ background: rgba(245,185,66,0.15); color: {WARNING}; }}
QLabel#chip[state="done"] {{ background: rgba(79,140,255,0.18); color: {ACCENT}; }}

/* Miscellaneous */
QSplitter::handle {{ background: {BG}; }}
QSplitter::handle:horizontal {{ width: 6px; }}
QSplitter::handle:vertical {{ height: 6px; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; }}
QScrollBar::handle:vertical {{ background: {SURFACE_3}; border-radius: 5px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QStatusBar {{ background: {BG}; color: {MUTED}; }}
QPlainTextEdit {{ background: {SURFACE}; border: none; color: {TEXT}; }}
"""


def apply_theme(app):
    """Applies the Fusion style, dark palette and style sheet to the application."""
    app.setStyle("Fusion")
    pal = QPalette()
    for role, c in ((QPalette.Window, BG), (QPalette.WindowText, TEXT), (QPalette.Base, SURFACE),
                    (QPalette.AlternateBase, SURFACE_2), (QPalette.Text, TEXT),
                    (QPalette.Button, SURFACE_2), (QPalette.ButtonText, TEXT),
                    (QPalette.Highlight, ACCENT), (QPalette.HighlightedText, "#ffffff"),
                    (QPalette.ToolTipBase, SURFACE_2), (QPalette.ToolTipText, TEXT)):
        pal.setColor(role, QColor(c))
    app.setPalette(pal)
    app.setStyleSheet(QSS)
    pg.setConfigOptions(antialias=True, background=SURFACE, foreground=MUTED)


def icon(name: str, color: str = TEXT) -> QIcon:
    """Vector icon (QtAwesome, Material Design set). Empty icon if the
    library is not installed: the GUI stays usable."""
    try:
        import qtawesome as qta
        return qta.icon(name, color=color)
    except Exception:
        return QIcon()


def style_plot(p: pg.PlotItem, title: str, units: str = ""):
    """Uniform plot style: subtle title, axes and grid."""
    p.setTitle(title, color=TEXT, size="10pt")
    p.showGrid(x=True, y=True, alpha=0.12)
    for ax in ("left", "bottom"):
        a = p.getAxis(ax)
        a.setPen(pg.mkPen(BORDER))
        a.setTextPen(pg.mkPen(MUTED))
    if units:
        p.setLabel("left", units, color=MUTED)


def mono_font(size: int = 9) -> QFont:
    f = QFont()
    f.setFamilies(["JetBrains Mono", "DejaVu Sans Mono", "Consolas", "Menlo", "monospace"])
    f.setPointSize(size)
    f.setStyleHint(QFont.Monospace)
    return f
