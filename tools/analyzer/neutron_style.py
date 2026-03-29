#UI style and theme

import sys
from PyQt5.QtGui import QFont

def get_app_font():
    font = QFont()
    if sys.platform == "darwin":
        font = QFont("SF Pro Text", 11)
    elif sys.platform == "win32":
        font = QFont("Segoe UI", 10)
    else:
        font = QFont("Ubuntu", 10)
    font.setLetterSpacing(QFont.PercentageSpacing, 102)
    return font

def get_title_font():
    if sys.platform == "darwin":
        return QFont("SF Pro Display", 20, QFont.Light)
    elif sys.platform == "win32":
        return QFont("Segoe UI", 20, QFont.Light)
    else:
        return QFont("Ubuntu", 20, QFont.Light)

MAIN_STYLESHEET = """
QMainWindow { background-color: #1a1a1a; }
QWidget { background-color: #1a1a1a; }
QLabel { color: #e8e8e8; }
QTextEdit {
    background-color: #0d0d0d;
    color: #d0d0d0;
    border: 1px solid #333;
    border-radius: 4px;
    padding: 12px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 11px;
    line-height: 1.6;
}
QScrollArea {
    border: none;
    background-color: #1a1a1a;
}
QScrollBar:vertical {
    background-color: #1a1a1a;
    width: 12px;
    border: none;
}
QScrollBar::handle:vertical {
    background-color: #3a3a3a;
    border-radius: 6px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background-color: #4a4a4a;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
"""

BUTTON_STYLESHEET = """
QPushButton {
    background-color: #0a84ff;
    color: white;
    border: none;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 500;
}
QPushButton:hover { background-color: #0077ed; }
QPushButton:pressed { background-color: #006edb; }
QPushButton:disabled {
    background-color: #2a2a2a;
    color: #666;
}
"""

SLIDER_STYLESHEET = """
QSlider::groove:horizontal {
    background: #2a2a2a;
    height: 4px;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #0a84ff;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::handle:horizontal:hover {
    background: #0077ed;
}
"""