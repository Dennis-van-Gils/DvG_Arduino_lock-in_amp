#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Arduino lock-in amplifier
"""
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/DvG_Arduino_lock-in_amp"
__date__ = "03-02-2022"
__version__ = "1.0.0"
# pylint: disable=invalid-name

import os
import time as Time

from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpacerItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtCore import QDateTime
import pyqtgraph as pg
import numpy as np
import psutil

from dvg_pyqtgraph_threadsafe import HistoryChartCurve, PlotCurve, LegendSelect

# from dvg_debug_functions import dprint

from Alia_protocol_serial import Alia, Waveform
from Alia_qdev import Alia_qdev
from dvg_ringbuffer_fir_filter_GUI import Filter_design_GUI
import dvg_monkeypatch_pyqtgraph  # pylint: disable=unused-import

# Constants
UPDATE_INTERVAL_WALL_CLOCK = 50  # 50 [ms]

# Show debug info in terminal? Warning: Slow! Do not leave on unintentionally.
DEBUG_TIMING = False

# OpenGL tested succesful in Windows & Linux, untested in MacOS
TRY_USING_OPENGL = True  # if (os.name == "nt" or os.name == "posix") else False

if TRY_USING_OPENGL:
    try:
        import OpenGL.GL as gl  # pylint: disable=unused-import

        pg.setConfigOptions(useOpenGL=True)
        pg.setConfigOptions(enableExperimental=True)
        pg.setConfigOptions(antialias=True)
        print("OpenGL hardware acceleration enabled.")
        USING_OPENGL = True
    except:  # pylint: disable=bare-except
        pg.setConfigOptions(useOpenGL=False)
        pg.setConfigOptions(enableExperimental=False)
        pg.setConfigOptions(antialias=False)
        print("WARNING: Could not enable OpenGL hardware acceleration.")
        print("Check if prerequisite 'PyOpenGL' library is installed.")
        USING_OPENGL = False
else:
    print("OpenGL hardware acceleration is disabled.")
    USING_OPENGL = False

# Default settings for graphs
COLOR_GRAPH_BG = QtGui.QColor(0, 20, 20)
COLOR_GRAPH_FG = QtGui.QColor(240, 240, 240)
pg.setConfigOption("background", COLOR_GRAPH_BG)
pg.setConfigOption("foreground", COLOR_GRAPH_FG)

# Stylesheets
# Note: (Top Right Bottom Left)
# fmt: off
COLOR_BG           = "rgb(240, 240, 240)"
COLOR_LED_GREEN    = "rgb(0  , 238, 118)"
COLOR_LED_RED      = "rgb(205, 92 , 92 )"
COLOR_GROUP_BG     = "rgb(252, 208, 173)"
COLOR_READ_ONLY    = "rgb(250, 230, 210)"
COLOR_TAB_ACTIVE   = "rgb(207, 225, 225)"
COLOR_TAB          = "rgb(234, 235, 233)"
COLOR_HOVER        = "rgb(229, 241, 251)"
COLOR_HOVER_BORDER = "rgb(0  , 120, 215)"

SS_HOVER = (
        "QLineEdit:hover {"
            "background: " + COLOR_HOVER + ";"
            "border: 1px solid " + COLOR_HOVER_BORDER + ";}"
        "QCheckBox:hover {"
            "background: " + COLOR_HOVER + ";"
            "border: 0px solid " + COLOR_HOVER_BORDER + ";}"
        "QRadioButton:hover {"
            "background: " + COLOR_HOVER + ";"
            "border: 0px solid " + COLOR_HOVER_BORDER + ";}")

SS_TEXTBOX_READ_ONLY = (
        "QLineEdit {"
            "border: 1px solid black}"
        "QLineEdit::read-only {"
            "border: 1px solid gray;"
            "background: " + COLOR_READ_ONLY + "}"
        "QPlainTextEdit {"
            "border: 1px solid black}"
        "QPlainTextEdit[readOnly=\"true\"] {"
            "border: 1px solid gray;"
            "background-color: " + COLOR_READ_ONLY + "}")

SS_TABS = (
        "QTabWidget::pane {"
            "border: 0px solid gray;}"
        "QTabBar::tab:selected {"
            "background: " + COLOR_TAB_ACTIVE + "; "
            "border-bottom-color: " + COLOR_TAB_ACTIVE + ";}"
        "QTabWidget>QWidget>QWidget {"
            "border: 2px solid gray;"
            "background: " + COLOR_TAB_ACTIVE + ";} "
        "QTabBar::tab {"
            "background: " + COLOR_TAB + ";"
            "border: 2px solid gray;"
            "border-bottom-color: " + COLOR_TAB + ";"
            "border-top-left-radius: 4px;"
            "border-top-right-radius: 4px;"
            "min-width: 119px;"
            "padding: 6px;} "
        "QTabBar::tab:hover {"
            "background: " + COLOR_HOVER + ";"
            "border: 2px solid " + COLOR_HOVER_BORDER + ";"
            "border-bottom-color: " + COLOR_HOVER + ";"
            "border-top-left-radius: 4px;"
            "border-top-right-radius: 4px;"
            "padding: 6px;} "
        "QTabWidget::tab-bar {"
            "left: 0px;}")

SS_GROUP = (
        "QGroupBox {"
            "background-color: " + COLOR_GROUP_BG + ";"
            "border: 2px solid gray;"
            "border-radius: 0 px;"
            "font: bold;"
            "margin: 0 0 0 0 px;"
            "padding: 14 0 0 0 px;}"
        "QGroupBox::title {"
            "subcontrol-origin: margin;"
            "subcontrol-position: top left;"
            "margin: 0 0 0 0 px;"
            "padding: 0 0 0 0 px;"
            "top: 4 px;"
            "left: 4 px;}"
        "QGroupBox::flat {"
            "border: 0 px;"
            "border-radius: 0 0 px;"
            "padding: 0 0 0 0 px;}")

SS_TOGGLE_BUTTON = (
        "QPushButton {"
            "background-color: " + COLOR_BG + ";"
            "color: black;" +
            "border-style: outset;"
            "border-width: 2px;"
            "border-radius: 4px;"
            "border-color: gray;"
            "text-align: center;"
            "padding: 1px 1px 1px 1px;}"
        "QPushButton:hover {"
            "background: " + COLOR_HOVER + ";"
            "border-color: " + COLOR_HOVER_BORDER + ";}"
        "QPushButton:disabled {"
            "color: grey;}"
        "QPushButton:checked {"
            "background-color: " + COLOR_LED_GREEN + ";"
            "color: black;" +
            "font-weight: normal;"
            "border-style: inset;}")

SS_LED_RECT = (
        "QPushButton {"
            "background-color: " + COLOR_LED_RED + ";"
            "border: 1px solid grey;"
            "min-height: 30px;"
            "min-width: 60px;"
            "max-width: 60px}"
        "QPushButton::disabled {"
            "border-radius: 10px;"
            "color: black}"
        "QPushButton::checked {"
            "background-color: " + COLOR_LED_GREEN + "}")
# fmt: on


def create_Toggle_button(text="", minimumHeight=40):
    button = QPushButton(text, checkable=True)
    button.setStyleSheet(SS_TOGGLE_BUTTON)
    if minimumHeight is not None:
        button.setMinimumHeight(minimumHeight)
    return button


def create_LED_indicator_rect(initial_state=False, text=""):
    button = QPushButton(text, checkable=True, enabled=False)
    button.setStyleSheet(SS_LED_RECT)
    button.setChecked(initial_state)
    return button


class CustomAxis(pg.AxisItem):
    @property
    def nudge(self):
        if not hasattr(self, "_nudge"):
            self._nudge = 5
        return self._nudge

    @nudge.setter
    def nudge(self, nudge):
        self._nudge = nudge
        s = self.size()
        # call resizeEvent indirectly
        self.resize(s + QtCore.QSizeF(1, 1))
        self.resize(s)

    def resizeEvent(self, ev=None):
        # s = self.size()

        ## Set the position of the label
        nudge = self.nudge
        br = self.label.boundingRect()
        p = QtCore.QPointF(0, 0)
        if self.orientation == "left":
            p.setY(int(self.size().height() / 2 + br.width() / 2))
            p.setX(-nudge)
        elif self.orientation == "right":
            p.setY(int(self.size().height() / 2 + br.width() / 2))
            p.setX(int(self.size().width() - br.height() + nudge))
        elif self.orientation == "top":
            p.setY(-nudge)
            p.setX(int(self.size().width() / 2.0 - br.width() / 2.0))
        elif self.orientation == "bottom":
            p.setX(int(self.size().width() / 2.0 - br.width() / 2.0))
            p.setY(int(self.size().height() - br.height() + nudge))
        self.label.setPos(p)
        self.picture = None


def apply_PlotItem_style(pi, title="", bottom="", left="", right=""):
    # Note: We are not using 'title' but use label 'top' instead

    p_title = {
        "color": COLOR_GRAPH_FG.name(),
        "font-size": "12pt",
        "font-family": "Helvetica",
        "font-weight": "bold",
    }
    p_label = {
        "color": COLOR_GRAPH_FG.name(),
        "font-size": "12pt",
        "font-family": "Helvetica",
    }
    pi.setLabel("bottom", bottom, **p_label)
    pi.setLabel("left", left, **p_label)
    pi.setLabel("top", title, **p_title)
    pi.setLabel("right", right, **p_label)

    pi.getAxis("bottom").nudge -= 8
    pi.getAxis("left").nudge -= 4
    pi.getAxis("top").nudge -= 6

    pi.showGrid(x=1, y=1)

    font = QtGui.QFont()
    font.setPixelSize(16)
    # fmt: off
    pi.getAxis("bottom").setTickFont(font)
    pi.getAxis("left")  .setTickFont(font)
    pi.getAxis("top")   .setTickFont(font)
    pi.getAxis("right") .setTickFont(font)

    pi.getAxis("bottom").setStyle(tickTextOffset=10)
    pi.getAxis("left")  .setStyle(tickTextOffset=10)

    pi.getAxis("bottom").setHeight(60)
    pi.getAxis("left")  .setWidth(90)
    pi.getAxis("top")   .setHeight(40)
    pi.getAxis("right") .setWidth(16)

    pi.getAxis("top")  .setStyle(showValues=False)
    pi.getAxis("right").setStyle(showValues=False)
    # fmt: on


# Fonts
FONT_MONOSPACE = QtGui.QFont("Courier")
FONT_MONOSPACE.setFamily("Monospace")
FONT_MONOSPACE.setStyleHint(QtGui.QFont.Monospace)

# ------------------------------------------------------------------------------
#   MainWindow
# ------------------------------------------------------------------------------


class MainWindow(QWidget):
    def __init__(self, alia: Alia, alia_qdev: Alia_qdev, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        if DEBUG_TIMING:
            self.tick = Time.perf_counter()

        self.alia = alia
        self.alia_qdev = alia_qdev
        c = alia.config  # Short-hand

        self.proc = psutil.Process(os.getpid())
        self.cpu_count = psutil.cpu_count()
        self.prev_time_CPU_load = QDateTime.currentDateTime()

        # Collect all upcoming graphs and curves into a list
        self.all_graphs = list()
        self.all_curves = list()

        self.setGeometry(250, 50, 1200, 960)
        self.setWindowTitle("Arduino lock-in amplifier")
        self.setStyleSheet(SS_TEXTBOX_READ_ONLY + SS_GROUP + SS_HOVER + SS_TABS)

        # Define styles for plotting curves
        # fmt: off
        self.PEN_01 = pg.mkPen(color=[255, 30 , 180], width=3)
        self.PEN_02 = pg.mkPen(color=[255, 255, 90 ], width=3)
        self.PEN_03 = pg.mkPen(color=[0  , 255, 255], width=3)
        self.PEN_04 = pg.mkPen(color=[255, 255, 255], width=3)
        self.PEN_05 = pg.mkPen(color=[0  , 255, 0  ], width=3)
        self.BRUSH_03 = pg.mkBrush(0, 255, 255, 64)
        # fmt: on

        # Column width left of timeseries graphs
        LEFT_COLUMN_WIDTH = 134

        # Textbox widths for fitting N characters using the current font
        ex8 = 8 + 8 * QtGui.QFontMetrics(QtGui.QFont()).averageCharWidth()
        ex10 = 8 + 10 * QtGui.QFontMetrics(QtGui.QFont()).averageCharWidth()

        def Header():  # pylint: disable=unused-variable
            pass  # IDE bookmark

        # -----------------------------------
        # -----------------------------------
        #   FRAME: Header
        # -----------------------------------
        # -----------------------------------

        # Left box
        self.qlbl_update_counter = QLabel("0")
        self.qlbl_sample_rate = QLabel("Sample rate: {:,.0f} Hz".format(c.Fs))
        self.qlbl_CPU_syst = QLabel("CPU system : nan%")
        self.qlbl_CPU_proc = QLabel("CPU process: nan%")
        self.qlbl_DAQ_rate = QLabel("Blocks/s: nan")
        self.qlbl_DAQ_rate.setMinimumWidth(100)

        vbox_left = QVBoxLayout()
        vbox_left.addWidget(self.qlbl_sample_rate)
        vbox_left.addWidget(self.qlbl_CPU_syst)
        vbox_left.addWidget(self.qlbl_CPU_proc)
        # vbox_left.addStretch(1)
        vbox_left.addWidget(self.qlbl_DAQ_rate)
        vbox_left.addWidget(self.qlbl_update_counter)

        # Middle box
        self.qlbl_title = QLabel(
            "Arduino lock-in amplifier",
            font=QtGui.QFont("Palatino", 14, weight=QtGui.QFont.Bold),
        )
        self.qlbl_title.setAlignment(QtCore.Qt.AlignCenter)
        self.qlbl_cur_date_time = QLabel("00-00-0000    00:00:00")
        self.qlbl_cur_date_time.setAlignment(QtCore.Qt.AlignCenter)
        self.qpbt_record = create_Toggle_button(
            "Click to start recording to file", minimumHeight=40
        )

        vbox_middle = QVBoxLayout()
        vbox_middle.addWidget(self.qlbl_title)
        vbox_middle.addWidget(self.qlbl_cur_date_time)
        vbox_middle.addWidget(self.qpbt_record)

        # Right box
        self.qpbt_exit = QPushButton("Exit")
        self.qpbt_exit.clicked.connect(self.close)
        self.qpbt_exit.setMinimumHeight(30)

        p = {"alignment": QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter}
        vbox_right = QVBoxLayout(spacing=4)
        vbox_right.addWidget(self.qpbt_exit)
        vbox_right.addStretch(1)
        self.qlbl_GitHub = QLabel(
            '<a href="%s">GitHub source</a>' % __url__, **p
        )
        self.qlbl_GitHub.setTextFormat(QtCore.Qt.RichText)
        self.qlbl_GitHub.setTextInteractionFlags(
            QtCore.Qt.TextBrowserInteraction
        )
        self.qlbl_GitHub.setOpenExternalLinks(True)
        vbox_right.addWidget(self.qlbl_GitHub)
        vbox_right.addWidget(QLabel(__author__, **p))
        vbox_right.addWidget(QLabel("v%s" % __version__, **p))

        # Round up frame
        hbox_header = QHBoxLayout()
        hbox_header.addLayout(vbox_left)
        hbox_header.addStretch(1)
        hbox_header.addLayout(vbox_middle)
        hbox_header.addStretch(1)
        hbox_header.addLayout(vbox_right)

        def Tabs():  # pylint: disable=unused-variable
            pass  # IDE bookmark

        # -----------------------------------
        # -----------------------------------
        #   FRAME: Tabs
        # -----------------------------------
        # -----------------------------------

        self.tabs = QTabWidget()
        self.tab_main = QWidget()
        self.tab_mixer = QWidget()
        self.tab_power_spectrum = QWidget()
        self.tab_filter_1_design = QWidget()
        self.tab_filter_2_design = QWidget()
        self.tab_diagram = QWidget()
        self.tab_settings = QWidget()

        # fmt: off
        self.tabs.addTab(self.tab_main           , "Main")
        self.tabs.addTab(self.tab_mixer          , "Mixer")
        self.tabs.addTab(self.tab_power_spectrum , "Spectrum")
        self.tabs.addTab(self.tab_filter_1_design, "Filter design @ sig_I")
        self.tabs.addTab(self.tab_filter_2_design, "Filter design @ mix_X/Y")
        self.tabs.addTab(self.tab_diagram        , "Diagram")
        self.tabs.addTab(self.tab_settings       , "Settings")
        # fmt: on

        def Sidebar():  # pylint: disable=unused-variable
            pass  # IDE bookmark

        # -----------------------------------
        # -----------------------------------
        #   FRAME: Sidebar
        # -----------------------------------
        # -----------------------------------

        # On/off
        self.qpbt_ENA_lockin = create_Toggle_button("Lock-in OFF")
        self.qpbt_ENA_lockin.clicked.connect(self.process_qpbt_ENA_lockin)

        # QGROUP: Reference signal
        self.qcbx_ref_waveform = QComboBox()
        for waveform in Waveform:
            if waveform.value > -1:
                self.qcbx_ref_waveform.addItem(waveform.name, waveform.value)
        self.qcbx_ref_waveform.setCurrentIndex(c.ref_waveform.value)

        p1 = {"maximumWidth": ex8, "minimumWidth": ex8}
        p2 = {"maximumWidth": ex10, "minimumWidth": ex10}
        self.qlin_ref_freq = QLineEdit("%.3f" % c.ref_freq, **p2)
        self.qlin_ref_V_offset = QLineEdit("%.3f" % c.ref_V_offset, **p1)
        self.qlin_ref_V_ampl_RMS = QLineEdit("%.3f" % c.ref_V_ampl_RMS, **p1)
        self.qlin_ref_V_ampl = QLineEdit(
            "%.3f" % c.ref_V_ampl, readOnly=True, **p1
        )
        self.qlbl_ref_is_clipping = QLabel(self.get_clipping_text())

        self.qlin_ref_freq.setAlignment(QtCore.Qt.AlignRight)
        self.qlin_ref_V_offset.setAlignment(QtCore.Qt.AlignRight)
        self.qlin_ref_V_ampl_RMS.setAlignment(QtCore.Qt.AlignRight)
        self.qlin_ref_V_ampl.setAlignment(QtCore.Qt.AlignRight)
        self.qlbl_ref_is_clipping.setAlignment(QtCore.Qt.AlignCenter)

        self.qcbx_ref_waveform.currentIndexChanged.connect(
            self.process_qcbx_ref_waveform
        )
        self.qlin_ref_freq.editingFinished.connect(self.process_qlin_ref_freq)
        self.qlin_ref_V_offset.editingFinished.connect(
            self.process_qlin_ref_V_offset
        )
        self.qlin_ref_V_ampl_RMS.editingFinished.connect(
            self.process_qlin_ref_V_ampl_RMS
        )

        # fmt: off
        p  = {"alignment": QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight}
        p2 = {"alignment": QtCore.Qt.AlignVCenter | QtCore.Qt.AlignHCenter}
        i = 0
        grid = QGridLayout(spacing=4)
        grid.addWidget(QLabel("waveform:", **p) , i, 0)
        grid.addWidget(self.qcbx_ref_waveform   , i, 1, 1, 2); i+=1
        grid.addWidget(QLabel("freq:", **p)     , i, 0)
        grid.addWidget(self.qlin_ref_freq       , i, 1)
        grid.addWidget(QLabel("Hz")             , i, 2); i+=1
        grid.addWidget(QLabel("offset:", **p)   , i, 0)
        grid.addWidget(self.qlin_ref_V_offset   , i, 1)
        grid.addWidget(QLabel("V")              , i, 2); i+=1
        grid.addWidget(QLabel("ampl:", **p)     , i, 0)
        grid.addWidget(self.qlin_ref_V_ampl_RMS , i, 1)
        grid.addWidget(QLabel("V<sub><b>RMS</b></sub>"), i, 2); i+=1
        grid.addWidget(self.qlin_ref_V_ampl     , i, 1)
        grid.addWidget(QLabel("V")              , i, 2); i+=1
        grid.addWidget(QLabel("clipping?", **p) , i, 0)
        grid.addWidget(self.qlbl_ref_is_clipping, i, 1, 1, 2)
        # fmt: on

        qgrp_refsig = QGroupBox("Reference signal: ref_X*")
        qgrp_refsig.setLayout(grid)

        # QGROUP: Connections
        grid = QGridLayout(spacing=4)
        grid.addWidget(
            QLabel(
                "Output: ref_X*\n"
                "  [ 0.0, %.1f] V\n"
                "  pin A0 wrt GND" % c.A_REF,
                font=FONT_MONOSPACE,
            ),
            0,
            0,
        )
        grid.addWidget(
            QLabel(
                "Input: sig_I\n" "  [-3.3, 3.3] V\n" "  pin A1(+), A2(-)"
                if c.ADC_DIFFERENTIAL
                else "Input: sig_I\n"
                "  [ 0.0, %.1f] V\n"
                "  pin A1 wrt GND" % c.A_REF,
                font=FONT_MONOSPACE,
            ),
            1,
            0,
        )

        qgrp_connections = QGroupBox("Analog connections")
        qgrp_connections.setLayout(grid)

        # QGROUP: Axes controls
        self.qpbt_fullrange_xy = QPushButton("Full range")
        self.qpbt_fullrange_xy.clicked.connect(self.process_qpbt_fullrange_xy)
        self.qpbt_autorange_xy = QPushButton("Auto range")
        self.qpbt_autorange_xy.clicked.connect(self.process_qpbt_autorange_xy)
        self.qpbt_autorange_x = QPushButton("Auto x", maximumWidth=80)
        self.qpbt_autorange_x.clicked.connect(self.process_qpbt_autorange_x)
        self.qpbt_autorange_y = QPushButton("Auto y", maximumWidth=80)
        self.qpbt_autorange_y.clicked.connect(self.process_qpbt_autorange_y)

        # fmt: off
        grid = QGridLayout(spacing=4)
        grid.addWidget(self.qpbt_fullrange_xy, 0, 0, 1, 2)
        grid.addWidget(self.qpbt_autorange_xy, 2, 0, 1, 2)
        grid.addWidget(self.qpbt_autorange_x , 3, 0)
        grid.addWidget(self.qpbt_autorange_y , 3, 1)
        # fmt: on

        qgrp_axes_controls = QGroupBox("Zoom timeseries")
        qgrp_axes_controls.setLayout(grid)

        # QGROUP: Filter settled?
        self.LED_filt_1_settled = create_LED_indicator_rect(False, "NO")
        self.LED_filt_2_settled = create_LED_indicator_rect(False, "NO")

        # fmt: off
        grid = QGridLayout(spacing=4)
        grid.addWidget(QLabel("Filter @ sig_I")  , 0, 0)
        grid.addWidget(self.LED_filt_1_settled   , 0, 1)
        grid.addWidget(QLabel("Filter @ mix_X/Y"), 1, 0)
        grid.addWidget(self.LED_filt_2_settled   , 1, 1)
        # fmt: on

        qgrp_settling = QGroupBox("Filters settled?")
        qgrp_settling.setLayout(grid)

        # Round up frame
        vbox_sidebar = QVBoxLayout()
        vbox_sidebar.addItem(QSpacerItem(0, 30))
        vbox_sidebar.addWidget(self.qpbt_ENA_lockin, stretch=0)
        vbox_sidebar.addWidget(qgrp_refsig, stretch=0)
        vbox_sidebar.addWidget(qgrp_connections, stretch=0)
        vbox_sidebar.addWidget(qgrp_axes_controls, stretch=0)
        vbox_sidebar.addWidget(qgrp_settling, stretch=0)
        vbox_sidebar.addStretch(1)

        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        #
        #   TAB PAGE: Main
        #
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------

        def Reference_and_signal():  # pylint: disable=unused-variable
            pass  # IDE bookmark

        # -----------------------------------
        # -----------------------------------
        #   FRAME: Reference and signal
        # -----------------------------------
        # -----------------------------------

        # Graph: Readings
        self.pw_refsig = pg.PlotWidget(
            axisItems={
                "bottom": CustomAxis(orientation="bottom"),
                "left": CustomAxis(orientation="left"),
                "top": CustomAxis(orientation="top"),
                "right": CustomAxis(orientation="right"),
            }
        )
        self.all_graphs.append(self.pw_refsig)

        self.pi_refsig = self.pw_refsig.getPlotItem()
        apply_PlotItem_style(self.pi_refsig, "Readings", "ms", "V")
        self.pi_refsig.setXRange(
            -c.BLOCK_SIZE * c.SAMPLING_PERIOD * 1e3, 0.01, padding=0
        )
        self.pi_refsig.setYRange(0.0, 3.3, padding=0.05)
        self.pi_refsig.setAutoVisible(x=True, y=True)
        self.pi_refsig.setClipToView(True)
        self.pi_refsig.setLimits(
            xMin=-c.BLOCK_SIZE * c.SAMPLING_PERIOD * 1e3,
            xMax=0.01,
            yMin=-3.465 if c.ADC_DIFFERENTIAL else -0.165,
            yMax=3.465,
        )

        self.hcc_ref_X = HistoryChartCurve(
            capacity=c.BLOCK_SIZE,
            linked_curve=self.pi_refsig.plot(pen=self.PEN_01, name="ref_X*"),
        )
        self.hcc_ref_Y = HistoryChartCurve(
            capacity=c.BLOCK_SIZE,
            linked_curve=self.pi_refsig.plot(pen=self.PEN_02, name="ref_Y*"),
        )
        self.hcc_sig_I = HistoryChartCurve(
            capacity=c.BLOCK_SIZE,
            linked_curve=self.pi_refsig.plot(pen=self.PEN_03, name="sig_I"),
        )
        curves = [self.hcc_ref_X, self.hcc_ref_Y, self.hcc_sig_I]
        self.all_curves.extend(curves)
        self.hcc_ref_X.setVisible(True)
        self.hcc_ref_Y.setVisible(False)
        self.legend_refsig = LegendSelect(
            linked_curves=curves,
            hide_toggle_button=False,
            box_bg_color=COLOR_GRAPH_BG,
        )

        self.hcc_ref_X.x_axis_divisor = 1000  # From [us] to [ms]
        self.hcc_ref_Y.x_axis_divisor = 1000  # From [us] to [ms]
        self.hcc_sig_I.x_axis_divisor = 1000  # From [us] to [ms]

        # QGROUP: Readings
        p = {"maximumWidth": ex8, "minimumWidth": ex8, "readOnly": True}
        self.qlin_time = QLineEdit(readOnly=True)
        self.qlin_sig_I_max = QLineEdit(**p)
        self.qlin_sig_I_min = QLineEdit(**p)
        self.qlin_sig_I_avg = QLineEdit(**p)
        self.qlin_sig_I_std = QLineEdit(**p)
        self.qlin_time.setAlignment(QtCore.Qt.AlignHCenter)
        self.qlin_sig_I_max.setAlignment(QtCore.Qt.AlignRight)
        self.qlin_sig_I_min.setAlignment(QtCore.Qt.AlignRight)
        self.qlin_sig_I_avg.setAlignment(QtCore.Qt.AlignRight)
        self.qlin_sig_I_std.setAlignment(QtCore.Qt.AlignRight)

        # fmt: off
        i = 0
        grid = QGridLayout(spacing=4)
        grid.addWidget(self.qlin_time         , i, 0, 1, 3); i+=1
        grid.addItem(QSpacerItem(0, 6)        , i, 0)      ; i+=1
        grid.addLayout(self.legend_refsig.grid, i, 0, 1, 3); i+=1
        grid.addItem(QSpacerItem(0, 6)        , i, 0); i+=1
        grid.addWidget(QLabel("sig_I:")       , i, 0); i+=1
        grid.addItem(QSpacerItem(0, 4)        , i, 0); i+=1
        grid.addWidget(QLabel("max")          , i, 0)
        grid.addWidget(self.qlin_sig_I_max    , i, 1)
        grid.addWidget(QLabel("V")            , i, 2); i+=1
        grid.addWidget(QLabel("min")          , i, 0)
        grid.addWidget(self.qlin_sig_I_min    , i, 1)
        grid.addWidget(QLabel("V")            , i, 2); i+=1
        grid.addItem(QSpacerItem(0, 4)        , i, 0); i+=1
        grid.addWidget(QLabel("avg")          , i, 0)
        grid.addWidget(self.qlin_sig_I_avg    , i, 1)
        grid.addWidget(QLabel("V")            , i, 2); i+=1
        grid.addWidget(QLabel("std")          , i, 0)
        grid.addWidget(self.qlin_sig_I_std    , i, 1)
        grid.addWidget(QLabel("V")            , i, 2); i+=1
        grid.setAlignment(QtCore.Qt.AlignTop)
        # fmt: on

        qgrp_readings = QGroupBox("Readings")
        qgrp_readings.setLayout(grid)

        def LIA_output():  # pylint: disable=unused-variable
            pass  # IDE bookmark

        # -----------------------------------
        # -----------------------------------
        #   FRAME: LIA output
        # -----------------------------------
        # -----------------------------------

        # Graph: (X or R) and (Y or T)
        self.pw_XR = pg.PlotWidget(
            axisItems={
                "bottom": CustomAxis(orientation="bottom"),
                "left": CustomAxis(orientation="left"),
                "top": CustomAxis(orientation="top"),
                "right": CustomAxis(orientation="right"),
            }
        )
        self.pw_YT = pg.PlotWidget(
            axisItems={
                "bottom": CustomAxis(orientation="bottom"),
                "left": CustomAxis(orientation="left"),
                "top": CustomAxis(orientation="top"),
                "right": CustomAxis(orientation="right"),
            }
        )
        self.all_graphs.append(self.pw_XR)
        self.all_graphs.append(self.pw_YT)

        self.pi_XR = self.pw_XR.getPlotItem()
        self.pi_YT = self.pw_YT.getPlotItem()
        apply_PlotItem_style(self.pi_XR, "", "ms", "V")
        apply_PlotItem_style(self.pi_YT, "", "ms")

        self.pi_XR.setXRange(
            -c.BLOCK_SIZE * c.SAMPLING_PERIOD * 1e3, 0.01, padding=0
        )
        self.pi_XR.setYRange(0, 5, padding=0.05)
        self.pi_XR.setAutoVisible(x=True, y=True)
        self.pi_XR.setClipToView(True)
        self.pi_XR.request_autorange_y = False
        self.pi_XR.setLimits(
            xMin=-c.BLOCK_SIZE * c.SAMPLING_PERIOD * 1e3,
            xMax=0.01,
        )

        self.pi_YT.setXRange(
            -c.BLOCK_SIZE * c.SAMPLING_PERIOD * 1e3, 0.01, padding=0
        )
        self.pi_YT.setYRange(-180, 180, padding=0.1)
        self.pi_YT.setAutoVisible(x=True, y=True)
        self.pi_YT.setClipToView(True)
        self.pi_YT.request_autorange_y = False
        self.pi_YT.setLimits(
            xMin=-c.BLOCK_SIZE * c.SAMPLING_PERIOD * 1e3,
            xMax=0.01,
        )

        self.hcc_LIA_XR = HistoryChartCurve(
            capacity=c.BLOCK_SIZE,
            linked_curve=self.pi_XR.plot(pen=self.PEN_03, name="X/R"),
        )
        self.hcc_LIA_YT = HistoryChartCurve(
            capacity=c.BLOCK_SIZE,
            linked_curve=self.pi_YT.plot(pen=self.PEN_03, name="Y/\u0398"),
        )
        self.all_curves.extend([self.hcc_LIA_XR, self.hcc_LIA_YT])

        self.hcc_LIA_XR.x_axis_divisor = 1000  # From [us] to [ms]
        self.hcc_LIA_YT.x_axis_divisor = 1000  # From [us] to [ms]

        self.grid_pws_XYRT = QGridLayout(spacing=0)
        self.grid_pws_XYRT.addWidget(self.pw_XR, 0, 0)
        self.grid_pws_XYRT.addWidget(self.pw_YT, 0, 1)

        # QGROUP: Show X-Y or R-Theta
        self.qrbt_XR_X = QRadioButton("X")
        self.qrbt_XR_R = QRadioButton("R", checked=True)
        self.qrbt_YT_Y = QRadioButton("Y")
        self.qrbt_YT_T = QRadioButton("\u0398", checked=True)
        self.qrbt_XR_X.clicked.connect(self.process_qrbt_XR)
        self.qrbt_XR_R.clicked.connect(self.process_qrbt_XR)
        self.qrbt_YT_Y.clicked.connect(self.process_qrbt_YT)
        self.qrbt_YT_T.clicked.connect(self.process_qrbt_YT)
        self.process_qrbt_XR()
        self.process_qrbt_YT()

        vbox = QVBoxLayout(spacing=4)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.addWidget(self.qrbt_XR_X)
        vbox.addWidget(self.qrbt_XR_R)
        qgrp_XR = QGroupBox(flat=True)
        qgrp_XR.setLayout(vbox)

        vbox = QVBoxLayout(spacing=4)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.addWidget(self.qrbt_YT_Y)
        vbox.addWidget(self.qrbt_YT_T)
        qgrp_YT = QGroupBox(flat=True)
        qgrp_YT.setLayout(vbox)

        hbox = QHBoxLayout(spacing=4)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(qgrp_XR)
        hbox.addWidget(qgrp_YT)

        p = {"maximumWidth": ex8, "minimumWidth": ex8, "readOnly": True}
        self.qlin_X_avg = QLineEdit(**p)
        self.qlin_Y_avg = QLineEdit(**p)
        self.qlin_R_avg = QLineEdit(**p)
        self.qlin_T_avg = QLineEdit(**p)
        self.qlin_X_avg.setAlignment(QtCore.Qt.AlignRight)
        self.qlin_Y_avg.setAlignment(QtCore.Qt.AlignRight)
        self.qlin_R_avg.setAlignment(QtCore.Qt.AlignRight)
        self.qlin_T_avg.setAlignment(QtCore.Qt.AlignRight)

        # fmt: off
        i = 0
        grid = QGridLayout(spacing=4)
        grid.addLayout(hbox                , i, 0, 1, 3); i+=1
        grid.addItem(QSpacerItem(0, 8)     , i, 0); i+=1
        grid.addWidget(QLabel("avg X")     , i, 0)
        grid.addWidget(self.qlin_X_avg     , i, 1)
        grid.addWidget(QLabel("V")         , i, 2); i+=1
        grid.addWidget(QLabel("avg Y")     , i, 0)
        grid.addWidget(self.qlin_Y_avg     , i, 1)
        grid.addWidget(QLabel("V")         , i, 2); i+=1
        grid.addItem(QSpacerItem(0, 8)     , i, 0); i+=1
        grid.addWidget(QLabel("avg R")     , i, 0)
        grid.addWidget(self.qlin_R_avg     , i, 1)
        grid.addWidget(QLabel("V")         , i, 2); i+=1
        grid.addWidget(QLabel("avg \u0398"), i, 0)
        grid.addWidget(self.qlin_T_avg     , i, 1)
        grid.addWidget(QLabel("\u00B0")    , i, 2)
        grid.setAlignment(QtCore.Qt.AlignTop)
        # fmt: on

        qgrp_XRYT = QGroupBox("X/R, Y/\u0398")
        qgrp_XRYT.setLayout(grid)

        # -----------------------------------
        # -----------------------------------
        #   Round up tab page 'Main'
        # -----------------------------------
        # -----------------------------------

        # fmt: off
        grid = QGridLayout(spacing=0)
        grid.addWidget(qgrp_readings     , 0, 0, QtCore.Qt.AlignTop)
        grid.addWidget(self.pw_refsig    , 0, 1)
        grid.addWidget(qgrp_XRYT         , 1, 0, QtCore.Qt.AlignTop)
        grid.addLayout(self.grid_pws_XYRT, 1, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnMinimumWidth(0, LEFT_COLUMN_WIDTH)
        # fmt: on

        self.tab_main.setLayout(grid)

        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        #
        #   TAB PAGE: Mixer
        #
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------

        def Filter_1_output():  # pylint: disable=unused-variable
            pass  # IDE bookmark

        # -----------------------------------
        # -----------------------------------
        #   FRAME: Filter 1 output
        # -----------------------------------
        # -----------------------------------

        # Graph: Filter @ sig_I
        self.pw_filt_1 = pg.PlotWidget(
            axisItems={
                "bottom": CustomAxis(orientation="bottom"),
                "left": CustomAxis(orientation="left"),
                "top": CustomAxis(orientation="top"),
                "right": CustomAxis(orientation="right"),
            }
        )
        self.all_graphs.append(self.pw_filt_1)

        self.pi_filt_1 = self.pw_filt_1.getPlotItem()
        apply_PlotItem_style(self.pi_filt_1, "Filter @ sig_I", "ms", "V")
        self.pi_filt_1.setXRange(
            -c.BLOCK_SIZE * c.SAMPLING_PERIOD * 1e3, 0.01, padding=0
        )
        self.pi_filt_1.setYRange(-3.3, 3.3, padding=0.025)
        self.pi_filt_1.setAutoVisible(x=True, y=True)
        self.pi_filt_1.setClipToView(True)
        self.pi_filt_1.setLimits(
            xMin=-c.BLOCK_SIZE * c.SAMPLING_PERIOD * 1e3,
            xMax=0.01,
            yMin=-3.465,
            yMax=3.465,
        )

        self.hcc_filt_1_in = HistoryChartCurve(
            capacity=c.BLOCK_SIZE,
            linked_curve=self.pi_filt_1.plot(pen=self.PEN_03, name="sig_I"),
        )
        self.hcc_filt_1_out = HistoryChartCurve(
            capacity=c.BLOCK_SIZE,
            linked_curve=self.pi_filt_1.plot(pen=self.PEN_04, name="filt_I"),
        )
        curves = [self.hcc_filt_1_in, self.hcc_filt_1_out]
        self.all_curves.extend(curves)
        self.legend_filt_1 = LegendSelect(
            linked_curves=curves,
            hide_toggle_button=False,
            box_bg_color=COLOR_GRAPH_BG,
        )

        self.hcc_filt_1_in.x_axis_divisor = 1000  # From [us] to [ms]
        self.hcc_filt_1_out.x_axis_divisor = 1000  # From [us] to [ms]

        # QGROUP: Filter output
        p = {"maximumWidth": ex8, "minimumWidth": ex8, "readOnly": True}
        self.qlin_filt_I_max = QLineEdit(**p)
        self.qlin_filt_I_min = QLineEdit(**p)
        self.qlin_filt_I_avg = QLineEdit(**p)
        self.qlin_filt_I_std = QLineEdit(**p)
        self.qlin_filt_I_max.setAlignment(QtCore.Qt.AlignRight)
        self.qlin_filt_I_min.setAlignment(QtCore.Qt.AlignRight)
        self.qlin_filt_I_avg.setAlignment(QtCore.Qt.AlignRight)
        self.qlin_filt_I_std.setAlignment(QtCore.Qt.AlignRight)

        # fmt: off
        i = 0
        grid = QGridLayout(spacing=4)
        grid.addLayout(self.legend_filt_1.grid, i, 0, 1, 3); i+=1
        grid.addItem(QSpacerItem(0, 6)        , i, 0)      ; i+=1
        grid.addWidget(QLabel("filt_I:")      , i, 0)      ; i+=1
        grid.addItem(QSpacerItem(0, 4)        , i, 0)      ; i+=1
        grid.addWidget(QLabel("max")          , i, 0)
        grid.addWidget(self.qlin_filt_I_max   , i, 1)
        grid.addWidget(QLabel("V")            , i, 2)      ; i+=1
        grid.addWidget(QLabel("min")          , i, 0)
        grid.addWidget(self.qlin_filt_I_min   , i, 1)
        grid.addWidget(QLabel("V")            , i, 2)      ; i+=1
        grid.addItem(QSpacerItem(0, 4)        , i, 0)      ; i+=1
        grid.addWidget(QLabel("avg")          , i, 0)
        grid.addWidget(self.qlin_filt_I_avg   , i, 1)
        grid.addWidget(QLabel("V")            , i, 2)      ; i+=1
        grid.addWidget(QLabel("std")          , i, 0)
        grid.addWidget(self.qlin_filt_I_std   , i, 1)
        grid.addWidget(QLabel("V")            , i, 2)      ; i+=1
        grid.setAlignment(QtCore.Qt.AlignTop)
        # fmt: on

        qgrp_filt_1 = QGroupBox("Filter @ sig_I")
        qgrp_filt_1.setLayout(grid)

        def Mixer():  # pylint: disable=unused-variable
            pass  # IDE bookmark

        # -----------------------------------
        # -----------------------------------
        #   FRAME: Mixer
        # -----------------------------------
        # -----------------------------------

        # Graph: Mixer
        self.pw_mixer = pg.PlotWidget(
            axisItems={
                "bottom": CustomAxis(orientation="bottom"),
                "left": CustomAxis(orientation="left"),
                "top": CustomAxis(orientation="top"),
                "right": CustomAxis(orientation="right"),
            }
        )
        self.all_graphs.append(self.pw_mixer)

        self.pi_mixer = self.pw_mixer.getPlotItem()
        apply_PlotItem_style(self.pi_mixer, "Mixer", "ms", "V")
        self.pi_mixer.setXRange(
            -c.BLOCK_SIZE * c.SAMPLING_PERIOD * 1e3, 0.01, padding=0
        )
        self.pi_mixer.setYRange(-5, 5, padding=0.025)
        self.pi_mixer.setAutoVisible(x=True, y=True)
        self.pi_mixer.setClipToView(True)
        self.pi_mixer.setLimits(
            xMin=-c.BLOCK_SIZE * c.SAMPLING_PERIOD * 1e3,
            xMax=0.01,
            yMin=-5.25,
            yMax=5.25,
        )

        self.hcc_mix_X = HistoryChartCurve(
            capacity=c.BLOCK_SIZE,
            linked_curve=self.pi_mixer.plot(pen=self.PEN_01, name="mix_X"),
        )
        self.hcc_mix_Y = HistoryChartCurve(
            capacity=c.BLOCK_SIZE,
            linked_curve=self.pi_mixer.plot(pen=self.PEN_02, name="mix_Y"),
        )
        curves = [self.hcc_mix_X, self.hcc_mix_Y]
        self.all_curves.extend(curves)
        self.legend_mixer = LegendSelect(
            linked_curves=curves,
            hide_toggle_button=False,
            box_bg_color=COLOR_GRAPH_BG,
        )

        self.hcc_mix_X.x_axis_divisor = 1000  # From [us] to [ms]
        self.hcc_mix_Y.x_axis_divisor = 1000  # From [us] to [ms]

        # QGROUP: Mixer
        qgrp_mixer = QGroupBox("Mixer")
        qgrp_mixer.setLayout(self.legend_mixer.grid)

        # -----------------------------------
        # -----------------------------------
        #   Round up tab page 'Mixer'
        # -----------------------------------
        # -----------------------------------

        # fmt: off
        grid = QGridLayout(spacing=0)
        grid.addWidget(qgrp_filt_1   , 0, 0, QtCore.Qt.AlignTop)
        grid.addWidget(self.pw_filt_1, 0, 1)
        grid.addWidget(qgrp_mixer    , 1, 0, QtCore.Qt.AlignTop)
        grid.addWidget(self.pw_mixer , 1, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnMinimumWidth(0, LEFT_COLUMN_WIDTH)
        # fmt: on

        self.tab_mixer.setLayout(grid)

        def Power_spectrum():  # pylint: disable=unused-variable
            pass  # IDE bookmark

        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        #
        #   TAB PAGE: Power spectrum
        #
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------

        # Graph: Power spectrum
        self.pw_PS = pg.PlotWidget(
            axisItems={
                "bottom": CustomAxis(orientation="bottom"),
                "left": CustomAxis(orientation="left"),
                "top": CustomAxis(orientation="top"),
                "right": CustomAxis(orientation="right"),
            }
        )
        self.all_graphs.append(self.pw_PS)

        self.pi_PS = self.pw_PS.getPlotItem()
        apply_PlotItem_style(
            self.pi_PS, "Power spectrum (Welch)", "Hz", "dBV<sub>RMS</sub>"
        )
        self.pi_PS.setAutoVisible(x=True, y=True)
        self.pi_PS.setXRange(0, c.F_Nyquist, padding=0)
        self.pi_PS.setYRange(-165, 10, padding=0)
        self.pi_PS.setClipToView(True)
        self.pi_PS.setLimits(
            xMin=-1,
            xMax=c.F_Nyquist,
            yMin=-165,
            yMax=10,
        )

        self.pc_PS_sig_I = PlotCurve(
            self.pi_PS.plot(pen=self.PEN_03, name="sig_I")
        )
        self.pc_PS_filt_I = PlotCurve(
            self.pi_PS.plot(pen=self.PEN_04, name="filt_I")
        )
        self.pc_PS_mix_X = PlotCurve(
            self.pi_PS.plot(pen=self.PEN_01, name="mix_X")
        )
        self.pc_PS_mix_Y = PlotCurve(
            self.pi_PS.plot(pen=self.PEN_02, name="mix_Y")
        )
        self.pc_PS_R = PlotCurve(self.pi_PS.plot(pen=self.PEN_05, name="R"))
        self.pc_PS_filt_I.setVisible(False)
        self.pc_PS_mix_X.setVisible(False)
        self.pc_PS_mix_Y.setVisible(False)
        self.pc_PS_R.setVisible(False)
        curves = [
            self.pc_PS_sig_I,
            self.pc_PS_filt_I,
            self.pc_PS_mix_X,
            self.pc_PS_mix_Y,
            self.pc_PS_R,
        ]
        self.all_curves.extend(curves)
        self.legend_PS = LegendSelect(
            linked_curves=curves,
            hide_toggle_button=False,
            box_bg_color=COLOR_GRAPH_BG,
        )

        # QGROUP: Zoom
        # fmt: off
        self.qpbt_PS_zoom_DC   = QPushButton("DC")
        self.qpbt_PS_zoom_ref  = QPushButton("ref_freq")
        self.qpbt_PS_zoom_2ref = QPushButton("2x ref_freq")
        self.qpbt_PS_zoom_s1   = QPushButton("0 - 0.5 kHz")
        self.qpbt_PS_zoom_s2   = QPushButton("0 - 1 kHz")
        self.qpbt_PS_zoom_s3   = QPushButton("0 - 2.5 kHz")
        self.qpbt_PS_zoom_all  = QPushButton("Full range")
        # fmt: on

        self.qpbt_PS_zoom_DC.clicked.connect(
            lambda: self.plot_zoom_x(self.pi_PS, 0, 10)
        )
        self.qpbt_PS_zoom_ref.clicked.connect(
            lambda: self.plot_zoom_x(
                self.pi_PS,
                c.ref_freq - 10,
                c.ref_freq + 10,
            )
        )
        self.qpbt_PS_zoom_2ref.clicked.connect(
            lambda: self.plot_zoom_x(
                self.pi_PS,
                2 * c.ref_freq - 10,
                2 * c.ref_freq + 10,
            )
        )
        self.qpbt_PS_zoom_s1.clicked.connect(
            lambda: self.plot_zoom_x(self.pi_PS, 0, 500)
        )
        self.qpbt_PS_zoom_s2.clicked.connect(
            lambda: self.plot_zoom_x(self.pi_PS, 0, 1000)
        )
        self.qpbt_PS_zoom_s3.clicked.connect(
            lambda: self.plot_zoom_x(self.pi_PS, 0, 2500)
        )
        self.qpbt_PS_zoom_all.clicked.connect(
            lambda: self.plot_zoom_x(self.pi_PS, 0, c.F_Nyquist)
        )

        # fmt: off
        grid = QGridLayout()
        grid.addWidget(self.qpbt_PS_zoom_DC  , 0, 0)
        grid.addWidget(self.qpbt_PS_zoom_ref , 0, 1)
        grid.addWidget(self.qpbt_PS_zoom_2ref, 0, 2)
        grid.addWidget(self.qpbt_PS_zoom_s1  , 0, 3)
        grid.addWidget(self.qpbt_PS_zoom_s2  , 0, 4)
        grid.addWidget(self.qpbt_PS_zoom_s3  , 0, 5)
        grid.addWidget(self.qpbt_PS_zoom_all , 0, 6)
        # fmt: on

        qgrp_zoom = QGroupBox("Zoom")
        qgrp_zoom.setLayout(grid)

        # QGROUP: Power spectrum
        grid = QGridLayout(spacing=4)
        grid.addItem(QSpacerItem(0, 4), 0, 0)
        grid.addLayout(self.legend_PS.grid, 1, 0)
        grid.setAlignment(QtCore.Qt.AlignTop)

        qgrp_PS = QGroupBox("Pow. spectrum")
        qgrp_PS.setLayout(grid)

        # -----------------------------------
        # -----------------------------------
        #   Round up tab page 'Power spectrum'
        # -----------------------------------
        # -----------------------------------

        # fmt: off
        grid = QGridLayout(spacing=0)
        grid.addWidget(qgrp_PS   , 0, 0, 2, 1, QtCore.Qt.AlignTop)
        grid.addWidget(qgrp_zoom , 0, 1)
        grid.addWidget(self.pw_PS, 1, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnMinimumWidth(0, LEFT_COLUMN_WIDTH)
        # fmt: on

        self.tab_power_spectrum.setLayout(grid)

        def Filter_1_design():  # pylint: disable=unused-variable
            pass  # IDE bookmark

        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        #
        #   TAB PAGE: Filter design @ sig_I
        #
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------

        # Graph: Filter response @ sig_I
        self.pw_filt_1_resp = pg.PlotWidget(
            axisItems={
                "bottom": CustomAxis(orientation="bottom"),
                "left": CustomAxis(orientation="left"),
                "top": CustomAxis(orientation="top"),
                "right": CustomAxis(orientation="right"),
            }
        )
        self.all_graphs.append(self.pw_filt_1_resp)

        self.pi_filt_1_resp = self.pw_filt_1_resp.getPlotItem()
        apply_PlotItem_style(
            self.pi_filt_1_resp,
            "Filter response @ sig_I",
            "Hz",
            "amplitude attenuation \u2014 dBV",
        )
        self.pi_filt_1_resp.setAutoVisible(x=True, y=True)
        self.pi_filt_1_resp.enableAutoRange("x", False)
        self.pi_filt_1_resp.enableAutoRange("y", True)
        self.pi_filt_1_resp.setClipToView(True)
        self.pi_filt_1_resp.setLimits(
            xMin=-1, xMax=c.F_Nyquist, yMin=-120, yMax=20
        )

        if 0:  # pylint: disable=using-constant-test
            # This will show individual symbols per data point
            # Only enable for debugging
            self.curve_filt_1_resp = pg.ScatterPlotItem(
                pen=self.PEN_03, size=8, symbol="o"
            )
        else:
            # Fast line plotting
            self.curve_filt_1_resp = pg.PlotCurveItem(
                pen=self.PEN_03, brush=self.BRUSH_03
            )
        self.pi_filt_1_resp.addItem(self.curve_filt_1_resp)
        self.update_plot_filt_1_resp()
        self.plot_zoom_ROI_filt_1()

        # QGROUP: Zoom
        self.qpbt_filt_1_resp_zoom_DC = QPushButton("DC")
        self.qpbt_filt_1_resp_zoom_50 = QPushButton("50 Hz")
        self.qpbt_filt_1_resp_zoom_s1 = QPushButton("0 - 200 Hz")
        self.qpbt_filt_1_resp_zoom_s2 = QPushButton("0 - 1 kHz")
        self.qpbt_filt_1_resp_zoom_all = QPushButton("Full range")
        self.qpbt_filt_1_resp_zoom_ROI = QPushButton("Region of interest")

        self.qpbt_filt_1_resp_zoom_DC.clicked.connect(
            lambda: self.plot_zoom_x(self.pi_filt_1_resp, 0, 10)
        )
        self.qpbt_filt_1_resp_zoom_50.clicked.connect(
            lambda: self.plot_zoom_x(self.pi_filt_1_resp, 45, 55)
        )
        self.qpbt_filt_1_resp_zoom_s1.clicked.connect(
            lambda: self.plot_zoom_x(self.pi_filt_1_resp, 0, 200)
        )
        self.qpbt_filt_1_resp_zoom_s2.clicked.connect(
            lambda: self.plot_zoom_x(self.pi_filt_1_resp, 0, 1000)
        )
        self.qpbt_filt_1_resp_zoom_all.clicked.connect(
            lambda: self.plot_zoom_x(self.pi_filt_1_resp, 0, c.F_Nyquist)
        )
        self.qpbt_filt_1_resp_zoom_ROI.clicked.connect(
            self.plot_zoom_ROI_filt_1
        )

        grid = QGridLayout()
        grid.addWidget(self.qpbt_filt_1_resp_zoom_DC, 0, 0)
        grid.addWidget(self.qpbt_filt_1_resp_zoom_50, 0, 1)
        grid.addWidget(self.qpbt_filt_1_resp_zoom_s1, 0, 2)
        grid.addWidget(self.qpbt_filt_1_resp_zoom_s2, 0, 3)
        grid.addWidget(self.qpbt_filt_1_resp_zoom_all, 0, 4)
        grid.addWidget(self.qpbt_filt_1_resp_zoom_ROI, 0, 5)

        qgrp_zoom = QGroupBox("Zoom")
        qgrp_zoom.setLayout(grid)

        # QGROUP: Filter design
        self.filt_1_design_GUI = Filter_design_GUI(self.alia_qdev.firf_1_sig_I)
        self.filt_1_design_GUI.signal_filter_design_updated.connect(
            self.update_plot_filt_1_resp
        )

        # -----------------------------------
        # -----------------------------------
        #   Round up tab page 'Filter response @ sig_I
        # -----------------------------------
        # -----------------------------------

        grid = QGridLayout(spacing=0)
        grid.addWidget(
            self.filt_1_design_GUI.qgrp, 0, 0, 2, 1, QtCore.Qt.AlignTop
        )
        grid.addWidget(qgrp_zoom, 0, 1)
        grid.addWidget(self.pw_filt_1_resp, 1, 1)
        grid.setColumnStretch(1, 1)

        self.tab_filter_1_design.setLayout(grid)

        def Filter_2_design():  # pylint: disable=unused-variable
            pass  # IDE bookmark

        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        #
        #   TAB PAGE: Filter design @ mix_X/Y
        #
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------

        # Graph: Filter response @ mix_X/Y
        self.pw_filt_2_resp = pg.PlotWidget(
            axisItems={
                "bottom": CustomAxis(orientation="bottom"),
                "left": CustomAxis(orientation="left"),
                "top": CustomAxis(orientation="top"),
                "right": CustomAxis(orientation="right"),
            }
        )
        self.all_graphs.append(self.pw_filt_2_resp)

        self.pi_filt_2_resp = self.pw_filt_2_resp.getPlotItem()
        apply_PlotItem_style(
            self.pi_filt_2_resp,
            "Filter response @ mix_X/Y",
            "Hz",
            "amplitude attenuation \u2014 dBV",
        )
        self.pi_filt_2_resp.setAutoVisible(x=True, y=True)
        self.pi_filt_2_resp.enableAutoRange("x", False)
        self.pi_filt_2_resp.enableAutoRange("y", True)
        self.pi_filt_2_resp.setClipToView(True)
        self.pi_filt_2_resp.setLimits(
            xMin=-1, xMax=c.F_Nyquist, yMin=-120, yMax=20
        )

        if 0:  # pylint: disable=using-constant-test
            # Show individual symbols per data point
            # Only enable for debugging
            self.curve_filt_2_resp = pg.ScatterPlotItem(
                pen=self.PEN_03, size=8, symbol="o"
            )
        else:
            # Fast line plotting
            self.curve_filt_2_resp = pg.PlotCurveItem(
                pen=self.PEN_03, brush=self.BRUSH_03
            )
        self.pi_filt_2_resp.addItem(self.curve_filt_2_resp)
        self.update_plot_filt_2_resp()
        self.plot_zoom_ROI_filt_2()

        # QGROUP: Zoom
        self.qpbt_filt_2_resp_zoom_DC = QPushButton("DC")
        self.qpbt_filt_2_resp_zoom_50 = QPushButton("50 Hz")
        self.qpbt_filt_2_resp_zoom_s1 = QPushButton("0 - 200 Hz")
        self.qpbt_filt_2_resp_zoom_s2 = QPushButton("0 - 1 kHz")
        self.qpbt_filt_2_resp_zoom_all = QPushButton("Full range")
        self.qpbt_filt_2_resp_zoom_ROI = QPushButton("Region of interest")

        self.qpbt_filt_2_resp_zoom_DC.clicked.connect(
            lambda: self.plot_zoom_x(self.pi_filt_2_resp, 0, 10)
        )
        self.qpbt_filt_2_resp_zoom_50.clicked.connect(
            lambda: self.plot_zoom_x(self.pi_filt_2_resp, 45, 55)
        )
        self.qpbt_filt_2_resp_zoom_s1.clicked.connect(
            lambda: self.plot_zoom_x(self.pi_filt_2_resp, 0, 200)
        )
        self.qpbt_filt_2_resp_zoom_s2.clicked.connect(
            lambda: self.plot_zoom_x(self.pi_filt_2_resp, 0, 1000)
        )
        self.qpbt_filt_2_resp_zoom_all.clicked.connect(
            lambda: self.plot_zoom_x(self.pi_filt_2_resp, 0, c.F_Nyquist)
        )
        self.qpbt_filt_2_resp_zoom_ROI.clicked.connect(
            self.plot_zoom_ROI_filt_2
        )

        grid = QGridLayout()
        grid.addWidget(self.qpbt_filt_2_resp_zoom_DC, 0, 0)
        grid.addWidget(self.qpbt_filt_2_resp_zoom_50, 0, 1)
        grid.addWidget(self.qpbt_filt_2_resp_zoom_s1, 0, 2)
        grid.addWidget(self.qpbt_filt_2_resp_zoom_s2, 0, 3)
        grid.addWidget(self.qpbt_filt_2_resp_zoom_all, 0, 4)
        grid.addWidget(self.qpbt_filt_2_resp_zoom_ROI, 0, 5)

        qgrp_zoom = QGroupBox("Zoom")
        qgrp_zoom.setLayout(grid)

        # QGROUP: Filter design
        self.filt_2_design_GUI = Filter_design_GUI(
            [self.alia_qdev.firf_2_mix_X, self.alia_qdev.firf_2_mix_Y],
            hide_ACDC_coupling_controls=True,
        )
        self.filt_2_design_GUI.signal_filter_design_updated.connect(
            self.update_plot_filt_2_resp
        )

        # -----------------------------------
        # -----------------------------------
        #   Round up tab page 'Filter response @ mix_X/Y
        # -----------------------------------
        # -----------------------------------

        grid = QGridLayout(spacing=0)
        grid.addWidget(
            self.filt_2_design_GUI.qgrp, 0, 0, 2, 1, QtCore.Qt.AlignTop
        )
        grid.addWidget(qgrp_zoom, 0, 1)
        grid.addWidget(self.pw_filt_2_resp, 1, 1)
        grid.setColumnStretch(1, 1)

        self.tab_filter_2_design.setLayout(grid)

        def Diagram():  # pylint: disable=unused-variable
            pass  # IDE bookmark

        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        #
        #   TAB PAGE: Diagram
        #
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------

        # Schematic diagram
        qlbl = QLabel()
        qpix = QtGui.QPixmap("user_manual/fig_diagram__single-ended.png")
        qlbl.setPixmap(qpix)

        # -----------------------------------
        # -----------------------------------
        #   Round up tab page 'Settings'
        # -----------------------------------
        # -----------------------------------

        grid = QGridLayout(spacing=0)
        grid.addWidget(qlbl, 0, 0, QtCore.Qt.AlignTop)
        grid.setColumnStretch(1, 1)

        self.tab_diagram.setLayout(grid)

        def Settings():  # pylint: disable=unused-variable
            pass  # IDE bookmark

        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        #
        #   TAB PAGE: Settings
        #
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------

        p = {
            "maximumWidth": ex8,
            "alignment": QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight,
        }
        self.qlin_ADC_autocal_info = QTextEdit(
            "It is advised to perform the autocalibration routine at least once"
            " each time that the Arduino is flashed with new lock-in amplifier"
            " firmware. This routine will try to correct the offset and linear"
            " gain factor of the ADC such that the signal acquired by the ADC"
            " fits best on top of the signal as output by the DAC. Typically,"
            " this will improve the absolute accuracy of the ADC by several"
            " millivolt.<br><br>"
            "The calibration data (<b>offsetcorr & gaincorr</b>) should"
            " afterwards be stored in the persistent flash memory of the"
            " Arduino by clicking <b>'Store calibration in flash'</b>. Any"
            " reboot of the Arduino will then automaticaly load in the last"
            " stored calibration data. You can check if the calibration is"
            " already loaded in by checking the value <b>'Is valid?'</b>."
            "<br><br>"
            "During the routine the DAC voltage output will be internally"
            " routed to the ADC input. The DAC output will also be present at"
            " the analog output pin [A0]. There is no need to manually connect"
            " pin [A0] to [A1]. It is advised to first disconnect pins [A0] and"
            " [A1] as to minimize interference from any external circuitry. The"
            " routine will output a low voltage ~0.3 V, followed by a high"
            " voltage ~3 V for each around 75 ms.",
            readOnly=True,
        )
        self.qlbl_ADC_autocal_is_valid = QLabel("", **p)
        self.qlin_ADC_gaincorr = QLineEdit("", readOnly=True, **p)
        self.qlin_ADC_offsetcorr = QLineEdit("", readOnly=True, **p)
        self.qpbt_ADC_perform_autocal = QPushButton("Perform autocalibration")
        self.qpbt_ADC_perform_autocal.clicked.connect(
            self.process_qpbt_ADC_perform_autocal
        )
        self.qpbt_ADC_store_autocal = QPushButton("Store calibration in flash")
        self.qpbt_ADC_store_autocal.clicked.connect(
            self.process_qpbt_ADC_store_autocal
        )

        self.update_qgrp_ADC_calibration()
        self.alia_qdev.signal_ADC_autocalibration_was_performed.connect(
            self.update_qgrp_ADC_calibration
        )
        self.alia_qdev.signal_ADC_autocalibration_was_stored.connect(
            self.confirm_ADC_autocalibration_was_stored
        )

        # fmt: off
        grid = QGridLayout(spacing=4)
        grid.addWidget(self.qlin_ADC_autocal_info     , 0, 0, 1, 3)
        grid.addWidget(QLabel("Is valid?")            , 1, 0)
        grid.addWidget(self.qlbl_ADC_autocal_is_valid , 1, 1)
        grid.addWidget(QLabel("Offsetcorr")           , 2, 0)
        grid.addWidget(self.qlin_ADC_offsetcorr       , 2, 1)
        grid.addWidget(QLabel("Gaincorr")             , 3, 0)
        grid.addWidget(self.qlin_ADC_gaincorr         , 3, 1)
        grid.addItem(QSpacerItem(0, 4)                , 4, 0)
        grid.addWidget(self.qpbt_ADC_perform_autocal  , 5, 0, 1, 2)
        grid.addWidget(self.qpbt_ADC_store_autocal    , 6, 0, 1, 2)
        # fmt: on

        qgrp = QGroupBox("ADC calibration")
        qgrp.setLayout(grid)

        # -----------------------------------
        # -----------------------------------
        #   Round up tab page 'Settings'
        # -----------------------------------
        # -----------------------------------

        grid = QGridLayout()
        grid.addWidget(qgrp, 0, 0, QtCore.Qt.AlignTop)
        grid.setColumnStretch(1, 1)

        self.tab_settings.setLayout(grid)

        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        #
        #   Round up full window
        #
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------

        vbox = QVBoxLayout(self)
        vbox.addLayout(hbox_header)

        hbox = QHBoxLayout()
        hbox.addWidget(self.tabs, stretch=1)
        hbox.addLayout(vbox_sidebar, stretch=0)

        vbox.addItem(QSpacerItem(0, 10))
        vbox.addLayout(hbox, stretch=1)

        # -----------------------------------
        # -----------------------------------
        #   Create wall clock timer
        # -----------------------------------
        # -----------------------------------

        self.timer_wall_clock = QtCore.QTimer()
        self.timer_wall_clock.timeout.connect(self.update_wall_clock)
        self.timer_wall_clock.start(UPDATE_INTERVAL_WALL_CLOCK)

        # -----------------------------------
        # -----------------------------------
        #   Connect signals to slots
        # -----------------------------------
        # -----------------------------------

        self.alia_qdev.signal_DAQ_updated.connect(self.update_GUI)
        self.alia_qdev.signal_DAQ_paused.connect(self.update_GUI)

        self.alia_qdev.signal_ref_waveform_is_set.connect(
            self.update_newly_set_ref_waveform
        )
        self.alia_qdev.signal_ref_freq_is_set.connect(
            self.update_newly_set_ref_freq
        )
        self.alia_qdev.signal_ref_V_offset_is_set.connect(
            self.update_newly_set_ref_V_offset
        )
        self.alia_qdev.signal_ref_V_ampl_is_set.connect(
            self.update_newly_set_ref_V_ampl
        )

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    #   Handle controls
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    @QtCore.pyqtSlot()
    def update_wall_clock(self):
        cur_date_time = QDateTime.currentDateTime()
        self.qlbl_cur_date_time.setText(
            "%s    %s"
            % (
                cur_date_time.toString("dd-MM-yyyy"),
                cur_date_time.toString("HH:mm:ss"),
            )
        )

        if self.prev_time_CPU_load.msecsTo(cur_date_time) > 1000:
            cpu_syst = psutil.cpu_percent(interval=None)
            cpu_proc = self.proc.cpu_percent(interval=None) / self.cpu_count
            self.qlbl_CPU_syst.setText("CPU system : %.1f%%" % cpu_syst)
            self.qlbl_CPU_proc.setText("CPU process: %.1f%%" % cpu_proc)
            self.prev_time_CPU_load = cur_date_time

    @QtCore.pyqtSlot()
    def update_GUI(self):
        if DEBUG_TIMING:
            tock = Time.perf_counter()
            print("%.2f GUI" % (tock - self.tick))
            self.tick = tock

        # Major visual changes upcoming. Reduce CPU overhead by momentarily
        # disabling screen repaints and GUI events.
        self.setUpdatesEnabled(False)
        for graph in self.all_graphs:
            graph.setUpdatesEnabled(False)

        alia_qdev = self.alia_qdev
        self.qlbl_update_counter.setText("%i" % alia_qdev.update_counter_DAQ)

        if alia_qdev.worker_DAQ._paused:  # pylint: disable=protected-access
            self.qpbt_ENA_lockin.setText("Lock-in OFF")
            self.qpbt_ENA_lockin.setChecked(False)
            self.qlbl_DAQ_rate.setText("Blocks/s: paused")
        else:
            self.qpbt_ENA_lockin.setText("Lock-in ON")
            self.qpbt_ENA_lockin.setChecked(True)
            self.qlbl_DAQ_rate.setText(
                "Blocks/s: %.1f" % alia_qdev.obtained_DAQ_rate_Hz
            )

        if np.isnan(alia_qdev.state.time[0]):
            h, m, s = (0, 0, 0)
        else:
            m, s = divmod(alia_qdev.state.time[0] / 1e6, 60)
            h, m = divmod(m, 60)

        self.qlin_time.setText("%02.0f : %02.0f : %06.3f" % (h, m, s))
        self.qlin_sig_I_max.setText("%.4f" % alia_qdev.state.sig_I_max)
        self.qlin_sig_I_min.setText("%.4f" % alia_qdev.state.sig_I_min)
        self.qlin_sig_I_avg.setText("%.4f" % alia_qdev.state.sig_I_avg)
        self.qlin_sig_I_std.setText("%.4f" % alia_qdev.state.sig_I_std)
        self.qlin_filt_I_max.setText("%.4f" % alia_qdev.state.filt_I_max)
        self.qlin_filt_I_min.setText("%.4f" % alia_qdev.state.filt_I_min)
        self.qlin_filt_I_avg.setText("%.4f" % alia_qdev.state.filt_I_avg)
        self.qlin_filt_I_std.setText("%.4f" % alia_qdev.state.filt_I_std)
        self.qlin_X_avg.setText("%.4f" % alia_qdev.state.X_avg)
        self.qlin_Y_avg.setText("%.4f" % alia_qdev.state.Y_avg)
        self.qlin_R_avg.setText("%.4f" % alia_qdev.state.R_avg)
        self.qlin_T_avg.setText("%.3f" % alia_qdev.state.T_avg)

        if alia_qdev.firf_1_sig_I.filter_has_settled:
            self.LED_filt_1_settled.setChecked(True)
            self.LED_filt_1_settled.setText("YES")
        else:
            self.LED_filt_1_settled.setChecked(False)
            self.LED_filt_1_settled.setText("NO")

        if alia_qdev.firf_2_mix_X.filter_has_settled:
            self.LED_filt_2_settled.setChecked(True)
            self.LED_filt_2_settled.setText("YES")
        else:
            self.LED_filt_2_settled.setChecked(False)
            self.LED_filt_2_settled.setText("NO")

        # Update threadsafe curves
        self.update_curves()

        # Re-enable screen repaints and GUI events
        self.setUpdatesEnabled(True)
        for graph in self.all_graphs:
            graph.setUpdatesEnabled(True)

    @QtCore.pyqtSlot()
    def clear_curves_stage_1_and_2(self):
        self.hcc_filt_1_in.clear()
        self.hcc_filt_1_out.clear()
        self.hcc_mix_X.clear()
        self.hcc_mix_Y.clear()
        self.hcc_LIA_XR.clear()
        self.hcc_LIA_YT.clear()

    @QtCore.pyqtSlot()
    def process_qpbt_ENA_lockin(self):
        if self.qpbt_ENA_lockin.isChecked():
            self.alia_qdev.turn_on()
            self.alia_qdev.state.reset()
            self.clear_curves_stage_1_and_2()
        else:
            self.alia_qdev.turn_off()

    @QtCore.pyqtSlot(int)
    def process_qcbx_ref_waveform(self, value: int):
        ref_waveform = Waveform(value)
        if ref_waveform != self.alia.config.ref_waveform:
            self.alia_qdev.set_ref_waveform(ref_waveform)

    @QtCore.pyqtSlot()
    def process_qlin_ref_freq(self):
        try:
            ref_freq = round(float(self.qlin_ref_freq.text()), 3)
        except ValueError:
            ref_freq = self.alia.config.ref_freq

        self.qlin_ref_freq.setText("%.3f" % ref_freq)
        if ref_freq != self.alia.config.ref_freq:
            self.alia_qdev.set_ref_freq(ref_freq)

    @QtCore.pyqtSlot()
    def process_qlin_ref_V_offset(self):
        try:
            ref_V_offset = round(float(self.qlin_ref_V_offset.text()), 3)
        except ValueError:
            ref_V_offset = self.alia.config.ref_V_offset

        self.qlin_ref_V_offset.setText("%.3f" % ref_V_offset)
        if ref_V_offset != self.alia.config.ref_V_offset:
            self.alia_qdev.set_ref_V_offset(ref_V_offset)

    @QtCore.pyqtSlot()
    def process_qlin_ref_V_ampl_RMS(self):
        try:
            ref_V_ampl_RMS = round(float(self.qlin_ref_V_ampl_RMS.text()), 3)
        except ValueError:
            ref_V_ampl_RMS = self.alia.config.ref_V_ampl_RMS

        self.qlin_ref_V_ampl_RMS.setText("%.3f" % ref_V_ampl_RMS)
        if ref_V_ampl_RMS != self.alia.config.ref_V_ampl_RMS:
            self.alia_qdev.set_ref_V_ampl_RMS(ref_V_ampl_RMS)

    @QtCore.pyqtSlot()
    def update_newly_set_ref_waveform(self):
        self.qcbx_ref_waveform.setCurrentIndex(
            self.alia.config.ref_waveform.value
        )
        self.qlin_ref_V_ampl.setText("%.3f" % self.alia.config.ref_V_ampl)
        self.qlbl_ref_is_clipping.setText(self.get_clipping_text())
        # QApplication.processEvents()

        self.alia_qdev.state.reset()
        self.clear_curves_stage_1_and_2()

    @QtCore.pyqtSlot()
    def update_newly_set_ref_freq(self):
        self.qlin_ref_freq.setText("%.3f" % self.alia.config.ref_freq)
        QApplication.processEvents()

        # TODO: the extra distance 'roll_off_width' to stay away from
        # f_cutoff should be calculated based on the roll-off width of the
        # filter, instead of hard-coded
        roll_off_width = 5  # [Hz]
        f_cutoff = 2 * self.alia.config.ref_freq - roll_off_width
        if f_cutoff > self.alia.config.F_Nyquist - roll_off_width:
            print("WARNING: Filter @ mix_X/Y can't reach desired cut-off freq.")
            f_cutoff = self.alia.config.F_Nyquist - roll_off_width

        self.alia_qdev.firf_2_mix_X.compute_firwin_and_freqz(
            firwin_cutoff=f_cutoff
        )
        self.alia_qdev.firf_2_mix_Y.compute_firwin_and_freqz(
            firwin_cutoff=f_cutoff
        )
        self.filt_2_design_GUI.update_filter_design()
        self.update_plot_filt_2_resp()
        self.plot_zoom_ROI_filt_2()

        self.alia_qdev.state.reset()
        self.clear_curves_stage_1_and_2()

    @QtCore.pyqtSlot()
    def update_newly_set_ref_V_offset(self):
        self.qlin_ref_V_offset.setText("%.3f" % self.alia.config.ref_V_offset)
        self.qlbl_ref_is_clipping.setText(self.get_clipping_text())
        # QApplication.processEvents()

        self.alia_qdev.state.reset()
        self.clear_curves_stage_1_and_2()

    @QtCore.pyqtSlot()
    def update_newly_set_ref_V_ampl(self):
        """Applies to both newly set V_ampl and V_ampl_RMS"""
        self.qlin_ref_V_ampl_RMS.setText(
            "%.3f" % self.alia.config.ref_V_ampl_RMS
        )
        self.qlin_ref_V_ampl.setText("%.3f" % self.alia.config.ref_V_ampl)
        self.qlbl_ref_is_clipping.setText(self.get_clipping_text())
        # QApplication.processEvents()

        self.alia_qdev.state.reset()
        self.clear_curves_stage_1_and_2()

    def get_clipping_text(self) -> str:
        if (
            self.alia.config.ref_is_clipping_HI
            and self.alia.config.ref_is_clipping_LO
        ):
            return "!! HIGH & LOW !!"
        elif self.alia.config.ref_is_clipping_HI:
            return "!! HIGH !!"
        elif self.alia.config.ref_is_clipping_LO:
            return "!! LOW !!"
        else:
            return ""

    @QtCore.pyqtSlot()
    def process_qpbt_fullrange_xy(self):
        self.process_qpbt_autorange_x()
        self.pi_refsig.setYRange(
            -3.3 if self.alia.config.ADC_DIFFERENTIAL else 0, 3.3, padding=0.05
        )
        self.pi_filt_1.setYRange(-3.3, 3.3, padding=0.025)
        self.pi_mixer.setYRange(-5, 5, padding=0.05)

        if self.qrbt_XR_X.isChecked():
            self.pi_XR.setYRange(-5, 5, padding=0.05)
        else:
            self.pi_XR.setYRange(0, 5, padding=0.05)
        if self.qrbt_YT_Y.isChecked():
            self.pi_YT.setYRange(-5, 5, padding=0.05)
        else:
            self.pi_YT.setYRange(-180, 180, padding=0.1)

    @QtCore.pyqtSlot()
    def process_qpbt_autorange_xy(self):
        self.process_qpbt_autorange_x()
        self.process_qpbt_autorange_y()

    @QtCore.pyqtSlot()
    def process_qpbt_autorange_x(self):
        plot_items = (
            self.pi_refsig,
            self.pi_filt_1,
            self.pi_mixer,
            self.pi_XR,
            self.pi_YT,
        )
        for pi in plot_items:
            pi.enableAutoRange("x", False)
            pi.setXRange(
                -self.alia.config.BLOCK_SIZE
                * self.alia.config.SAMPLING_PERIOD
                * 1e3,
                0.01,
                padding=0,
            )

    @QtCore.pyqtSlot()
    def process_qpbt_autorange_y(self):
        plot_items = (self.pi_refsig, self.pi_filt_1, self.pi_mixer)
        for pi in plot_items:
            pi.enableAutoRange("y", True)
            pi.enableAutoRange("y", False)
        self.autorange_y_XR()
        self.autorange_y_YT()

    @QtCore.pyqtSlot()
    def process_qrbt_XR(self):
        if self.qrbt_XR_X.isChecked():
            self.hcc_LIA_XR.curve.setPen(self.PEN_01)
            self.pi_XR.setLabel("top", "X")
        else:
            self.hcc_LIA_XR.curve.setPen(self.PEN_03)
            self.pi_XR.setLabel("top", "R")

        if (
            self.alia_qdev.worker_DAQ._paused  # pylint: disable=protected-access
        ):
            # The graphs are not being updated with the newly chosen timeseries
            # automatically because the lock-in is not running. It is safe
            # however to make a copy of the alia_qdev.state timeseries,
            # because the GUI can't interfere with the DAQ thread now that it is
            # in paused mode. Hence, we copy new data into the graph.
            if np.sum(self.alia_qdev.state.time_2) == 0:
                return

            if self.qrbt_XR_X.isChecked():
                self.hcc_LIA_XR.extendData(
                    self.alia_qdev.state.time_2, self.alia_qdev.state.X
                )
            else:
                self.hcc_LIA_XR.extendData(
                    self.alia_qdev.state.time_2, self.alia_qdev.state.R
                )
            self.hcc_LIA_XR.update()
            self.autorange_y_XR()
        else:
            # To be handled by update_chart_LIA_output
            self.pi_XR.request_autorange_y = True

    @QtCore.pyqtSlot()
    def process_qrbt_YT(self):
        if self.qrbt_YT_Y.isChecked():
            self.hcc_LIA_YT.curve.setPen(self.PEN_02)
            self.pi_YT.setLabel("top", "Y")
            self.pi_YT.setLabel("left", text="V")
        else:
            self.hcc_LIA_YT.curve.setPen(self.PEN_03)
            self.pi_YT.setLabel("top", "%s" % chr(0x398))
            self.pi_YT.setLabel("left", text="deg")

        if (
            self.alia_qdev.worker_DAQ._paused  # pylint: disable=protected-access
        ):
            # The graphs are not being updated with the newly chosen timeseries
            # automatically because the lock-in is not running. It is safe
            # however to make a copy of the alia_qdev.state timeseries,
            # because the GUI can't interfere with the DAQ thread now that it is
            # in paused mode. Hence, we copy new data into the graph.
            if np.sum(self.alia_qdev.state.time_2) == 0:
                return

            if self.qrbt_YT_Y.isChecked():
                self.hcc_LIA_YT.extendData(
                    self.alia_qdev.state.time_2, self.alia_qdev.state.Y
                )
            else:
                self.hcc_LIA_YT.extendData(
                    self.alia_qdev.state.time_2, self.alia_qdev.state.T
                )
            self.hcc_LIA_YT.update()
            self.autorange_y_YT()
        else:
            # To be handled by update_chart_LIA_output
            self.pi_YT.request_autorange_y = True

    def autorange_y_XR(self):
        if self.qrbt_XR_X.isChecked():
            self.pi_XR.setLimits(yMin=-5.25, yMax=5.25)
        else:
            self.pi_XR.setLimits(yMin=-0.1, yMax=5.25)

        if self.hcc_LIA_XR.size[0] == 0:
            if self.qrbt_XR_X.isChecked():
                self.pi_XR.setYRange(-5, 5, padding=0.05)
            else:
                self.pi_XR.setYRange(0, 5, padding=0.05)
        else:
            self.pi_XR.enableAutoRange("y", True)
            self.pi_XR.enableAutoRange("y", False)
            _XRange, YRange = self.pi_XR.viewRange()
            self.pi_XR.setYRange(YRange[0], YRange[1], padding=1.0)

    def autorange_y_YT(self):
        if self.qrbt_YT_Y.isChecked():
            self.pi_YT.setLimits(yMin=-5.25, yMax=5.25)
        else:
            self.pi_YT.setLimits(yMin=-185, yMax=185)

        if self.hcc_LIA_YT.size[0] == 0:
            if self.qrbt_YT_Y.isChecked():
                self.pi_YT.setYRange(-5, 5, padding=0.05)
            else:
                self.pi_YT.setYRange(-180, 180, padding=0.1)
        else:
            self.pi_YT.enableAutoRange("y", True)
            self.pi_YT.enableAutoRange("y", False)
            _XRange, YRange = self.pi_YT.viewRange()
            self.pi_YT.setYRange(YRange[0], YRange[1], padding=1.0)

    @QtCore.pyqtSlot()
    def update_qgrp_ADC_calibration(self):
        c = self.alia.config  # Short-hand
        self.qlbl_ADC_autocal_is_valid.setText(
            "yes" if c.ADC_autocal_is_valid else "no"
        )
        self.qlin_ADC_gaincorr.setText("%d" % c.ADC_autocal_gaincorr)
        self.qlin_ADC_offsetcorr.setText("%d" % c.ADC_autocal_offsetcorr)
        self.qpbt_ADC_store_autocal.setEnabled(c.ADC_autocal_is_valid)

    @QtCore.pyqtSlot()
    def process_qpbt_ADC_perform_autocal(self):
        reply = QMessageBox.question(
            self,
            "Perform autocalibration?",
            "You are about to perform the ADC autocalibration routine.\n"
            "It is advised to first disconnect pins [A0] and [A1].\n\n"
            # "Only implemented for single-ended mode, not differential.\n\n"
            "Are you sure you want to continue?",
            QMessageBox.Ok | QMessageBox.Cancel,
        )
        if reply == QMessageBox.Ok:
            self.alia_qdev.perform_ADC_autocalibration()

    @QtCore.pyqtSlot()
    def process_qpbt_ADC_store_autocal(self):
        reply = QMessageBox.question(
            self,
            "Store calibration in flash?",
            "You are about to store the ADC autocalibration results into the "
            "flash memory of the Arduino microcontroller.\n\n"
            "Are you sure you want to continue?",
            QMessageBox.Ok | QMessageBox.Cancel,
        )
        if reply == QMessageBox.Ok:
            self.alia_qdev.store_ADC_autocalibration()

    @QtCore.pyqtSlot()
    def confirm_ADC_autocalibration_was_stored(self):
        QMessageBox.information(
            self,
            "Store calibration in flash: succesful.",
            "The ADC autocalibration results have been stored succesfully "
            "in the microcontroller flash.",
            QMessageBox.Ok,
        )

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    #   Update graph routines
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    @QtCore.pyqtSlot()
    def update_curves(self):
        for curve in self.all_curves:
            curve.update()

        if self.pi_XR.request_autorange_y == True:
            self.pi_XR.request_autorange_y = False
            self.autorange_y_XR()

        if self.pi_YT.request_autorange_y == True:
            self.pi_YT.request_autorange_y = False
            self.autorange_y_YT()

    @QtCore.pyqtSlot()
    def update_plot_filt_1_resp(self):
        freqz = self.alia_qdev.firf_1_sig_I.freqz
        if isinstance(self.curve_filt_1_resp, pg.PlotCurveItem):
            self.curve_filt_1_resp.setFillLevel(np.min(freqz.ampl_dB))
        self.curve_filt_1_resp.setData(freqz.freq_Hz, freqz.ampl_dB)

    @QtCore.pyqtSlot()
    def update_plot_filt_2_resp(self):
        freqz = self.alia_qdev.firf_2_mix_X.freqz
        if isinstance(self.curve_filt_2_resp, pg.PlotCurveItem):
            self.curve_filt_2_resp.setFillLevel(np.min(freqz.ampl_dB))
        self.curve_filt_2_resp.setData(freqz.freq_Hz, freqz.ampl_dB)

    @QtCore.pyqtSlot()
    def plot_zoom_ROI_filt_1(self):
        self.setUpdatesEnabled(False)
        freqz = self.alia_qdev.firf_1_sig_I.freqz
        self.pi_filt_1_resp.setXRange(
            freqz.freq_Hz__ROI_start,
            freqz.freq_Hz__ROI_end,
            padding=0.02,
        )
        self.pi_filt_1_resp.enableAutoRange("y", True)
        self.pi_filt_1_resp.enableAutoRange("y", False)
        QApplication.processEvents()
        self.setUpdatesEnabled(True)

    @QtCore.pyqtSlot()
    def plot_zoom_ROI_filt_2(self):
        self.setUpdatesEnabled(False)
        freqz = self.alia_qdev.firf_2_mix_X.freqz
        self.pi_filt_2_resp.setXRange(
            freqz.freq_Hz__ROI_start,
            freqz.freq_Hz__ROI_end,
            padding=0.02,
        )
        self.pi_filt_2_resp.enableAutoRange("y", True)
        self.pi_filt_2_resp.enableAutoRange("y", False)
        QApplication.processEvents()
        self.setUpdatesEnabled(True)

    def plot_zoom_x(self, pi_plot: pg.PlotItem, xRangeLo, xRangeHi):
        self.setUpdatesEnabled(False)
        pi_plot.setXRange(xRangeLo, xRangeHi, padding=0.02)
        pi_plot.enableAutoRange("y", True)
        pi_plot.enableAutoRange("y", False)
        QApplication.processEvents()
        self.setUpdatesEnabled(True)


if __name__ == "__main__":
    exec(open("DvG_Arduino_lockin_amp.py").read())  # pylint: disable=exec-used
