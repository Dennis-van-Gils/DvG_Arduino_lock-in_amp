#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Arduino lock-in amplifier
"""
__author__      = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__         = "https://github.com/Dennis-van-Gils/DvG_Arduino_lock-in_amp"
__date__        = "01-06-2019"
__version__     = "1.0.0"

from PyQt5 import QtCore, QtGui
from PyQt5 import QtWidgets as QtWid
from PyQt5.QtCore import QDateTime
import pyqtgraph as pg
import numpy as np
import psutil

from DvG_pyqt_ChartHistory import ChartHistory
from DvG_pyqt_BufferedPlot import BufferedPlot
from DvG_pyqt_controls     import Legend_box
from DvG_pyqt_FileLogger   import FileLogger
from DvG_debug_functions   import dprint

import DvG_dev_Arduino_lockin_amp__fun_serial as lockin_functions
import DvG_dev_Arduino_lockin_amp__pyqt_lib   as lockin_pyqt_lib
from DvG_Buffered_FIR_Filter__GUI import Filter_design_GUI

# Monkey-patch errors in pyqtgraph v0.10
import pyqtgraph.exporters
import DvG_monkeypatch_pyqtgraph as pgmp
pg.PlotCurveItem.paintGL = pgmp.PlotCurveItem_paintGL
pg.exporters.ImageExporter.export = pgmp.ImageExporter_export

# Constants
UPDATE_INTERVAL_WALL_CLOCK = 50  # 50 [ms]

try:
    import OpenGL.GL as gl
    pg.setConfigOptions(useOpenGL=True)
    pg.setConfigOptions(enableExperimental=False)
    pg.setConfigOptions(antialias=False)
    print("OpenGL hardware acceleration enabled.")
    USING_OPENGL = True
except:
    pg.setConfigOptions(useOpenGL=False)
    pg.setConfigOptions(enableExperimental=False)
    pg.setConfigOptions(antialias=False)
    print("WARNING: Could not enable OpenGL hardware acceleration.")
    print("Check if prerequisite 'PyOpenGL' library is installed.")
    USING_OPENGL = False

# Stylesheets
# Note: (Top Right Bottom Left)
    
COLOR_BG           = "rgb(240, 240, 240)"
COLOR_LED_RED      = "rgb(0  , 238, 118)"
COLOR_LED_GREEN    = "rgb(205, 92 , 92 )"
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
            "min-width: 39ex;"
            "padding: 6px;} "
        "QTabBar::tab:hover {"
            "background: " + COLOR_HOVER + ";"
            "border: 2px solid " + COLOR_HOVER_BORDER + ";"
            "border-bottom-color: " + COLOR_HOVER + ";"
            "border-top-left-radius: 4px;"
            "border-top-right-radius: 4px;"
            "min-width: 42ex;"
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
            "background-color: " + COLOR_LED_RED + ";"
            "color: black;" +
            "font-weight: normal;"
            "border-style: inset;}")

SS_LED_RECT = (
        "QPushButton {"
            "background-color: " + COLOR_LED_GREEN + ";"
            "border: 1px solid grey;"
            "min-height: 30px;"
            "min-width: 60px;"
            "max-width: 60px}"
        "QPushButton::disabled {"
            "border-radius: 10px;"
            "color: black}"
        "QPushButton::checked {"
            "background-color: " + COLOR_LED_RED + "}")

def create_Toggle_button(text='', minimumHeight=40):
    button = QtWid.QPushButton(text, checkable=True)
    button.setStyleSheet(SS_TOGGLE_BUTTON)
    if minimumHeight is not None:
        button.setMinimumHeight(minimumHeight)
    return button

def create_LED_indicator_rect(initial_state=False, text=''):
    button = QtWid.QPushButton(text, checkable=True, enabled=False)
    button.setStyleSheet(SS_LED_RECT)
    button.setChecked(initial_state)
    return button

# ------------------------------------------------------------------------------
#   MainWindow
# ------------------------------------------------------------------------------

class MainWindow(QtWid.QWidget):    
    def __init__(self,
                 lockin     : lockin_functions.Arduino_lockin_amp,
                 lockin_pyqt: lockin_pyqt_lib.Arduino_lockin_amp_pyqt,
                 file_logger: FileLogger,
                 parent=None,
                 **kwargs):
        super().__init__(parent, **kwargs)
        
        self.lockin = lockin
        self.lockin_pyqt = lockin_pyqt
        self.file_logger = file_logger
        
        self.prev_time_CPU_load = QDateTime.currentDateTime();
        self.boost_fps_graphing = False
        
        self.setGeometry(250, 50, 1200, 960)
        self.setWindowTitle("Arduino lock-in amplifier")
        self.setStyleSheet(SS_TEXTBOX_READ_ONLY + SS_GROUP + SS_HOVER + 
                           SS_TABS)
        
        """ # Experimental
        with open('darkorange.stylesheet', 'r') as file:
            style = file.read()
        self.setStyleSheet(style)
        """
        
        # Define styles for plotting curves
        self.PEN_01 = pg.mkPen(color=[255, 30 , 180], width=3)
        self.PEN_02 = pg.mkPen(color=[255, 255, 90 ], width=3)
        self.PEN_03 = pg.mkPen(color=[0  , 255, 255], width=3)
        self.PEN_04 = pg.mkPen(color=[255, 255, 255], width=3)        
        self.BRUSH_03 = pg.mkBrush(0, 255, 255, 64)
        
        # Fonts
        FONT_MONOSPACE = QtGui.QFont("Courier")
        FONT_MONOSPACE.setFamily("Monospace")
        FONT_MONOSPACE.setStyleHint(QtGui.QFont.Monospace)
        
        # Column width left of timeseries graphs
        LEFT_COLUMN_WIDTH = 134
        
        # Textbox widths for fitting N 'x' characters using the current font
        e = QtGui.QLineEdit()
        ex8  = (8 + 8 * e.fontMetrics().width('x') + 
                e.textMargins().left()     + e.textMargins().right() + 
                e.contentsMargins().left() + e.contentsMargins().right())
        ex10 = (8 + 10 * e.fontMetrics().width('x') + 
                e.textMargins().left()     + e.textMargins().right() + 
                e.contentsMargins().left() + e.contentsMargins().right())
        ex12 = (8 + 12 * e.fontMetrics().width('x') + 
                e.textMargins().left()     + e.textMargins().right() + 
                e.contentsMargins().left() + e.contentsMargins().right())
        del e

        def Header(): pass # Spider IDE outline bookmark
        # -----------------------------------
        # -----------------------------------
        #   FRAME: Header
        # -----------------------------------
        # -----------------------------------

        # Left box
        self.qlbl_update_counter = QtWid.QLabel("0")
        self.qlbl_sample_rate = QtWid.QLabel("SAMPLE RATE: %.2f Hz" %
                                             (1/lockin.config.ISR_CLOCK))
        #self.qlbl_buffer_size = QtWid.QLabel("BUFFER SIZE  : %i" %
        #                                     lockin.config.BUFFER_SIZE)
        self.qlbl_CPU_load = QtWid.QLabel("CPU: nan%%")
        self.qlbl_DAQ_rate = QtWid.QLabel("Buffers/s: nan")
        self.qlbl_DAQ_rate.setMinimumWidth(100)

        vbox_left = QtWid.QVBoxLayout()
        vbox_left.addWidget(self.qlbl_sample_rate)
        vbox_left.addWidget(self.qlbl_CPU_load)
        vbox_left.addStretch(1)
        vbox_left.addWidget(self.qlbl_DAQ_rate)
        vbox_left.addWidget(self.qlbl_update_counter)

         # Middle box
        self.qlbl_title = QtWid.QLabel("Arduino lock-in amplifier",
                font=QtGui.QFont("Palatino", 14, weight=QtGui.QFont.Bold))
        self.qlbl_title.setAlignment(QtCore.Qt.AlignCenter)
        self.qlbl_cur_date_time = QtWid.QLabel("00-00-0000    00:00:00")
        self.qlbl_cur_date_time.setAlignment(QtCore.Qt.AlignCenter)
        self.qpbt_record = create_Toggle_button(
                "Click to start recording to file", minimumHeight=40)
        self.qpbt_record.clicked.connect(self.process_qpbt_record)

        vbox_middle = QtWid.QVBoxLayout()
        vbox_middle.addWidget(self.qlbl_title)
        vbox_middle.addWidget(self.qlbl_cur_date_time)
        vbox_middle.addWidget(self.qpbt_record)

        # Right box
        self.qpbt_exit = QtWid.QPushButton("Exit")
        self.qpbt_exit.clicked.connect(self.close)
        self.qpbt_exit.setMinimumHeight(30)

        p = {'alignment': QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter}
        vbox_right = QtWid.QVBoxLayout()
        vbox_right.addWidget(self.qpbt_exit)
        vbox_right.addStretch(1)
        self.qlbl_GitHub = QtWid.QLabel("<a href=\"%s\">GitHub source</a>" %
                                        __url__, **p)
        self.qlbl_GitHub.setTextFormat(QtCore.Qt.RichText)
        self.qlbl_GitHub.setTextInteractionFlags(QtCore.Qt.TextBrowserInteraction)
        self.qlbl_GitHub.setOpenExternalLinks(True)
        vbox_right.addWidget(self.qlbl_GitHub)
        vbox_right.addWidget(QtWid.QLabel(__author__, **p))
        vbox_right.addWidget(QtWid.QLabel(__date__, **p))

        # Round up frame
        hbox_header = QtWid.QHBoxLayout()
        hbox_header.addLayout(vbox_left)
        hbox_header.addStretch(1)
        hbox_header.addLayout(vbox_middle)
        hbox_header.addStretch(1)
        hbox_header.addLayout(vbox_right)
        
        def Tabs(): pass # Spider IDE outline bookmark
        # -----------------------------------
        # -----------------------------------
        #   FRAME: Tabs
        # -----------------------------------
        # -----------------------------------
        
        self.tabs = QtWid.QTabWidget()
        self.tab_main  = QtWid.QWidget()
        self.tab_mixer = QtWid.QWidget()
        self.tab_power_spectrum  = QtWid.QWidget()
        self.tab_filter_1_design = QtWid.QWidget()
        self.tab_filter_2_design = QtWid.QWidget()
        self.tab_diagram = QtWid.QWidget()
        self.tab_settings = QtWid.QWidget()
        
        self.tabs.addTab(self.tab_main           , "Main")
        self.tabs.addTab(self.tab_mixer          , "Mixer")
        self.tabs.addTab(self.tab_power_spectrum , "Spectrum")
        self.tabs.addTab(self.tab_filter_1_design, "Filter design @ sig_I")
        self.tabs.addTab(self.tab_filter_2_design, "Filter design @ mix_X/Y")
        self.tabs.addTab(self.tab_diagram        , "Diagram")
        self.tabs.addTab(self.tab_settings       , "Settings")
        
        def Sidebar(): pass # Spider IDE outline bookmark
        # -----------------------------------
        # -----------------------------------
        #   FRAME: Sidebar
        # -----------------------------------
        # -----------------------------------

        # On/off
        self.qpbt_ENA_lockin = create_Toggle_button("Lock-in OFF")
        self.qpbt_ENA_lockin.clicked.connect(self.process_qpbt_ENA_lockin)
        
        # QGROUP: Reference signal
        p1 = {'maximumWidth': ex8, 'minimumWidth': ex8}
        p2 = {**p1, 'readOnly': True}
        self.qlin_set_ref_freq = (
                QtWid.QLineEdit("%.2f" % lockin.config.ref_freq, **p1))
        self.qlin_read_ref_freq = (
                QtWid.QLineEdit("%.2f" % lockin.config.ref_freq, **p2))
        self.qlin_set_ref_V_offset = (
                QtWid.QLineEdit("%.3f" % lockin.config.ref_V_offset, **p1))
        self.qlin_read_ref_V_offset = (
                QtWid.QLineEdit("%.3f" % lockin.config.ref_V_offset, **p2))
        self.qlin_set_ref_V_ampl = (
                QtWid.QLineEdit("%.3f" % lockin.config.ref_V_ampl, **p1))
        self.qlin_read_ref_V_ampl = (
                QtWid.QLineEdit("%.3f" % lockin.config.ref_V_ampl, **p2))

        self.qlin_set_ref_freq.editingFinished.connect(
                self.process_qlin_set_ref_freq)
        self.qlin_set_ref_V_offset.editingFinished.connect(
                self.process_qlin_set_ref_V_offset)
        self.qlin_set_ref_V_ampl.editingFinished.connect(
                self.process_qlin_set_ref_V_ampl)
        
        p  = {'alignment': QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight}
        p2 = {'alignment': QtCore.Qt.AlignVCenter | QtCore.Qt.AlignHCenter}
        i = 0;
        grid = QtWid.QGridLayout(spacing=4)
        grid.addWidget(QtWid.QLabel("ref_X*: cosine wave", **p2)
                                                   , i, 0, 1, 4); i+=1
        grid.addItem(QtWid.QSpacerItem(0, 4)       , i, 0); i+=1
        grid.addWidget(QtWid.QLabel("Set")         , i, 1)
        grid.addWidget(QtWid.QLabel("Read")        , i, 2); i+=1
        grid.addWidget(QtWid.QLabel("freq:", **p)  , i, 0)
        grid.addWidget(self.qlin_set_ref_freq      , i, 1)
        grid.addWidget(self.qlin_read_ref_freq     , i, 2)
        grid.addWidget(QtWid.QLabel("Hz")          , i, 3); i+=1
        grid.addWidget(QtWid.QLabel("offset:", **p), i, 0)
        grid.addWidget(self.qlin_set_ref_V_offset  , i, 1)
        grid.addWidget(self.qlin_read_ref_V_offset , i, 2)
        grid.addWidget(QtWid.QLabel("V")           , i, 3); i+=1
        grid.addWidget(QtWid.QLabel("ampl:", **p)  , i, 0)
        grid.addWidget(self.qlin_set_ref_V_ampl    , i, 1)
        grid.addWidget(self.qlin_read_ref_V_ampl   , i, 2)
        grid.addWidget(QtWid.QLabel("V")           , i, 3)
        
        qgrp_refsig = QtWid.QGroupBox("Reference signal")
        qgrp_refsig.setLayout(grid)
        
        # QGROUP: Connections
        grid = QtWid.QGridLayout(spacing=4)
        grid.addWidget(QtWid.QLabel("Output: ref_X*\n"
                                    "  [ 0.0, %.1f] V\n"
                                    "  pin A0 wrt GND" %
                                    lockin.config.A_REF,
                                    font=FONT_MONOSPACE), 0, 0)
        grid.addWidget(QtWid.QLabel("Input: sig_I\n"
                                    "  [-%.1f, %.1f] V\n"
                                    "  pin A1(+), A2(-)" %
                                    (lockin.config.A_REF, lockin.config.A_REF),
                                    font=FONT_MONOSPACE), 1, 0)
        
        qgrp_connections = QtWid.QGroupBox("Analog connections")
        qgrp_connections.setLayout(grid)        

        # QGROUP: Axes controls
        self.qpbt_fullrange_xy = QtWid.QPushButton("Full range")
        self.qpbt_fullrange_xy.clicked.connect(self.process_qpbt_fullrange_xy)
        self.qpbt_autorange_xy = QtWid.QPushButton("Auto range")
        self.qpbt_autorange_xy.clicked.connect(self.process_qpbt_autorange_xy)
        self.qpbt_autorange_x = QtWid.QPushButton("Auto x", maximumWidth=80)
        self.qpbt_autorange_x.clicked.connect(self.process_qpbt_autorange_x)
        self.qpbt_autorange_y = QtWid.QPushButton("Auto y", maximumWidth=80)
        self.qpbt_autorange_y.clicked.connect(self.process_qpbt_autorange_y)
        
        grid = QtWid.QGridLayout(spacing=4)
        grid.addWidget(self.qpbt_fullrange_xy, 0, 0, 1, 2)
        grid.addWidget(self.qpbt_autorange_xy, 2, 0, 1, 2)
        grid.addWidget(self.qpbt_autorange_x , 3, 0)
        grid.addWidget(self.qpbt_autorange_y , 3, 1)
        
        qgrp_axes_controls = QtWid.QGroupBox("Zoom timeseries")
        qgrp_axes_controls.setLayout(grid)

        # QGROUP: Filter deques settled?
        self.LED_filt_1_deque_settled = create_LED_indicator_rect(False, 'NO')
        self.LED_filt_2_deque_settled = create_LED_indicator_rect(False, 'NO')
        
        grid = QtWid.QGridLayout(spacing=4)
        grid.addWidget(QtWid.QLabel("Filter @ sig_I")  , 0, 0)
        grid.addWidget(self.LED_filt_1_deque_settled   , 0, 1)
        grid.addWidget(QtWid.QLabel("Filter @ mix_X/Y"), 1, 0)
        grid.addWidget(self.LED_filt_2_deque_settled   , 1, 1)
        
        qgrp_settling = QtWid.QGroupBox("Buffers settled?")
        qgrp_settling.setLayout(grid)

        # Round up frame
        vbox_sidebar = QtWid.QVBoxLayout()
        vbox_sidebar.addItem(QtWid.QSpacerItem(0, 30))
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

        def Reference_and_signal(): pass # Spider IDE outline bookmark
        # -----------------------------------
        # -----------------------------------
        #   FRAME: Reference and signal
        # -----------------------------------
        # -----------------------------------
        
        # Chart: Readings
        self.gw_refsig = pg.GraphicsWindow()
        self.gw_refsig.setBackground([20, 20, 20])
        self.pi_refsig = self.gw_refsig.addPlot()

        p = {'color': '#BBB', 'font-size': '10pt'}
        self.pi_refsig.showGrid(x=1, y=1)
        self.pi_refsig.setTitle('Readings', **p)
        self.pi_refsig.setLabel('bottom', text='time [ms]', **p)
        self.pi_refsig.setLabel('left', text='voltage [V]', **p)
        self.pi_refsig.setXRange(-lockin.config.BUFFER_SIZE * 
                                 lockin.config.ISR_CLOCK * 1e3,
                                 0, padding=0.01)
        self.pi_refsig.setYRange(-3.3, 3.3, padding=0.05)
        self.pi_refsig.setAutoVisible(x=True, y=True)
        self.pi_refsig.setClipToView(True)
        self.pi_refsig.setLimits(xMin=-(lockin.config.BUFFER_SIZE + 1) * 
                                 lockin.config.ISR_CLOCK * 1e3,
                                 xMax=0)

        self.CH_ref_X = ChartHistory(lockin.config.BUFFER_SIZE,
                                     self.pi_refsig.plot(pen=self.PEN_01))
        self.CH_ref_Y = ChartHistory(lockin.config.BUFFER_SIZE,
                                     self.pi_refsig.plot(pen=self.PEN_02))
        self.CH_sig_I = ChartHistory(lockin.config.BUFFER_SIZE,
                                     self.pi_refsig.plot(pen=self.PEN_03))
        self.CH_ref_X.x_axis_divisor = 1000     # From [us] to [ms]
        self.CH_ref_Y.x_axis_divisor = 1000     # From [us] to [ms]
        self.CH_sig_I.x_axis_divisor = 1000     # From [us] to [ms]
        self.CHs_refsig = [self.CH_ref_X, self.CH_ref_Y, self.CH_sig_I]

        # QGROUP: Readings
        self.legend_box_refsig = Legend_box(
                text=["ref_X*", "ref_Y*", "sig_I"],
                pen=[self.PEN_01, self.PEN_02, self.PEN_03],
                checked=[False, False, True])
        ([chkb.clicked.connect(self.process_chkbs_legend_box_refsig) for chkb
          in self.legend_box_refsig.chkbs])
        
        p1 = {'maximumWidth': ex12, 'minimumWidth': ex12, 'readOnly': True}
        p2 = {'maximumWidth': ex8 , 'minimumWidth': ex8 , 'readOnly': True}
        self.qlin_time      = QtWid.QLineEdit(**p1)
        self.qlin_sig_I_max = QtWid.QLineEdit(**p2)
        self.qlin_sig_I_min = QtWid.QLineEdit(**p2)
        self.qlin_sig_I_avg = QtWid.QLineEdit(**p2)
        self.qlin_sig_I_std = QtWid.QLineEdit(**p2)

        i = 0
        grid = QtWid.QGridLayout(spacing=4)
        grid.addWidget(self.qlin_time             , i, 0, 1, 2)
        grid.addWidget(QtWid.QLabel("us")         , i, 2)      ; i+=1
        grid.addItem(QtWid.QSpacerItem(0, 6)      , i, 0)      ; i+=1
        grid.addLayout(self.legend_box_refsig.grid, i, 0, 1, 3); i+=1
        grid.addItem(QtWid.QSpacerItem(0, 6)      , i, 0)      ; i+=1
        grid.addWidget(QtWid.QLabel("max")        , i, 0)
        grid.addWidget(self.qlin_sig_I_max        , i, 1)
        grid.addWidget(QtWid.QLabel("V")          , i, 2)      ; i+=1
        grid.addWidget(QtWid.QLabel("min")        , i, 0)
        grid.addWidget(self.qlin_sig_I_min        , i, 1)
        grid.addWidget(QtWid.QLabel("V")          , i, 2)      ; i+=1
        grid.addItem(QtWid.QSpacerItem(0, 4)      , i, 0)      ; i+=1
        grid.addWidget(QtWid.QLabel("avg")        , i, 0)
        grid.addWidget(self.qlin_sig_I_avg        , i, 1)
        grid.addWidget(QtWid.QLabel("V")          , i, 2)      ; i+=1
        grid.addWidget(QtWid.QLabel("std")        , i, 0)
        grid.addWidget(self.qlin_sig_I_std        , i, 1)
        grid.addWidget(QtWid.QLabel("V")          , i, 2)      ; i+=1
        grid.setAlignment(QtCore.Qt.AlignTop)

        qgrp_readings = QtWid.QGroupBox("Readings")
        qgrp_readings.setLayout(grid)

        def LIA_output(): pass # Spider IDE outline bookmark
        # -----------------------------------
        # -----------------------------------
        #   FRAME: LIA output
        # -----------------------------------
        # -----------------------------------

        # Charts: (X or R) and (Y or T)
        self.gw_XRYT = pg.GraphicsWindow()
        self.gw_XRYT.setBackground([20, 20, 20])
        self.pi_XR = self.gw_XRYT.addPlot()
        self.pi_YT = self.gw_XRYT.addPlot()
        
        p = {'color': '#BBB', 'font-size': '10pt'}
        self.pi_XR.showGrid(x=1, y=1)
        self.pi_XR.setTitle('R', **p)
        self.pi_XR.setLabel('bottom', text='time [ms]', **p)
        self.pi_XR.setLabel('left', text='voltage [V]', **p)
        self.pi_XR.setXRange(-lockin.config.BUFFER_SIZE *
                             lockin.config.ISR_CLOCK * 1e3,
                             0, padding=0.01)
        self.pi_XR.setYRange(0, 5, padding=0.05)
        self.pi_XR.setAutoVisible(x=True, y=True)
        self.pi_XR.setClipToView(True)
        self.pi_XR.request_autorange_y = False
        self.pi_XR.setLimits(xMin=-(lockin.config.BUFFER_SIZE + 1) * 
                             lockin.config.ISR_CLOCK * 1e3,
                             xMax=0)
        
        self.pi_YT.showGrid(x=1, y=1)
        self.pi_YT.setTitle('\u0398', **p)
        self.pi_YT.setLabel('bottom', text='time [ms]', **p)
        self.pi_YT.setLabel('left', text='phase [deg]', **p)
        self.pi_YT.setXRange(-lockin.config.BUFFER_SIZE *
                             lockin.config.ISR_CLOCK * 1e3,
                             0, padding=0.01)
        self.pi_YT.setYRange(-90, 90, padding=0.1)
        self.pi_YT.setAutoVisible(x=True, y=True)
        self.pi_YT.setClipToView(True)
        self.pi_YT.request_autorange_y = False
        self.pi_YT.setLimits(xMin=-(lockin.config.BUFFER_SIZE + 1) * 
                             lockin.config.ISR_CLOCK * 1e3,
                             xMax=0)

        self.CH_LIA_XR = ChartHistory(lockin.config.BUFFER_SIZE,
                                      self.pi_XR.plot(pen=self.PEN_03))
        self.CH_LIA_YT = ChartHistory(lockin.config.BUFFER_SIZE,
                                      self.pi_YT.plot(pen=self.PEN_03))
        self.CH_LIA_XR.x_axis_divisor = 1000     # From [us] to [ms]
        self.CH_LIA_YT.x_axis_divisor = 1000     # From [us] to [ms]
        self.CHs_LIA_output = [self.CH_LIA_XR, self.CH_LIA_YT]
        
        # QGROUP: Show X-Y or R-Theta
        self.qrbt_XR_X = QtWid.QRadioButton("X")
        self.qrbt_XR_R = QtWid.QRadioButton("R", checked=True)
        self.qrbt_YT_Y = QtWid.QRadioButton("Y")
        self.qrbt_YT_T = QtWid.QRadioButton("\u0398", checked=True)
        self.qrbt_XR_X.clicked.connect(self.process_qrbt_XR)
        self.qrbt_XR_R.clicked.connect(self.process_qrbt_XR)
        self.qrbt_YT_Y.clicked.connect(self.process_qrbt_YT)
        self.qrbt_YT_T.clicked.connect(self.process_qrbt_YT)
        
        vbox = QtWid.QVBoxLayout(spacing=4)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.addWidget(self.qrbt_XR_X)
        vbox.addWidget(self.qrbt_XR_R)
        qgrp_XR = QtWid.QGroupBox(flat=True)
        qgrp_XR.setLayout(vbox)
        
        vbox = QtWid.QVBoxLayout(spacing=4)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.addWidget(self.qrbt_YT_Y)
        vbox.addWidget(self.qrbt_YT_T)
        qgrp_YT = QtWid.QGroupBox(flat=True)
        qgrp_YT.setLayout(vbox)
        
        hbox = QtWid.QHBoxLayout(spacing=4)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(qgrp_XR)
        hbox.addWidget(qgrp_YT)
        
        p = {'maximumWidth': ex8, 'minimumWidth': ex8, 'readOnly': True}
        self.qlin_X_avg = QtWid.QLineEdit(**p)
        self.qlin_Y_avg = QtWid.QLineEdit(**p)
        self.qlin_R_avg = QtWid.QLineEdit(**p)
        self.qlin_T_avg = QtWid.QLineEdit(**p)
        
        i = 0
        grid = QtWid.QGridLayout(spacing=4)
        grid.addLayout(hbox                      , i, 0, 1, 3); i+=1
        grid.addItem(QtWid.QSpacerItem(0, 8)     , i, 0); i+=1
        grid.addWidget(QtWid.QLabel("avg X")     , i, 0)
        grid.addWidget(self.qlin_X_avg           , i, 1)
        grid.addWidget(QtWid.QLabel("V")         , i, 2); i+=1
        grid.addWidget(QtWid.QLabel("avg Y")     , i, 0)
        grid.addWidget(self.qlin_Y_avg           , i, 1)
        grid.addWidget(QtWid.QLabel("V")         , i, 2); i+=1
        grid.addItem(QtWid.QSpacerItem(0, 8)     , i, 0); i+=1
        grid.addWidget(QtWid.QLabel("avg R")     , i, 0)
        grid.addWidget(self.qlin_R_avg           , i, 1)
        grid.addWidget(QtWid.QLabel("V")         , i, 2); i+=1
        grid.addWidget(QtWid.QLabel("avg \u0398"), i, 0)
        grid.addWidget(self.qlin_T_avg           , i, 1)
        grid.addWidget(QtWid.QLabel("\u00B0")    , i, 2)
        grid.setAlignment(QtCore.Qt.AlignTop)
        
        qgrp_XRYT = QtWid.QGroupBox("X/R, Y/\u0398")
        qgrp_XRYT.setLayout(grid)
     
        # -----------------------------------
        # -----------------------------------
        #   Round up tab page 'Main'
        # -----------------------------------
        # -----------------------------------
        
        grid = QtWid.QGridLayout(spacing=0)
        grid.addWidget(qgrp_readings , 0, 0, QtCore.Qt.AlignTop)
        grid.addWidget(self.gw_refsig, 0, 1)
        grid.addWidget(qgrp_XRYT     , 1, 0, QtCore.Qt.AlignTop)
        grid.addWidget(self.gw_XRYT  , 1, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnMinimumWidth(0, LEFT_COLUMN_WIDTH)
        
        self.tab_main.setLayout(grid)
        
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        #
        #   TAB PAGE: Mixer
        #
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        
        def Filter_1_output(): pass # Spider IDE outline bookmark
        # -----------------------------------
        # -----------------------------------
        #   FRAME: Filter 1 output
        # -----------------------------------
        # -----------------------------------
        
        # Chart: Filter @ sig_I
        self.gw_filt_1 = pg.GraphicsWindow()
        self.gw_filt_1.setBackground([20, 20, 20])
        self.pi_filt_1 = self.gw_filt_1.addPlot()
        
        p = {'color': '#BBB', 'font-size': '10pt'}
        self.pi_filt_1.showGrid(x=1, y=1)
        self.pi_filt_1.setTitle('Filter @ sig_I', **p)
        self.pi_filt_1.setLabel('bottom', text='time [ms]', **p)
        self.pi_filt_1.setLabel('left', text='voltage [V]', **p)
        self.pi_filt_1.setXRange(-lockin.config.BUFFER_SIZE *
                                    lockin.config.ISR_CLOCK * 1e3,
                                    0, padding=0.01)
        self.pi_filt_1.setYRange(-5, 5, padding=0.05)
        self.pi_filt_1.setAutoVisible(x=True, y=True)
        self.pi_filt_1.setClipToView(True)
        self.pi_filt_1.setLimits(xMin=-(lockin.config.BUFFER_SIZE + 1) *
                                  lockin.config.ISR_CLOCK * 1e3,
                                  xMax=0)
        
        self.CH_filt_1_in  = ChartHistory(lockin.config.BUFFER_SIZE,
                                          self.pi_filt_1.plot(pen=self.PEN_03))
        self.CH_filt_1_out = ChartHistory(lockin.config.BUFFER_SIZE,
                                          self.pi_filt_1.plot(pen=self.PEN_04))
        self.CH_filt_1_in.x_axis_divisor = 1000     # From [us] to [ms]
        self.CH_filt_1_out.x_axis_divisor = 1000    # From [us] to [ms]
        self.CHs_filt_1 = [self.CH_filt_1_in, self.CH_filt_1_out]
        
        # QGROUP: Filter output
        self.legend_box_filt_1 = Legend_box(text=['sig_I', 'filt_I'],
                                             pen=[self.PEN_03, self.PEN_04])
        ([chkb.clicked.connect(self.process_chkbs_legend_box_filt_1) for chkb
          in self.legend_box_filt_1.chkbs])
    
        qgrp_filt_1 = QtWid.QGroupBox("Filter @ sig_I")
        qgrp_filt_1.setLayout(self.legend_box_filt_1.grid)
        
        def Mixer(): pass # Spider IDE outline bookmark
        # -----------------------------------
        # -----------------------------------
        #   FRAME: Mixer
        # -----------------------------------
        # -----------------------------------
        
        # Chart: Mixer
        self.gw_mixer = pg.GraphicsWindow()
        self.gw_mixer.setBackground([20, 20, 20])
        self.pi_mixer = self.gw_mixer.addPlot()
        
        p = {'color': '#BBB', 'font-size': '10pt'}
        self.pi_mixer.showGrid(x=1, y=1)
        self.pi_mixer.setTitle('Mixer', **p)
        self.pi_mixer.setLabel('bottom', text='time [ms]', **p)
        self.pi_mixer.setLabel('left', text='voltage [V]', **p)
        self.pi_mixer.setXRange(-lockin.config.BUFFER_SIZE *
                                lockin.config.ISR_CLOCK * 1e3,
                                 0, padding=0.01)
        self.pi_mixer.setYRange(-5, 5, padding=0.05)
        self.pi_mixer.setAutoVisible(x=True, y=True)
        self.pi_mixer.setClipToView(True)
        self.pi_mixer.setLimits(xMin=-(lockin.config.BUFFER_SIZE + 1) *
                                lockin.config.ISR_CLOCK * 1e3,
                                xMax=0)
        
        self.CH_mix_X = ChartHistory(lockin.config.BUFFER_SIZE,
                                     self.pi_mixer.plot(pen=self.PEN_01))
        self.CH_mix_Y = ChartHistory(lockin.config.BUFFER_SIZE,
                                     self.pi_mixer.plot(pen=self.PEN_02))
        self.CH_mix_X.x_axis_divisor = 1000     # From [us] to [ms]
        self.CH_mix_Y.x_axis_divisor = 1000     # From [us] to [ms]
        self.CHs_mixer = [self.CH_mix_X, self.CH_mix_Y]
        
        # QGROUP: Mixer
        self.legend_box_mixer = Legend_box(text=['mix_X', 'mix_Y'],
                                           pen=[self.PEN_01, self.PEN_02])
        ([chkb.clicked.connect(self.process_chkbs_legend_box_mixer) for chkb
          in self.legend_box_mixer.chkbs])
    
        qgrp_mixer = QtWid.QGroupBox("Mixer")
        qgrp_mixer.setLayout(self.legend_box_mixer.grid)
        
        # -----------------------------------
        # -----------------------------------
        #   Round up tab page 'Mixer'
        # -----------------------------------
        # -----------------------------------

        grid = QtWid.QGridLayout(spacing=0)
        grid.addWidget(qgrp_filt_1   , 0, 0, QtCore.Qt.AlignTop)
        grid.addWidget(self.gw_filt_1, 0, 1)
        grid.addWidget(qgrp_mixer    , 1, 0, QtCore.Qt.AlignTop)
        grid.addWidget(self.gw_mixer , 1, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnMinimumWidth(0, LEFT_COLUMN_WIDTH)
        
        self.tab_mixer.setLayout(grid)
        
        def Power_spectrum(): pass # Spider IDE outline bookmark
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        #
        #   TAB PAGE: Power spectrum
        #
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        
        # Plot: Power spectrum
        self.gw_PS = pg.GraphicsWindow()
        self.gw_PS.setBackground([20, 20, 20])        
        self.pi_PS = self.gw_PS.addPlot()
        
        p = {'color': '#BBB', 'font-size': '10pt'}
        self.pi_PS.showGrid(x=1, y=1)
        self.pi_PS.setTitle('Power spectrum (Welch)', **p)
        self.pi_PS.setLabel('bottom', text='frequency [Hz]', **p)
        self.pi_PS.setLabel('left', text='power [dBV]', **p)
        self.pi_PS.setAutoVisible(x=True, y=True)
        self.pi_PS.setXRange(0, self.lockin.config.F_Nyquist, padding=0.02)
        self.pi_PS.setYRange(-110, 0, padding=0.02)
        self.pi_PS.setClipToView(True)
        self.pi_PS.setLimits(xMin=0, xMax=self.lockin.config.F_Nyquist)
        
        self.BP_PS_1 = BufferedPlot(self.pi_PS.plot(pen=self.PEN_03))
        self.BP_PS_2 = BufferedPlot(self.pi_PS.plot(pen=self.PEN_04))
        self.BP_PS_3 = BufferedPlot(self.pi_PS.plot(pen=self.PEN_01))
        self.BP_PS_4 = BufferedPlot(self.pi_PS.plot(pen=self.PEN_02))
        self.BP_PSs = [self.BP_PS_1, self.BP_PS_2, self.BP_PS_3, self.BP_PS_4]
        
        # QGROUP: Zoom
        self.qpbt_PS_zoom_DC        = QtWid.QPushButton('DC')
        self.qpbt_PS_zoom_f_ref     = QtWid.QPushButton('ref_freq')
        self.qpbt_PS_zoom_dbl_f_ref = QtWid.QPushButton('2x ref_freq')
        self.qpbt_PS_zoom_low       = QtWid.QPushButton('0 - 200 Hz')
        self.qpbt_PS_zoom_mid       = QtWid.QPushButton('0 - 1 kHz')
        self.qpbt_PS_zoom_all       = QtWid.QPushButton('Full range')
        
        self.qpbt_PS_zoom_DC.clicked.connect(lambda:
            self.plot_zoom_x(self.pi_PS, 0, 10))
        self.qpbt_PS_zoom_f_ref.clicked.connect(lambda:
            self.plot_zoom_x(self.pi_PS,
                             lockin.config.ref_freq - 10,
                             lockin.config.ref_freq + 10))
        self.qpbt_PS_zoom_dbl_f_ref.clicked.connect(lambda:
            self.plot_zoom_x(self.pi_PS,
                             2*lockin.config.ref_freq - 10,
                             2*lockin.config.ref_freq + 10))
        self.qpbt_PS_zoom_low.clicked.connect(lambda:
            self.plot_zoom_x(self.pi_PS, 0, 200))
        self.qpbt_PS_zoom_mid.clicked.connect(lambda:
            self.plot_zoom_x(self.pi_PS, 0, 1000))
        self.qpbt_PS_zoom_all.clicked.connect(lambda:
            self.plot_zoom_x(self.pi_PS, 0, self.lockin.config.F_Nyquist))
            
        grid = QtWid.QGridLayout()
        grid.addWidget(self.qpbt_PS_zoom_DC       , 0, 0)
        grid.addWidget(self.qpbt_PS_zoom_f_ref    , 0, 1)
        grid.addWidget(self.qpbt_PS_zoom_dbl_f_ref, 0, 2)
        grid.addWidget(self.qpbt_PS_zoom_low      , 0, 3)
        grid.addWidget(self.qpbt_PS_zoom_mid      , 0, 4)
        grid.addWidget(self.qpbt_PS_zoom_all      , 0, 5)
        
        qgrp_zoom = QtWid.QGroupBox("Zoom")
        qgrp_zoom.setLayout(grid)
        
        # QGROUP: Power spectrum
        self.legend_box_PS = Legend_box(
                text=['sig_I', 'filt_I', 'mix_X', 'mix_Y'],
                pen=[self.PEN_03, self.PEN_04, self.PEN_01, self.PEN_02],
                checked=[False, False, False, False])
        ([chkb.clicked.connect(self.process_chkbs_legend_box_PS) for chkb
          in self.legend_box_PS.chkbs])
    
        grid = QtWid.QGridLayout(spacing=4)
        grid.addWidget(QtWid.QLabel("CPU intensive!<br/>Only check<br/>"
                                    "when needed."), 0, 0)
        grid.addItem(QtWid.QSpacerItem(0, 4)       , 1, 0)
        grid.addLayout(self.legend_box_PS.grid     , 2, 0)
        grid.setAlignment(QtCore.Qt.AlignTop)
    
        qgrp_PS = QtWid.QGroupBox("Pow. spectrum")
        qgrp_PS.setLayout(grid)
        
        # -----------------------------------
        # -----------------------------------
        #   Round up tab page 'Power spectrum'
        # -----------------------------------
        # -----------------------------------        
        
        grid = QtWid.QGridLayout(spacing=0)
        grid.addWidget(qgrp_PS   , 0, 0, 2, 1, QtCore.Qt.AlignTop)
        grid.addWidget(qgrp_zoom , 0, 1)
        grid.addWidget(self.gw_PS, 1, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnMinimumWidth(0, LEFT_COLUMN_WIDTH)
        
        self.tab_power_spectrum.setLayout(grid)
                
        def Filter_1_design(): pass # Spider IDE outline bookmark
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        #
        #   TAB PAGE: Filter design @ sig_I
        #
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        
        # Plot: Filter response @ sig_I
        self.gw_filt_1_resp = pg.GraphicsWindow()
        self.gw_filt_1_resp.setBackground([20, 20, 20])        
        self.pi_filt_1_resp = self.gw_filt_1_resp.addPlot()
        
        p = {'color': '#BBB', 'font-size': '10pt'}
        self.pi_filt_1_resp.showGrid(x=1, y=1)
        self.pi_filt_1_resp.setTitle('Filter response', **p)
        self.pi_filt_1_resp.setLabel('bottom', text='frequency [Hz]', **p)
        self.pi_filt_1_resp.setLabel('left', text='amplitude attenuation [dB]',
                                     **p)
        self.pi_filt_1_resp.setAutoVisible(x=True, y=True)
        self.pi_filt_1_resp.enableAutoRange('x', False)
        self.pi_filt_1_resp.enableAutoRange('y', True)
        self.pi_filt_1_resp.setClipToView(True)
        self.pi_filt_1_resp.setLimits(xMin=0,
                                      xMax=self.lockin.config.F_Nyquist,
                                      yMin=-120)
        
        if 0:
            # Only enable for debugging
            # This will show individual symbols per data point, which is slow
            self.curve_filt_1_resp = pg.ScatterPlotItem(pen=self.PEN_03,
                                                        size=8, symbol='o')
        else:
            # Fast line plotting
            self.curve_filt_1_resp = pg.PlotCurveItem(pen=self.PEN_03,
                                                      brush=self.BRUSH_03)
        self.pi_filt_1_resp.addItem(self.curve_filt_1_resp)
        self.update_plot_filt_1_resp()
        self.plot_zoom_ROI_filt_1()
        
        # QGROUP: Zoom
        self.qpbt_filt_1_resp_zoom_DC  = QtWid.QPushButton('DC')
        self.qpbt_filt_1_resp_zoom_50  = QtWid.QPushButton('50 Hz')
        self.qpbt_filt_1_resp_zoom_low = QtWid.QPushButton('0 - 200 Hz')
        self.qpbt_filt_1_resp_zoom_mid = QtWid.QPushButton('0 - 1 kHz')
        self.qpbt_filt_1_resp_zoom_all = QtWid.QPushButton('Full range')
        self.qpbt_filt_1_resp_zoom_ROI = QtWid.QPushButton('Region of interest')

        self.qpbt_filt_1_resp_zoom_DC.clicked.connect(lambda:
            self.plot_zoom_x(self.pi_filt_1_resp, 0, 2))
        self.qpbt_filt_1_resp_zoom_50.clicked.connect(lambda:
            self.plot_zoom_x(self.pi_filt_1_resp, 47, 53))        
        self.qpbt_filt_1_resp_zoom_low.clicked.connect(lambda:
            self.plot_zoom_x(self.pi_filt_1_resp, 0, 200))
        self.qpbt_filt_1_resp_zoom_mid.clicked.connect(lambda:
            self.plot_zoom_x(self.pi_filt_1_resp, 0, 1000))
        self.qpbt_filt_1_resp_zoom_all.clicked.connect(lambda:
            self.plot_zoom_x(self.pi_filt_1_resp, 0, self.lockin.config.F_Nyquist))
        self.qpbt_filt_1_resp_zoom_ROI.clicked.connect(
                self.plot_zoom_ROI_filt_1)
            
        grid = QtWid.QGridLayout()
        grid.addWidget(self.qpbt_filt_1_resp_zoom_DC , 0, 0)
        grid.addWidget(self.qpbt_filt_1_resp_zoom_50 , 0, 1)
        grid.addWidget(self.qpbt_filt_1_resp_zoom_low, 0, 2)
        grid.addWidget(self.qpbt_filt_1_resp_zoom_mid, 0, 3)
        grid.addWidget(self.qpbt_filt_1_resp_zoom_all, 0, 4)
        grid.addWidget(self.qpbt_filt_1_resp_zoom_ROI, 0, 5)
        
        qgrp_zoom = QtWid.QGroupBox("Zoom")
        qgrp_zoom.setLayout(grid)
        
        # QGROUP: Filter design
        self.filt_1_design_GUI = Filter_design_GUI(
                self.lockin_pyqt.firf_1_sig_I)
        self.filt_1_design_GUI.signal_filter_design_updated.connect(
                self.update_plot_filt_1_resp)
        
        # -----------------------------------
        # -----------------------------------
        #   Round up tab page 'Filter response @ sig_I
        # -----------------------------------
        # -----------------------------------
        
        grid = QtWid.QGridLayout(spacing=0)
        grid.addWidget(self.filt_1_design_GUI.qgrp
                                          , 0, 0, 2, 1, QtCore.Qt.AlignTop)
        grid.addWidget(qgrp_zoom          , 0, 1)
        grid.addWidget(self.gw_filt_1_resp, 1, 1)
        grid.setColumnStretch(1, 1)
        
        self.tab_filter_1_design.setLayout(grid)
        
        def Filter_2_design(): pass # Spider IDE outline bookmark
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        #
        #   TAB PAGE: Filter design @ mix_X/Y
        #
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        
        # Plot: Filter response @ mix_X/Y
        self.gw_filt_2_resp = pg.GraphicsWindow()
        self.gw_filt_2_resp.setBackground([20, 20, 20])        
        self.pi_filt_2_resp = self.gw_filt_2_resp.addPlot()
        
        p = {'color': '#BBB', 'font-size': '10pt'}
        self.pi_filt_2_resp.showGrid(x=1, y=1)
        self.pi_filt_2_resp.setTitle('Filter response', **p)
        self.pi_filt_2_resp.setLabel('bottom', text='frequency [Hz]', **p)
        self.pi_filt_2_resp.setLabel('left', text='amplitude attenuation [dB]',
                                     **p)
        self.pi_filt_2_resp.setAutoVisible(x=True, y=True)
        self.pi_filt_2_resp.enableAutoRange('x', False)
        self.pi_filt_2_resp.enableAutoRange('y', True)
        self.pi_filt_2_resp.setClipToView(True)
        self.pi_filt_2_resp.setLimits(xMin=0,
                                      xMax=self.lockin.config.F_Nyquist,
                                      yMin=-120)
        
        if 0:
            # Only enable for debugging
            # This will show individual symbols per data point, which is slow
            self.curve_filt_2_resp = pg.ScatterPlotItem(pen=self.PEN_03,
                                                        size=8, symbol='o')
        else:
            # Fast line plotting
            self.curve_filt_2_resp = pg.PlotCurveItem(pen=self.PEN_03,
                                                      brush=self.BRUSH_03)
        self.pi_filt_2_resp.addItem(self.curve_filt_2_resp)
        self.update_plot_filt_2_resp()
        self.plot_zoom_ROI_filt_2()
        
        # QGROUP: Zoom
        self.qpbt_filt_2_resp_zoom_DC  = QtWid.QPushButton('DC')
        self.qpbt_filt_2_resp_zoom_50  = QtWid.QPushButton('50 Hz')
        self.qpbt_filt_2_resp_zoom_low = QtWid.QPushButton('0 - 200 Hz')
        self.qpbt_filt_2_resp_zoom_mid = QtWid.QPushButton('0 - 1 kHz')
        self.qpbt_filt_2_resp_zoom_all = QtWid.QPushButton('Full range')
        self.qpbt_filt_2_resp_zoom_ROI = QtWid.QPushButton('Region of interest')

        self.qpbt_filt_2_resp_zoom_DC.clicked.connect(lambda:
            self.plot_zoom_x(self.pi_filt_2_resp, 0, 2))
        self.qpbt_filt_2_resp_zoom_50.clicked.connect(lambda:
            self.plot_zoom_x(self.pi_filt_2_resp, 47, 53))        
        self.qpbt_filt_2_resp_zoom_low.clicked.connect(lambda:
            self.plot_zoom_x(self.pi_filt_2_resp, 0, 200))
        self.qpbt_filt_2_resp_zoom_mid.clicked.connect(lambda:
            self.plot_zoom_x(self.pi_filt_2_resp, 0, 1000))
        self.qpbt_filt_2_resp_zoom_all.clicked.connect(lambda:
            self.plot_zoom_x(self.pi_filt_2_resp, 0, self.lockin.config.F_Nyquist))
        self.qpbt_filt_2_resp_zoom_ROI.clicked.connect(
                self.plot_zoom_ROI_filt_2)
            
        grid = QtWid.QGridLayout()
        grid.addWidget(self.qpbt_filt_2_resp_zoom_DC , 0, 0)
        grid.addWidget(self.qpbt_filt_2_resp_zoom_50 , 0, 1)
        grid.addWidget(self.qpbt_filt_2_resp_zoom_low, 0, 2)
        grid.addWidget(self.qpbt_filt_2_resp_zoom_mid, 0, 3)
        grid.addWidget(self.qpbt_filt_2_resp_zoom_all, 0, 4)
        grid.addWidget(self.qpbt_filt_2_resp_zoom_ROI, 0, 5)
        
        qgrp_zoom = QtWid.QGroupBox("Zoom")
        qgrp_zoom.setLayout(grid)
        
        # QGROUP: Filter design
        self.filt_2_design_GUI = Filter_design_GUI(
                [self.lockin_pyqt.firf_2_mix_X, self.lockin_pyqt.firf_2_mix_Y])
        self.filt_2_design_GUI.signal_filter_design_updated.connect(
                self.update_plot_filt_2_resp)
        
        # -----------------------------------
        # -----------------------------------
        #   Round up tab page 'Filter response @ mix_X/Y
        # -----------------------------------
        # -----------------------------------
        
        grid = QtWid.QGridLayout(spacing=0)
        grid.addWidget(self.filt_2_design_GUI.qgrp
                                          , 0, 0, 2, 1, QtCore.Qt.AlignTop)
        grid.addWidget(qgrp_zoom          , 0, 1)
        grid.addWidget(self.gw_filt_2_resp, 1, 1)
        grid.setColumnStretch(1, 1)
        
        self.tab_filter_2_design.setLayout(grid)
        
        def Diagram(): pass # Spider IDE outline bookmark
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        #
        #   TAB PAGE: Diagram
        #
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        
        # Schematic diagram
        qlbl = QtGui.QLabel()
        qpix = QtGui.QPixmap('diagram_signal_processing.png')
        qlbl.setPixmap(qpix)
        
        # -----------------------------------
        # -----------------------------------
        #   Round up tab page 'Settings'
        # -----------------------------------
        # -----------------------------------
        
        grid = QtWid.QGridLayout(spacing=0)
        grid.addWidget(qlbl, 0, 0, QtCore.Qt.AlignTop)
        grid.setColumnStretch(1, 1)
        
        self.tab_diagram.setLayout(grid)
        
        def Settings(): pass # Spider IDE outline bookmark
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        #
        #   TAB PAGE: Settings
        #
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        
        self.qchk_boost_fps_graphing = QtWid.QCheckBox(
                "Boost fps graphing", checked=self.boost_fps_graphing)
        self.qchk_boost_fps_graphing.clicked.connect(
                self.process_qchk_boost_fps_graphing)
        
        grid = QtWid.QGridLayout(spacing=4)
        grid.addWidget(self.qchk_boost_fps_graphing, 0, 0)
        grid.addWidget(QtWid.QLabel(
                "Check to favor more frames per<br>"
                "second for graphing at the expense<br>"
                "of a higher CPU load and possibly<br>"
                "dropped samples."), 1, 0)
        grid.addWidget(QtWid.QLabel(
                "Uncheck whenever you encounter<br>"
                "reccuring dropped samples."), 2, 0)
        
        qgrp = QtWid.QGroupBox("Python program")
        qgrp.setLayout(grid)
        
        # -----------------------------------
        # -----------------------------------
        #   Round up tab page 'Settings'
        # -----------------------------------
        # -----------------------------------
        
        grid = QtWid.QGridLayout()
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

        vbox = QtWid.QVBoxLayout(self)
        vbox.addLayout(hbox_header)
        
        hbox = QtWid.QHBoxLayout()
        hbox.addWidget(self.tabs, stretch=1)
        hbox.addLayout(vbox_sidebar, stretch=0)
        
        vbox.addItem(QtWid.QSpacerItem(0, 10))
        vbox.addLayout(hbox)
        
        # List of all pyqtgraph graphics windows
        self.gws_all = [self.gw_refsig,
                        self.gw_XRYT,
                        self.gw_filt_1,
                        self.gw_mixer,
                        self.gw_PS,
                        self.gw_filt_1_resp,
                        self.gw_filt_2_resp]
        
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
        
        self.lockin_pyqt.signal_DAQ_updated.connect(
                self.update_GUI)
        
        self.lockin_pyqt.signal_DAQ_suspended.connect(
                self.update_GUI)
        
        self.lockin_pyqt.signal_ref_freq_is_set.connect(
                self.update_newly_set_ref_freq)
        
        self.lockin_pyqt.signal_ref_V_offset_is_set.connect(
                self.update_newly_set_ref_V_offset)
        
        self.lockin_pyqt.signal_ref_V_ampl_is_set.connect(
                self.update_newly_set_ref_V_ampl)
    
        self.file_logger.signal_set_recording_text.connect(
                self.set_text_qpbt_record)

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    #   Handle controls
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    @QtCore.pyqtSlot()
    def update_wall_clock(self):
        cur_date_time = QDateTime.currentDateTime()
        self.qlbl_cur_date_time.setText("%s    %s" %
                                        (cur_date_time.toString("dd-MM-yyyy"),
                                         cur_date_time.toString("HH:mm:ss")))
        
        if self.prev_time_CPU_load.msecsTo(cur_date_time) > 1000:
            self.qlbl_CPU_load.setText("CPU: %.1f%%" %
                                       psutil.cpu_percent(interval=None))
            self.prev_time_CPU_load = cur_date_time
    
    @QtCore.pyqtSlot()
    def update_GUI(self):
        #dprint("Update GUI")
        
        # Major visual changes upcoming. Reduce CPU overhead by momentarily
        # disabling screen repaints and GUI events.
        self.setUpdatesEnabled(False)
        
        LIA_pyqt = self.lockin_pyqt
        self.qlbl_update_counter.setText("%i" % LIA_pyqt.DAQ_update_counter)
        
        if LIA_pyqt.worker_DAQ.suspended:
            self.qlbl_DAQ_rate.setText("Buffers/s: paused")
        else:
            self.qlbl_DAQ_rate.setText("Buffers/s: %.1f" % 
                                       LIA_pyqt.obtained_DAQ_rate_Hz)
        
        self.qlin_time.setText("%i" % LIA_pyqt.state.time[0])
        self.qlin_sig_I_max.setText("%.4f" % LIA_pyqt.state.sig_I_max)
        self.qlin_sig_I_min.setText("%.4f" % LIA_pyqt.state.sig_I_min)
        self.qlin_sig_I_avg.setText("%.4f" % LIA_pyqt.state.sig_I_avg)
        self.qlin_sig_I_std.setText("%.4f" % LIA_pyqt.state.sig_I_std)
        self.qlin_X_avg.setText("%.4f" % np.mean(LIA_pyqt.state.X))
        self.qlin_Y_avg.setText("%.4f" % np.mean(LIA_pyqt.state.Y))
        self.qlin_R_avg.setText("%.4f" % np.mean(LIA_pyqt.state.R))
        self.qlin_T_avg.setText("%.3f" % np.mean(LIA_pyqt.state.T))
        
        if LIA_pyqt.firf_1_sig_I.deque_has_settled:
            self.LED_filt_1_deque_settled.setChecked(True)
            self.LED_filt_1_deque_settled.setText("YES")
        else:
            self.LED_filt_1_deque_settled.setChecked(False)
            self.LED_filt_1_deque_settled.setText("NO")
        if LIA_pyqt.firf_2_mix_X.deque_has_settled:
            self.LED_filt_2_deque_settled.setChecked(True)
            self.LED_filt_2_deque_settled.setText("YES")
        else:
            self.LED_filt_2_deque_settled.setChecked(False)
            self.LED_filt_2_deque_settled.setText("NO")
        
        self.update_chart_refsig()
        self.update_chart_filt_1()
        self.update_chart_mixer()
        self.update_chart_LIA_output()
        self.update_plot_PS()
        
        # Re-enable screen repaints and GUI events
        self.setUpdatesEnabled(True)
    
    @QtCore.pyqtSlot()
    def clear_chart_histories_stage_1_and_2(self):
        self.CH_filt_1_in.clear()
        self.CH_filt_1_out.clear()
        self.CH_mix_X.clear()
        self.CH_mix_Y.clear()
        self.CH_LIA_XR.clear()
        self.CH_LIA_YT.clear()

    @QtCore.pyqtSlot()
    def process_qpbt_ENA_lockin(self):
        if self.qpbt_ENA_lockin.isChecked():
            self.lockin_pyqt.turn_on()
            self.qpbt_ENA_lockin.setText("Lock-in ON")
            
            self.lockin_pyqt.state.reset()
            self.clear_chart_histories_stage_1_and_2()
        else:
            self.lockin_pyqt.turn_off()
            self.qpbt_ENA_lockin.setText("Lock-in OFF")
        
    @QtCore.pyqtSlot()
    def process_qpbt_record(self):
        if self.qpbt_record.isChecked():
            self.file_logger.start_recording()
        else:
            self.file_logger.stop_recording()

    @QtCore.pyqtSlot(str)
    def set_text_qpbt_record(self, text_str):
        self.qpbt_record.setText(text_str)
    
    @QtCore.pyqtSlot()
    def process_qlin_set_ref_freq(self):
        try:
            ref_freq = float(self.qlin_set_ref_freq.text())
        except ValueError:
            ref_freq = self.lockin.config.ref_freq
        
        # Clip between 0 and half the Nyquist frequency
        ref_freq = np.clip(ref_freq, 0, self.lockin.config.F_Nyquist/2)
        
        self.qlin_set_ref_freq.setText("%.2f" % ref_freq)
        if ref_freq != self.lockin.config.ref_freq:
            self.lockin_pyqt.set_ref_freq(ref_freq)
    
    @QtCore.pyqtSlot()
    def process_qlin_set_ref_V_offset(self):
        try:
            ref_V_offset = float(self.qlin_set_ref_V_offset.text())
        except ValueError:
            ref_V_offset = self.lockin.config.ref_V_offset
        
        # Clip between 0 and the analog voltage reference
        ref_V_offset = np.clip(ref_V_offset, 0, self.lockin.config.A_REF)
        
        self.qlin_set_ref_V_offset.setText("%.3f" % ref_V_offset)
        if ref_V_offset != self.lockin.config.ref_V_offset:
            self.lockin_pyqt.set_ref_V_offset(ref_V_offset)            
            QtWid.QApplication.processEvents()
            
    @QtCore.pyqtSlot()
    def process_qlin_set_ref_V_ampl(self):
        try:
            ref_V_ampl = float(self.qlin_set_ref_V_ampl.text())
        except ValueError:
            ref_V_ampl = self.lockin.config.ref_V_ampl
        
        # Clip between 0 and the analog voltage reference
        ref_V_ampl = np.clip(ref_V_ampl, 0, self.lockin.config.A_REF)
        
        self.qlin_set_ref_V_ampl.setText("%.3f" % ref_V_ampl)
        if ref_V_ampl != self.lockin.config.ref_V_ampl:
            self.lockin_pyqt.set_ref_V_ampl(ref_V_ampl)
            QtWid.QApplication.processEvents()
    
    @QtCore.pyqtSlot()
    def update_newly_set_ref_freq(self):
        self.qlin_read_ref_freq.setText("%.2f" % self.lockin.config.ref_freq)

        # TODO: the extra distance 'roll_off_width' to stay away from
        # f_cutoff should be calculated based on the roll-off width of the
        # filter, instead of hard-coded
        roll_off_width = 2; # [Hz]
        f_cutoff = 2*self.lockin.config.ref_freq - roll_off_width
        if f_cutoff > self.lockin.config.F_Nyquist - roll_off_width:
            print("WARNING: Filter @ mix_X/Y can't reach desired cut-off freq.")
            f_cutoff = self.lockin.config.F_Nyquist - roll_off_width
        
        self.lockin_pyqt.firf_2_mix_X.compute_firwin(cutoff=f_cutoff)
        self.lockin_pyqt.firf_2_mix_Y.compute_firwin(cutoff=f_cutoff)
        self.filt_2_design_GUI.update_filter_design()
        self.update_plot_filt_2_resp()
        self.plot_zoom_ROI_filt_2()
        
        self.lockin_pyqt.state.reset()
        self.clear_chart_histories_stage_1_and_2()
    
    @QtCore.pyqtSlot()
    def update_newly_set_ref_V_offset(self):
        self.qlin_read_ref_V_offset.setText("%.3f" %
                                            self.lockin.config.ref_V_offset)
        
        self.lockin_pyqt.state.reset()
        self.clear_chart_histories_stage_1_and_2()
        
    @QtCore.pyqtSlot()
    def update_newly_set_ref_V_ampl(self):
        self.qlin_read_ref_V_ampl.setText("%.3f" %
                                          self.lockin.config.ref_V_ampl)
        
        self.lockin_pyqt.state.reset()
        self.clear_chart_histories_stage_1_and_2()

    @QtCore.pyqtSlot()
    def process_chkbs_legend_box_refsig(self):
        if self.lockin.lockin_paused:
            self.update_chart_refsig()      # Force update graph
            
    @QtCore.pyqtSlot()
    def process_chkbs_legend_box_filt_1(self):
        if self.lockin.lockin_paused:
            self.update_chart_filt_1()     # Force update graph
            
    @QtCore.pyqtSlot()
    def process_chkbs_legend_box_mixer(self):
        if self.lockin.lockin_paused:
            self.update_chart_mixer()       # Force update graph
            
    @QtCore.pyqtSlot()
    def process_chkbs_legend_box_PS(self):
        if self.lockin.lockin_paused:
            L = self.lockin_pyqt
            
            if self.legend_box_PS.chkbs[0].isChecked():
                [f, P_dB] = L.compute_power_spectrum(L.state.deque_sig_I)
                if len(f) > 0: self.BP_PS_1.set_data(f, P_dB)
                    
            if self.legend_box_PS.chkbs[1].isChecked():
                [f, P_dB] = L.compute_power_spectrum(L.state.deque_filt_I)
                if len(f) > 0: self.BP_PS_2.set_data(f, P_dB)
                
            if self.legend_box_PS.chkbs[2].isChecked():
                [f, P_dB] = L.compute_power_spectrum(L.state.deque_mix_X)
                if len(f) > 0: self.BP_PS_3.set_data(f, P_dB)
                
            if self.legend_box_PS.chkbs[3].isChecked():
                [f, P_dB] = L.compute_power_spectrum(L.state.deque_mix_Y)
                if len(f) > 0: self.BP_PS_4.set_data(f, P_dB)
            
            self.update_plot_PS()           # Force update graph

    @QtCore.pyqtSlot()
    def process_qpbt_fullrange_xy(self):
        self.process_qpbt_autorange_x()
        self.pi_refsig.setYRange(-3.3, 3.3, padding=0.05)
        self.pi_filt_1.setYRange(-3.3, 3.3, padding=0.05)
        self.pi_mixer.setYRange (-5, 5, padding=0.05)
        
        if self.qrbt_XR_X.isChecked():
            self.pi_XR.setYRange(-5, 5, padding=0.05)
        else:
            self.pi_XR.setYRange(0, 5, padding=0.05)
        if self.qrbt_YT_Y.isChecked():
            self.pi_YT.setYRange(-5, 5, padding=0.05)
        else:
            self.pi_YT.setYRange(-90, 90, padding=0.1)

    @QtCore.pyqtSlot()
    def process_qpbt_autorange_xy(self):
        self.process_qpbt_autorange_x()
        self.process_qpbt_autorange_y()

    @QtCore.pyqtSlot()
    def process_qpbt_autorange_x(self):
        plot_items = [self.pi_refsig,
                      self.pi_filt_1,
                      self.pi_mixer,
                      self.pi_XR,
                      self.pi_YT]
        for pi in plot_items:
            pi.enableAutoRange('x', False)
            pi.setXRange(-self.lockin.config.BUFFER_SIZE *
                         self.lockin.config.ISR_CLOCK * 1e3, 0,
                         padding=0.01)

    @QtCore.pyqtSlot()
    def process_qpbt_autorange_y(self):
        plot_items = [self.pi_refsig,
                      self.pi_filt_1,
                      self.pi_mixer]
        for pi in plot_items:
            pi.enableAutoRange('y', True)
            pi.enableAutoRange('y', False)
        self.autorange_y_XR()
        self.autorange_y_YT()
    
    @QtCore.pyqtSlot()
    def process_qrbt_XR(self):        
        if self.qrbt_XR_X.isChecked():
            self.CH_LIA_XR.curve.setPen(self.PEN_01)
            self.pi_XR.setTitle('X')
        else:
            self.CH_LIA_XR.curve.setPen(self.PEN_03)
            self.pi_XR.setTitle('R')
            
        if self.lockin_pyqt.worker_DAQ.suspended:
            # The graphs are not being updated with the newly chosen timeseries
            # automatically because the lock-in is not running. It is safe
            # however to make a copy of the lockin_pyqt.state timeseries,
            # because the GUI can't interfere with the DAQ thread now that it is
            # in suspended mode. Hence, we copy new data into the chart.
            if self.qrbt_XR_X.isChecked():
                self.CH_LIA_XR.add_new_readings(self.lockin_pyqt.state.time2,
                                                self.lockin_pyqt.state.X)
            else:
                self.CH_LIA_XR.add_new_readings(self.lockin_pyqt.state.time2,
                                                self.lockin_pyqt.state.R)
            self.CH_LIA_XR.update_curve()        
            self.autorange_y_XR()
        else:
            # To be handled by update_chart_LIA_output
            self.pi_XR.request_autorange_y = True
    
    @QtCore.pyqtSlot()
    def process_qrbt_YT(self):
        if self.qrbt_YT_Y.isChecked():
            self.CH_LIA_YT.curve.setPen(self.PEN_02)
            self.pi_YT.setTitle('Y')
            self.pi_YT.setLabel('left', text='voltage [V]')
        else:
            self.CH_LIA_YT.curve.setPen(self.PEN_03)
            self.pi_YT.setTitle('%s' % chr(0x398))
            self.pi_YT.setLabel('left', text='phase [deg]')
            
        if self.lockin_pyqt.worker_DAQ.suspended:
            # The graphs are not being updated with the newly chosen timeseries
            # automatically because the lock-in is not running. It is safe
            # however to make a copy of the lockin_pyqt.state timeseries,
            # because the GUI can't interfere with the DAQ thread now that it is
            # in suspended mode. Hence, we copy new data into the chart.
            if self.qrbt_YT_Y.isChecked():
                self.CH_LIA_YT.add_new_readings(self.lockin_pyqt.state.time2,
                                                self.lockin_pyqt.state.Y)
            else:
                self.CH_LIA_YT.add_new_readings(self.lockin_pyqt.state.time2,
                                                self.lockin_pyqt.state.T)
            self.CH_LIA_YT.update_curve()
            self.autorange_y_YT()
        else:
            # To be handled by update_chart_LIA_output
            self.pi_YT.request_autorange_y = True
    
    def autorange_y_XR(self):
        if len(self.CH_LIA_XR._x) == 0:
            if self.qrbt_XR_X.isChecked():
                self.pi_XR.setYRange(-5, 5, padding=0.05)
            else:
                self.pi_XR.setYRange(0, 5, padding=0.05)
        else:
            self.pi_XR.enableAutoRange('y', True)
            self.pi_XR.enableAutoRange('y', False)
            XRange, YRange = self.pi_XR.viewRange()
            self.pi_XR.setYRange(YRange[0], YRange[1], padding=0.1)
        
    def autorange_y_YT(self):
        if len(self.CH_LIA_YT._x) == 0:
            if self.qrbt_YT_Y.isChecked():
                self.pi_YT.setYRange(-5, 5, padding=0.05)
            else:
                self.pi_YT.setYRange(-90, 90, padding=0.1)
        else:
            self.pi_YT.enableAutoRange('y', True)
            self.pi_YT.enableAutoRange('y', False)
            XRange, YRange = self.pi_YT.viewRange()
            self.pi_YT.setYRange(YRange[0], YRange[1], padding=0.1)
    
    
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    #   Update chart/plot routines
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    

    @QtCore.pyqtSlot()
    def update_chart_refsig(self):
        [CH.update_curve() for CH in self.CHs_refsig]
        for i in range(3):
            self.CHs_refsig[i].curve.setVisible(
                    self.legend_box_refsig.chkbs[i].isChecked())
            
    @QtCore.pyqtSlot()
    def update_chart_filt_1(self):
        [CH.update_curve() for CH in self.CHs_filt_1]
        for i in range(2):
            self.CHs_filt_1[i].curve.setVisible(
                    self.legend_box_filt_1.chkbs[i].isChecked())
            
    @QtCore.pyqtSlot()
    def update_chart_mixer(self):
        [CH.update_curve() for CH in self.CHs_mixer]
        for i in range(2):
            self.CHs_mixer[i].curve.setVisible(
                    self.legend_box_mixer.chkbs[i].isChecked())
        
    @QtCore.pyqtSlot()
    def update_chart_LIA_output(self):
        [CH.update_curve() for CH in self.CHs_LIA_output]
        
        if self.pi_XR.request_autorange_y == True:
            self.pi_XR.request_autorange_y = False
            self.autorange_y_XR()
        
        if self.pi_YT.request_autorange_y == True:
            self.pi_YT.request_autorange_y = False
            self.autorange_y_YT()
            
    @QtCore.pyqtSlot()
    def update_plot_PS(self):
        [BP.update_curve() for BP in self.BP_PSs]
        for i in range(len(self.BP_PSs)):
            self.BP_PSs[i].curve.setVisible(
                    self.legend_box_PS.chkbs[i].isChecked())
        
    @QtCore.pyqtSlot()
    def update_plot_filt_1_resp(self):
        firf = self.lockin_pyqt.firf_1_sig_I
        if isinstance(self.curve_filt_1_resp, pg.PlotCurveItem):
            self.curve_filt_1_resp.setFillLevel(np.min(firf.resp_ampl_dB))
        self.curve_filt_1_resp.setData(firf.resp_freq_Hz,
                                       firf.resp_ampl_dB)
        #self.pi_filt_1_resp.setTitle('Filter response: <br/>%s' %
        #                              self.filt_resp_construct_title(firf))
                
    @QtCore.pyqtSlot()
    def update_plot_filt_2_resp(self):
        firf = self.lockin_pyqt.firf_2_mix_X
        if isinstance(self.curve_filt_2_resp, pg.PlotCurveItem):
            self.curve_filt_2_resp.setFillLevel(np.min(firf.resp_ampl_dB))
        self.curve_filt_2_resp.setData(firf.resp_freq_Hz,
                                       firf.resp_ampl_dB)
        #self.pi_filt_2_resp.setTitle('Filter response: <br/>%s' %
        #                             self.filt_resp_construct_title(firf))    

    def filt_resp_construct_title(self, firf):
        __tmp1 = 'N_taps = %i' % firf.N_taps
        if isinstance(firf.window, str):
            __tmp2 = '%s' % firf.window
        else:
            __tmp2 = '%s' % [x for x in firf.window]
        __tmp3 = '%s Hz' % [round(x, 1) for x in firf.cutoff]        
        
        return ('%s, %s, %s' % (__tmp1, __tmp2, __tmp3))

    @QtCore.pyqtSlot()
    def plot_zoom_ROI_filt_1(self):
        firf = self.lockin_pyqt.firf_1_sig_I
        self.pi_filt_1_resp.setXRange(firf.resp_freq_Hz__ROI_start,
                                      firf.resp_freq_Hz__ROI_end,
                                      padding=0.01)

    @QtCore.pyqtSlot()
    def plot_zoom_ROI_filt_2(self):
        firf = self.lockin_pyqt.firf_2_mix_X
        self.pi_filt_2_resp.setXRange(firf.resp_freq_Hz__ROI_start,
                                      firf.resp_freq_Hz__ROI_end,
                                      padding=0.01)
        
    def plot_zoom_x(self, pi_plot: pg.PlotItem, xRangeLo, xRangeHi):
        pi_plot.setXRange(xRangeLo, xRangeHi, padding=0.02)
        pi_plot.enableAutoRange('y', True)
        pi_plot.enableAutoRange('y', False)
        
    @QtCore.pyqtSlot()
    def process_qchk_boost_fps_graphing(self):
        self.boost_fps_graphing = not self.boost_fps_graphing
        
if __name__ == "__main__":
    exec(open("DvG_Arduino_lockin_amp.py").read())