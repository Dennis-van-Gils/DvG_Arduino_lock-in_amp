#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Arduino lock-in amplifier
"""
__author__      = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__         = "https://github.com/Dennis-van-Gils/DvG_Arduino_lock-in_amp"
__date__        = "18-04-2019"
__version__     = "1.0.0"

from PyQt5 import QtCore, QtGui
from PyQt5 import QtWidgets as QtWid
from PyQt5.QtCore import QDateTime
import pyqtgraph as pg
import numpy as np

from DvG_pyqt_ChartHistory import ChartHistory
from DvG_pyqt_BufferedPlot import BufferedPlot
from DvG_pyqt_controls     import (create_Toggle_button,
                                   create_LED_indicator_rect,
                                   #SS_GROUP,
                                   SS_GROUP_BORDERLESS,
                                   SS_TEXTBOX_READ_ONLY,
                                   Legend_box)
from DvG_pyqt_FileLogger   import FileLogger
from DvG_debug_functions   import dprint

import DvG_dev_Arduino_lockin_amp__fun_serial as lockin_functions
import DvG_dev_Arduino_lockin_amp__pyqt_lib   as lockin_pyqt_lib

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

SS_GROUP = (
        "QGroupBox {"
            "background-color: " + "rgb(252, 208, 173)" + ";"
            "border: 2px solid gray;"
            "border-radius: 0px;"
            "font: bold ;"
            "padding: 8 0 0 0px;"
            "margin-top: 2ex}"
        "QGroupBox::title {"
            "subcontrol-origin: margin;"
            "subcontrol-position: top left;"
            "left: 6 px;"
            "padding: -5 3px}"
        "QGroupBox::flat {"
            "border: 0px;"
            "border-radius: 0 0px;"
            "padding: 0}")

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
        
        self.setGeometry(250, 50, 1200, 960)
        self.setWindowTitle("Arduino lock-in amplifier")
        self.setStyleSheet(SS_TEXTBOX_READ_ONLY + SS_GROUP)
        
        """
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
        
        # Textbox widths for fitting N characters using the current font
        e = QtGui.QLineEdit()
        width_chr8  = (8 + 8 * e.fontMetrics().width('x') + 
                       e.textMargins().left()     + e.textMargins().right() + 
                       e.contentsMargins().left() + e.contentsMargins().right())
        width_chr10 = (8 + 10 * e.fontMetrics().width('x') + 
                       e.textMargins().left()     + e.textMargins().right() + 
                       e.contentsMargins().left() + e.contentsMargins().right())
        width_chr12 = (8 + 12 * e.fontMetrics().width('x') + 
                       e.textMargins().left()     + e.textMargins().right() + 
                       e.contentsMargins().left() + e.contentsMargins().right())
        del e

        def _frame_Header(): pass # Spider IDE outline bookmark
        # -----------------------------------
        # -----------------------------------
        #   FRAME: Header
        # -----------------------------------
        # -----------------------------------

        # Left box
        self.qlbl_update_counter = QtWid.QLabel("0")
        self.qlbl_sample_rate = QtWid.QLabel("SAMPLE RATE: %.2f Hz" %
                                             (1/lockin.config.ISR_CLOCK))
        self.qlbl_buffer_size = QtWid.QLabel("BUFFER SIZE  : %i" %
                                             lockin.config.BUFFER_SIZE)
        self.qlbl_DAQ_rate = QtWid.QLabel("Buffers/s: nan")
        self.qlbl_DAQ_rate.setMinimumWidth(100)

        vbox_left = QtWid.QVBoxLayout()
        vbox_left.addWidget(self.qlbl_sample_rate)
        vbox_left.addWidget(self.qlbl_buffer_size)
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
        
        def _frame_Tabs(): pass # Spider IDE outline bookmark
        # -----------------------------------
        # -----------------------------------
        #   FRAME: Tabs
        # -----------------------------------
        # -----------------------------------
        
        self.tabs = QtWid.QTabWidget()
        #self.tabs.setStyleSheet("QWidget {background-color: gray}")
        #"""
        #blue = "rgb(220, 220, 238)"
        #greengray = "rgb(220, 220, 214)"
        greengray = "rgb(207, 225, 225)"
        greengraylighter = "rgb(234, 235, 233)"
        self.tabs.setStyleSheet(
                "QTabWidget::pane {"
                    "border: 0px solid gray;}"
                "QTabBar::tab:selected {"
                    "background: " + greengray + "; "
                    "border-bottom-color: " + greengray + ";}"
                "QTabWidget>QWidget>QWidget {"
                    "border: 2px solid gray;"
                    "background: " + greengray + ";} "
                "QTabBar::tab {"
                    "background: " + greengraylighter + ";"
                    "border: 2px solid gray;"
                    "border-bottom-color: " + greengraylighter + ";"
                    "border-top-left-radius: 4px;"
                    "border-top-right-radius: 4px;"
                    "min-width: 30ex;"
                    "padding: 6px;} "
                "QTabWidget::tab-bar {"
                    "left: 0px;}")
        #"""

        self.tab_main  = QtWid.QWidget()
        self.tab_mixer = QtWid.QWidget()
        self.tab_power_spectrum  = QtWid.QWidget()
        self.tab_filter_1_design = QtWid.QWidget()
        self.tab_filter_2_design = QtWid.QWidget()
        self.tab_mcu_board_info = QtWid.QWidget()
        
        self.tabs.addTab(self.tab_main           , "Main")
        self.tabs.addTab(self.tab_mixer          , "Mixer")
        self.tabs.addTab(self.tab_power_spectrum , "Spectrum")
        self.tabs.addTab(self.tab_filter_1_design, "Filter 1")
        self.tabs.addTab(self.tab_filter_2_design, "Filter 2")
        self.tabs.addTab(self.tab_mcu_board_info , "MCU board")
        
        def _frame_Sidebar(): pass # Spider IDE outline bookmark
        # -----------------------------------
        # -----------------------------------
        #   FRAME: Sidebar
        # -----------------------------------
        # -----------------------------------

        # On/off
        self.qpbt_ENA_lockin = create_Toggle_button("Lock-in OFF")
        self.qpbt_ENA_lockin.clicked.connect(self.process_qpbt_ENA_lockin)
        
        # QGROUP: Reference signal
        p1 = {'maximumWidth': width_chr8, 'minimumWidth': width_chr8}
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
        grid.addWidget(QtWid.QLabel("ref_X : cosine wave", **p2)
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
        grid.addWidget(QtWid.QLabel("Output: ref_X\n"
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
        self.qpbt_autorange_xy = QtWid.QPushButton("Autorange")
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
        
        qgrp_axes_controls = QtWid.QGroupBox("Graphs: timeseries")
        qgrp_axes_controls.setLayout(grid)

        # QGROUP: Filters settled?
        self.LED_settled_BG_filter = create_LED_indicator_rect(False, 'NO')
        self.LED_settled_LP_filter = create_LED_indicator_rect(False, 'NO')
        
        grid = QtWid.QGridLayout(spacing=4)
        grid.addWidget(QtWid.QLabel("1: band-stop"), 0, 0)
        grid.addWidget(self.LED_settled_BG_filter  , 0, 1)
        grid.addWidget(QtWid.QLabel("2: low-pass") , 1, 0)
        grid.addWidget(self.LED_settled_LP_filter  , 1, 1)
        
        qgrp_settling = QtWid.QGroupBox("Filters settled?")
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

        def _frame_Reference_and_signal(): pass # Spider IDE outline bookmark
        # -----------------------------------
        # -----------------------------------
        #   Frame: Reference and signal
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
                text=['ref_X', 'ref_Y', 'sig_I'],
                pen=[self.PEN_01, self.PEN_02, self.PEN_03],
                checked=[True, False, True])
        ([chkb.clicked.connect(self.process_chkbs_legend_box_refsig) for chkb
          in self.legend_box_refsig.chkbs])
        
        p1 = {'maximumWidth': width_chr12, 'minimumWidth': width_chr12,
              'readOnly': True}
        p2 = {'maximumWidth': width_chr8, 'minimumWidth': width_chr8,
              'readOnly': True}
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

        def _frame_LIA_output(): pass # Spider IDE outline bookmark
        # -----------------------------------
        # -----------------------------------
        #   FRAME: LIA output
        # -----------------------------------
        # -----------------------------------
        
        # Chart: X or R
        self.gw_XR = pg.GraphicsWindow()
        self.gw_XR.setBackground([20, 20, 20])
        self.pi_XR = self.gw_XR.addPlot()
        
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

        self.CH_LIA_XR = ChartHistory(lockin.config.BUFFER_SIZE,
                                      self.pi_XR.plot(pen=self.PEN_03))
        self.CH_LIA_XR.x_axis_divisor = 1000     # From [us] to [ms]
        
        # Chart: Y or T
        self.pi_YT = self.gw_XR.addPlot()
        
        p = {'color': '#BBB', 'font-size': '10pt'}
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
        
        self.CH_LIA_YT = ChartHistory(lockin.config.BUFFER_SIZE,
                                      self.pi_YT.plot(pen=self.PEN_03))
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
        qgrp_XR = QtWid.QGroupBox()
        qgrp_XR.setStyleSheet(SS_GROUP_BORDERLESS)
        qgrp_XR.setLayout(vbox)
        
        vbox = QtWid.QVBoxLayout(spacing=4)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.addWidget(self.qrbt_YT_Y)
        vbox.addWidget(self.qrbt_YT_T)
        qgrp_YT = QtWid.QGroupBox()
        qgrp_YT.setStyleSheet(SS_GROUP_BORDERLESS)
        qgrp_YT.setLayout(vbox)
        
        hbox = QtWid.QHBoxLayout(spacing=4)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(qgrp_XR)
        hbox.addWidget(qgrp_YT)
        
        p = {'maximumWidth': width_chr8, 'minimumWidth': width_chr8,
             'readOnly': True}
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
        grid.addWidget(self.gw_XR    , 1, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnMinimumWidth(0, LEFT_COLUMN_WIDTH)
        self.tab_main.setLayout(grid)
        
        #grid.setVerticalSpacing(0)
        
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        #
        #   TAB PAGE: Mixer
        #
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        
        def _frame_BS_filter_output(): pass # Spider IDE outline bookmark
        # -----------------------------------
        # -----------------------------------
        #   Frame: Band-stop filter output
        # -----------------------------------
        # -----------------------------------
        
        # Chart: Band-stop filter output
        self.gw_filt_BS = pg.GraphicsWindow()
        self.gw_filt_BS.setBackground([20, 20, 20])
        self.pi_filt_BS = self.gw_filt_BS.addPlot()
        
        p = {'color': '#BBB', 'font-size': '10pt'}
        self.pi_filt_BS.showGrid(x=1, y=1)
        self.pi_filt_BS.setTitle('Band-stop filter acting on sig_I', **p)
        self.pi_filt_BS.setLabel('bottom', text='time [ms]', **p)
        self.pi_filt_BS.setLabel('left', text='voltage [V]', **p)
        self.pi_filt_BS.setXRange(-lockin.config.BUFFER_SIZE *
                                    lockin.config.ISR_CLOCK * 1e3,
                                    0, padding=0.01)
        self.pi_filt_BS.setYRange(-5, 5, padding=0.05)
        self.pi_filt_BS.setAutoVisible(x=True, y=True)
        self.pi_filt_BS.setClipToView(True)
        self.pi_filt_BS.setLimits(xMin=-(lockin.config.BUFFER_SIZE + 1) *
                                  lockin.config.ISR_CLOCK * 1e3,
                                  xMax=0)
        
        self.CH_filt_BS_in  = ChartHistory(
                lockin.config.BUFFER_SIZE,
                self.pi_filt_BS.plot(pen=self.PEN_03))
        self.CH_filt_BS_out = ChartHistory(
                lockin.config.BUFFER_SIZE,
                self.pi_filt_BS.plot(pen=self.PEN_04))
        self.CH_filt_BS_in.x_axis_divisor = 1000     # From [us] to [ms]
        self.CH_filt_BS_out.x_axis_divisor = 1000    # From [us] to [ms]
        self.CHs_filt_BS = [self.CH_filt_BS_in, self.CH_filt_BS_out]
        
        # QGROUP: Band-stop filter output
        self.legend_box_filt_BS = Legend_box(text=['sig_I', 'filt_I'],
                                             pen=[self.PEN_03, self.PEN_04])
        ([chkb.clicked.connect(self.process_chkbs_legend_box_filt_BS) for chkb
          in self.legend_box_filt_BS.chkbs])
    
        qgrp_filt_BS = QtWid.QGroupBox("BS filter")

        qgrp_filt_BS.setLayout(self.legend_box_filt_BS.grid)
        
        def _frame_Mixer(): pass # Spider IDE outline bookmark
        # -----------------------------------
        # -----------------------------------
        #   Frame: Mixer
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
        grid.addWidget(qgrp_filt_BS   , 0, 0, QtCore.Qt.AlignTop)
        grid.addWidget(self.gw_filt_BS, 0, 1)
        grid.addWidget(qgrp_mixer     , 1, 0, QtCore.Qt.AlignTop)
        grid.addWidget(self.gw_mixer  , 1, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnMinimumWidth(0, LEFT_COLUMN_WIDTH)
        self.tab_mixer.setLayout(grid)
        
        def _frame_Power_spectrum(): pass # Spider IDE outline bookmark
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        #
        #   TAB PAGE: Power spectrum
        #
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        
        # Zoom controls
        self.qpbt_PS_zoom_low = QtWid.QPushButton('0 - 200 Hz')
        self.qpbt_PS_zoom_mid = QtWid.QPushButton('0 - 1 kHz')
        self.qpbt_PS_zoom_all = QtWid.QPushButton('Full range')
        
        self.qpbt_PS_zoom_low.clicked.connect(lambda:
            self.power_spectrum_zoom(0, 200))
        self.qpbt_PS_zoom_mid.clicked.connect(lambda:
            self.power_spectrum_zoom(0, 1000))
        self.qpbt_PS_zoom_all.clicked.connect(lambda:
            self.power_spectrum_zoom(0, self.lockin.config.F_Nyquist))
            
        grid = QtWid.QGridLayout()
        grid.addWidget(self.qpbt_PS_zoom_low, 0, 0)
        grid.addWidget(self.qpbt_PS_zoom_mid, 0, 1)
        grid.addWidget(self.qpbt_PS_zoom_all, 0, 2)
        
        qgrp_PS_zoom = QtWid.QGroupBox("Zoom")
        qgrp_PS_zoom.setLayout(grid)
        
        # Plot: Power spectrum
        self.gw_PS = pg.GraphicsWindow()
        self.gw_PS.setBackground([20, 20, 20])        
        self.pi_PS = self.gw_PS.addPlot()
        
        p = {'color': '#BBB', 'font-size': '10pt'}
        self.pi_PS.showGrid(x=1, y=1)
        self.pi_PS.setTitle('Power spectrum (Welch)', **p)
        self.pi_PS.setLabel('bottom', text='frequency [Hz]', **p)
        self.pi_PS.setLabel('left', text='power [dB]', **p)
        self.pi_PS.setAutoVisible(x=True, y=True)
        self.pi_PS.setXRange(0, self.lockin.config.F_Nyquist, padding=0.02)
        self.pi_PS.setYRange(-110, 0, padding=0.02)
        self.pi_PS.setClipToView(True)
        self.pi_PS.setLimits(xMin=0, xMax=self.lockin.config.F_Nyquist)
        
        self.BP_PS_1 = BufferedPlot(self.pi_PS.plot(pen=self.PEN_03))
        self.BP_PS_2 = BufferedPlot(self.pi_PS.plot(pen=self.PEN_04))
        
        grid_PS = QtWid.QGridLayout()
        grid_PS.addWidget(qgrp_PS_zoom, 0, 0)
        grid_PS.addWidget(self.gw_PS  , 1, 0)
        grid_PS.setRowStretch(1, 1)
        
        # QGROUP: Power spectrum
        self.legend_box_PS = Legend_box(text=['sig_I', 'filt_I'],
                                        pen=[self.PEN_03, self.PEN_04])
        ([chkb.clicked.connect(self.process_chkbs_legend_box_PS) for chkb
          in self.legend_box_PS.chkbs])
    
        grid = QtWid.QGridLayout(spacing=4)
        grid.addLayout(self.legend_box_PS.grid, 0, 0)
        grid.setAlignment(QtCore.Qt.AlignTop)
    
        qgrp_PS = QtWid.QGroupBox("Pow. spectrum")
        qgrp_PS.setLayout(grid)
        
        # -----------------------------------
        # -----------------------------------
        #   Round up tab page 'Power spectrum'
        # -----------------------------------
        # -----------------------------------
        
        grid = QtWid.QGridLayout(spacing=0)
        grid.addWidget(qgrp_PS, 0, 0, QtCore.Qt.AlignTop)
        grid.addLayout(grid_PS, 0, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnMinimumWidth(0, LEFT_COLUMN_WIDTH)
        self.tab_power_spectrum.setLayout(grid)
                
        def _frame_BS_filter_response(): pass # Spider IDE outline bookmark
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        #
        #   TAB PAGE: Filter response band-stop
        #
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        
        # Plot: Filter response band-stop
        self.gw_filt_resp_BS = pg.GraphicsWindow()
        self.gw_filt_resp_BS.setBackground([20, 20, 20])        
        self.pi_filt_resp_BS = self.gw_filt_resp_BS.addPlot()
        
        p = {'color': '#BBB', 'font-size': '10pt'}
        self.pi_filt_resp_BS.showGrid(x=1, y=1)
        self.pi_filt_resp_BS.setTitle('Filter response', **p)
        self.pi_filt_resp_BS.setLabel('bottom', text='frequency [Hz]', **p)
        self.pi_filt_resp_BS.setLabel('left', text='attenuation [dB]', **p)
        self.pi_filt_resp_BS.setAutoVisible(x=True, y=True)
        self.pi_filt_resp_BS.enableAutoRange('x', False)
        self.pi_filt_resp_BS.enableAutoRange('y', True)
        self.pi_filt_resp_BS.setClipToView(True)
        self.pi_filt_resp_BS.setLimits(xMin=0, xMax=self.lockin.config.F_Nyquist)
        
        self.curve_filt_resp_BS = pg.PlotCurveItem(pen=self.PEN_03,
                                                   brush=self.BRUSH_03)
        self.pi_filt_resp_BS.addItem(self.curve_filt_resp_BS)
        self.update_plot_filt_resp_BS()
        
        # Band-stop filter controls
        self.qtbl_filt_BS = QtWid.QTableWidget()

        default_font_pt = QtWid.QApplication.font().pointSize()
        self.qtbl_filt_BS.setStyleSheet(
                "QTableWidget {font-size: %ipt;"
                              "font-family: MS Shell Dlg 2}"
                "QHeaderView  {font-size: %ipt;"
                              "font-family: MS Shell Dlg 2}"
                "QHeaderView:section {background-color: lightgray}" %
                (default_font_pt, default_font_pt))
    
        self.qtbl_filt_BS.setRowCount(6)
        self.qtbl_filt_BS.setColumnCount(2)
        self.qtbl_filt_BS.setColumnWidth(0, width_chr8)
        self.qtbl_filt_BS.setColumnWidth(1, width_chr8)
        self.qtbl_filt_BS.setHorizontalHeaderLabels (['from', 'to'])
        self.qtbl_filt_BS.horizontalHeader().setDefaultAlignment(
                QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.qtbl_filt_BS.horizontalHeader().setSectionResizeMode(
                QtWid.QHeaderView.Fixed)
        self.qtbl_filt_BS.verticalHeader().setSectionResizeMode(
                QtWid.QHeaderView.Fixed)
        
        self.qtbl_filt_BS.setSizePolicy(
                QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.qtbl_filt_BS.setVerticalScrollBarPolicy(
                QtCore.Qt.ScrollBarAlwaysOff)
        self.qtbl_filt_BS.setHorizontalScrollBarPolicy(
                QtCore.Qt.ScrollBarAlwaysOff)
        self.qtbl_filt_BS.setFixedSize(
                self.qtbl_filt_BS.horizontalHeader().length() + 
                self.qtbl_filt_BS.verticalHeader().width() + 2,
                self.qtbl_filt_BS.verticalHeader().length() + 
                self.qtbl_filt_BS.horizontalHeader().height() + 2)
        
        self.qtbl_filt_BS_items = list()
        for row in range(self.qtbl_filt_BS.rowCount()):
            for col in range(self.qtbl_filt_BS.columnCount()):
                myItem = QtWid.QTableWidgetItem()
                myItem.setTextAlignment(QtCore.Qt.AlignRight |
                                        QtCore.Qt.AlignVCenter)
                self.qtbl_filt_BS_items.append(myItem)
                self.qtbl_filt_BS.setItem(row, col, myItem)
                
        p1 = {'maximumWidth': width_chr8, 'minimumWidth': width_chr8}
        p2 = {'alignment': QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight}
        self.qpbt_filt_BS_coupling = QtWid.QPushButton("AC", checkable=True)
        self.qlin_filt_BS_DC_cutoff = QtWid.QLineEdit(**{**p1, **p2})
        self.qlin_filt_BS_window = QtWid.QLineEdit(readOnly=True)
        self.qlin_filt_BS_window.setText(
                self.lockin_pyqt.firf_BS_sig_I.window_description)
                
        i = 0
        grid = QtWid.QGridLayout()
        grid.addWidget(QtWid.QLabel('Input coupling AC/DC:') , i, 0, 1, 3); i+=1
        grid.addWidget(self.qpbt_filt_BS_coupling            , i, 0, 1, 3); i+=1
        grid.addWidget(QtWid.QLabel('Cutoff:')               , i, 0)
        grid.addWidget(self.qlin_filt_BS_DC_cutoff           , i, 1)
        grid.addWidget(QtWid.QLabel('Hz')                    , i, 2)      ; i+=1
        grid.addItem(QtWid.QSpacerItem(0, 10)                , i, 0, 1, 3); i+=1
        grid.addWidget(QtWid.QLabel('Band-stop ranges [Hz]:'), i, 0, 1, 3); i+=1
        grid.addWidget(self.qtbl_filt_BS                     , i, 0, 1, 3); i+=1
        grid.addItem(QtWid.QSpacerItem(0, 10)                , i, 0, 1, 3); i+=1
        grid.addWidget(QtWid.QLabel('Window:')               , i, 0, 1, 3); i+=1
        grid.addWidget(self.qlin_filt_BS_window              , i, 0, 1, 3)
        grid.setAlignment(QtCore.Qt.AlignTop)
        
        qgrp_controls_filt_BS = QtWid.QGroupBox("Filter design")
        qgrp_controls_filt_BS.setLayout(grid)
        
        self.qpbt_filt_BS_coupling.clicked.connect(
                self.process_filt_BS_coupling)
        self.qlin_filt_BS_DC_cutoff.editingFinished.connect(
                self.process_filt_BS_coupling)        
        self.populate_filt_BS_design_controls()        
        self.qtbl_filt_BS.cellChanged.connect(
                self.process_qtbl_filt_BS_cellChanged)
        self.qtbl_filt_BS_cellChanged_lock = False  # Ignore cellChanged event
                                                    # when locked
        
        # -----------------------------------
        # -----------------------------------
        #   Round up tab page 'Filter response: band-stop'
        # -----------------------------------
        # -----------------------------------
        
        grid = QtWid.QGridLayout(spacing=0)
        grid.addWidget(qgrp_controls_filt_BS, 0, 0, QtCore.Qt.AlignTop)
        grid.addWidget(self.gw_filt_resp_BS, 0, 1)
        grid.setColumnStretch(1, 1)
        self.tab_filter_1_design.setLayout(grid)
        
        def _frame_LP_filter_response(): pass # Spider IDE outline bookmark
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        #
        #   TAB PAGE: Filter response low-pass
        #
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        
        # Plot: Filter response low-pass
        self.gw_filt_resp_LP = pg.GraphicsWindow()
        self.gw_filt_resp_LP.setBackground([20, 20, 20])        
        self.pi_filt_resp_LP = self.gw_filt_resp_LP.addPlot()
        
        p = {'color': '#BBB', 'font-size': '10pt'}
        self.pi_filt_resp_LP.showGrid(x=1, y=1)
        self.pi_filt_resp_LP.setTitle('Filter response', **p)
        self.pi_filt_resp_LP.setLabel('bottom', text='frequency [Hz]', **p)
        self.pi_filt_resp_LP.setLabel('left', text='attenuation [dB]', **p)
        self.pi_filt_resp_LP.setAutoVisible(x=True, y=True)
        self.pi_filt_resp_LP.enableAutoRange('x', False)
        self.pi_filt_resp_LP.enableAutoRange('y', True)
        self.pi_filt_resp_LP.setClipToView(True)
        self.pi_filt_resp_LP.setLimits(xMin=0, xMax=self.lockin.config.F_Nyquist)
        
        self.curve_filt_resp_LP = pg.PlotCurveItem(pen=self.PEN_03,
                                                   brush=self.BRUSH_03)
        self.pi_filt_resp_LP.addItem(self.curve_filt_resp_LP)
        self.update_plot_filt_resp_LP()
        
        # -----------------------------------
        # -----------------------------------
        #   Round up tab page 'Filter response: band-stop'
        # -----------------------------------
        # -----------------------------------
        
        hbox = QtWid.QHBoxLayout()
        hbox.addWidget(self.gw_filt_resp_LP, stretch=1)
        self.tab_filter_2_design.setLayout(hbox)
        
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
    
    @QtCore.pyqtSlot()
    def update_GUI(self):
        #dprint("Update GUI")
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
        
        if LIA_pyqt.firf_BS_sig_I.has_settled:
            self.LED_settled_BG_filter.setChecked(True)
            self.LED_settled_BG_filter.setText("YES")
        else:
            self.LED_settled_BG_filter.setChecked(False)
            self.LED_settled_BG_filter.setText("NO")
        if LIA_pyqt.firf_LP_mix_X.has_settled:
            self.LED_settled_LP_filter.setChecked(True)
            self.LED_settled_LP_filter.setText("YES")
        else:
            self.LED_settled_LP_filter.setChecked(False)
            self.LED_settled_LP_filter.setText("NO")
            
        self.update_chart_refsig()
        self.update_chart_filt_BS()
        self.update_chart_mixer()
        self.update_chart_LIA_output()
        self.update_plot_PS()
    
    @QtCore.pyqtSlot()
    def clear_chart_histories_stage_1_and_2(self):
        self.CH_filt_BS_in.clear()
        self.CH_filt_BS_out.clear()
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
            print("WARNING: Low-pass filter cannot reach desired cut-off freq.")
            f_cutoff = self.lockin.config.F_Nyquist - roll_off_width
        
        self.lockin_pyqt.firf_LP_mix_X.design_fir_filter(cutoff=f_cutoff)
        self.lockin_pyqt.firf_LP_mix_Y.design_fir_filter(cutoff=f_cutoff)
        self.update_plot_filt_resp_LP()
        
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
    def process_chkbs_legend_box_filt_BS(self):
        if self.lockin.lockin_paused:
            self.update_chart_filt_BS()     # Force update graph
            
    @QtCore.pyqtSlot()
    def process_chkbs_legend_box_mixer(self):
        if self.lockin.lockin_paused:
            self.update_chart_mixer()       # Force update graph
            
    @QtCore.pyqtSlot()
    def process_chkbs_legend_box_PS(self):
        if self.lockin.lockin_paused:
            self.update_plot_PS()           # Force update graph

    @QtCore.pyqtSlot()
    def process_qpbt_fullrange_xy(self):
        self.process_qpbt_autorange_x()
        self.pi_refsig.setYRange (-3.3, 3.3, padding=0.05)
        self.pi_filt_BS.setYRange(-3.3, 3.3, padding=0.05)
        self.pi_mixer.setYRange  (-5, 5, padding=0.05)
        
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
                      self.pi_filt_BS,
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
                      self.pi_filt_BS,
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
    def update_chart_filt_BS(self):
        [CH.update_curve() for CH in self.CHs_filt_BS]
        for i in range(2):
            self.CHs_filt_BS[i].curve.setVisible(
                    self.legend_box_filt_BS.chkbs[i].isChecked())
            
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
        self.BP_PS_1.update_curve()
        self.BP_PS_2.update_curve()
        self.BP_PS_1.curve.setVisible(self.legend_box_PS.chkbs[0].isChecked())
        self.BP_PS_2.curve.setVisible(self.legend_box_PS.chkbs[1].isChecked())

    def construct_title_plot_filt_resp(self, firf):
        __tmp1 = 'N_taps = %i' % firf.N_taps
        if isinstance(firf.window, str):
            __tmp2 = '%s' % firf.window
        else:
            __tmp2 = '%s' % [x for x in firf.window]
        __tmp3 = '%s Hz' % [round(x, 1) for x in firf.cutoff]        
        
        return ('%s, %s, %s' % (__tmp1, __tmp2, __tmp3))
    
    @QtCore.pyqtSlot()
    def process_filt_BS_coupling(self):
        DC_cutoff = self.qlin_filt_BS_DC_cutoff.text()
        try:
            DC_cutoff = float(DC_cutoff)
            DC_cutoff = round(DC_cutoff*10)/10;
        except:
            DC_cutoff = 0
        if DC_cutoff <= 0: DC_cutoff = 1.0
        self.qlin_filt_BS_DC_cutoff.setText("%.1f" % DC_cutoff)
        
        cutoff = self.lockin_pyqt.firf_BS_sig_I.cutoff
        if self.qpbt_filt_BS_coupling.isChecked():
            pass_zero = True
            if not self.lockin_pyqt.firf_BS_sig_I.pass_zero:
                cutoff = cutoff[1:]
        else:
            pass_zero = False
            if self.lockin_pyqt.firf_BS_sig_I.pass_zero:
                cutoff = np.insert(cutoff, 0, DC_cutoff)
            else:
                cutoff[0] = DC_cutoff
        
        self.lockin_pyqt.firf_BS_sig_I.design_fir_filter(cutoff=cutoff,
                                                         pass_zero=pass_zero)
        self.update_filt_BS_design()
        
    @QtCore.pyqtSlot(int, int)
    def process_qtbl_filt_BS_cellChanged(self, k, l):
        if self.qtbl_filt_BS_cellChanged_lock:
            return
        #print("cellChanged %i %i" % (k, l))
        
        # Construct the cutoff list
        if self.lockin_pyqt.firf_BS_sig_I.pass_zero:
            # Input coupling: DC
            cutoff = np.array([])
        else:
            # Input coupling: AC
            cutoff = self.lockin_pyqt.firf_BS_sig_I.cutoff[0]

        for row in range(self.qtbl_filt_BS.rowCount()):
            for col in range(self.qtbl_filt_BS.columnCount()):
                value = self.qtbl_filt_BS.item(row, col).text()
                try:
                    value = float(value)
                    value = round(value*10)/10
                    cutoff = np.append(cutoff, value)
                except ValueError:
                    value = None
        
        self.lockin_pyqt.firf_BS_sig_I.design_fir_filter(cutoff=cutoff)
        self.update_filt_BS_design()
    
    def update_filt_BS_design(self):
        self.populate_filt_BS_design_controls()
        self.lockin_pyqt.firf_BS_sig_I.calc_freqz_response()
        self.update_plot_filt_resp_BS()
    
    def populate_filt_BS_design_controls(self):
        freq_list = self.lockin_pyqt.firf_BS_sig_I.cutoff;
        
        if self.lockin_pyqt.firf_BS_sig_I.pass_zero:
            self.qpbt_filt_BS_coupling.setText("DC")
            self.qpbt_filt_BS_coupling.setChecked(True)
            self.qlin_filt_BS_DC_cutoff.setEnabled(False)
            self.qlin_filt_BS_DC_cutoff.setReadOnly(True)
        else:
            self.qpbt_filt_BS_coupling.setText("AC")
            self.qpbt_filt_BS_coupling.setChecked(False)
            self.qlin_filt_BS_DC_cutoff.setText("%.1f" % freq_list[0])
            self.qlin_filt_BS_DC_cutoff.setEnabled(True)
            self.qlin_filt_BS_DC_cutoff.setReadOnly(False)
            freq_list = freq_list[1:]
        
        self.qtbl_filt_BS_cellChanged_lock = True
        for row in range(self.qtbl_filt_BS.rowCount()):
            for col in range(self.qtbl_filt_BS.columnCount()):
                try:
                    freq = freq_list[row*2 + col]
                    freq_str = "%.1f" % freq
                except IndexError:
                    freq_str = ""
                self.qtbl_filt_BS_items[row*2 + col].setText(freq_str)
        self.qtbl_filt_BS_cellChanged_lock = False
    
    @QtCore.pyqtSlot()
    def update_plot_filt_resp_BS(self):
        firf = self.lockin_pyqt.firf_BS_sig_I
        self.curve_filt_resp_BS.setFillLevel(np.min(firf.resp_ampl_dB))
        self.curve_filt_resp_BS.setData(firf.resp_freq_Hz,
                                        firf.resp_ampl_dB)
        self.pi_filt_resp_BS.setXRange(firf.resp_freq_Hz__ROI_start,
                                       firf.resp_freq_Hz__ROI_end,
                                       padding=0.01)
        self.pi_filt_resp_BS.setTitle('Filter response: band-stop<br/>%s' %
                                      self.construct_title_plot_filt_resp(firf))
    @QtCore.pyqtSlot()
    def update_plot_filt_resp_LP(self):
        firf = self.lockin_pyqt.firf_LP_mix_X
        self.curve_filt_resp_LP.setFillLevel(np.min(firf.resp_ampl_dB))
        self.curve_filt_resp_LP.setData(firf.resp_freq_Hz,
                                        firf.resp_ampl_dB)
        self.pi_filt_resp_LP.setXRange(firf.resp_freq_Hz__ROI_start,
                                       firf.resp_freq_Hz__ROI_end,
                                       padding=0.01)
        self.pi_filt_resp_LP.setTitle('Filter response: low-pass<br/>%s' %
                                      self.construct_title_plot_filt_resp(firf))
        
    def power_spectrum_zoom(self, xRangeLo, xRangeHi):
        self.pi_PS.setXRange(xRangeLo, xRangeHi, padding=0.02)
        self.pi_PS.enableAutoRange('y', True)
        self.pi_PS.enableAutoRange('y', False)
        
if __name__ == "__main__":
    exec(open("DvG_Arduino_lockin_amp.py").read())