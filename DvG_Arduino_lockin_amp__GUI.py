#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Arduino lock-in amplifier
"""
__author__      = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__         = "https://github.com/Dennis-van-Gils/DvG_Arduino_lock-in_amp"
__date__        = "14-03-2019"
__version__     = "1.0.0"

from PyQt5 import QtCore, QtGui
from PyQt5 import QtWidgets as QtWid
from PyQt5.QtCore import QDateTime
import pyqtgraph as pg
import numpy as np

from DvG_pyqt_ChartHistory import ChartHistory
from DvG_pyqt_controls     import (create_Toggle_button,
                                   SS_GROUP,
                                   SS_TEXTBOX_READ_ONLY)
from DvG_pyqt_FileLogger   import FileLogger

import DvG_dev_Arduino_lockin_amp__fun_serial as lockin_functions
import DvG_dev_Arduino_lockin_amp__pyqt_lib   as lockin_pyqt_lib

# Constants
UPDATE_INTERVAL_WALL_CLOCK = 50  # 50 [ms]
CHART_HISTORY_TIME = 10          # 10  [s]

# Monkey patch error in pyqtgraph
import DvG_fix_pyqtgraph_PlotCurveItem
pg.PlotCurveItem.paintGL = DvG_fix_pyqtgraph_PlotCurveItem.paintGL

# DvG 21-01-2019: THE TRICK!!! GUI no longer slows down to a crawl when
# plotting massive data in curves
try:
    import OpenGL.GL as gl
    gl.glEnable(gl.GL_STENCIL_TEST)
    pg.setConfigOptions(useOpenGL=True)
    pg.setConfigOptions(enableExperimental=True)
    print("Enabled OpenGL hardware acceleration for graphing.")
except:
    #raise
    print("WARNING: Could not initiate OpenGL.")
    """
    print("Graphing will not be hardware accelerated.")
    print("Check if prerequisite 'PyOpenGL' library is installed.")
    print("Also, the videocard might not support stencil buffers.\n")
    """

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
        
        self.setGeometry(50, 50, 900, 800)
        self.setWindowTitle("Arduino lock-in amplifier")
        self.setStyleSheet(SS_TEXTBOX_READ_ONLY)
        
        # Define styles for plotting curves
        PEN_01 = pg.mkPen(color=[255, 30 , 180], width=3)
        PEN_02 = pg.mkPen(color=[255, 255, 90 ], width=3)
        PEN_03 = pg.mkPen(color=[0  , 255, 255], width=3)
        PEN_04 = pg.mkPen(color=[255, 255, 255], width=3)        
        BRUSH_03 = pg.mkBrush(0, 255, 255, 64)

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
        
        # -----------------------------------
        # -----------------------------------
        #   FRAME: Tabs
        # -----------------------------------
        # -----------------------------------
        
        self.tabs = QtWid.QTabWidget()
        self.tab_main           = QtWid.QWidget()
        self.tab_power_spectrum = QtWid.QWidget()
        self.tab_filter_1_resp = QtWid.QWidget()
        self.tab_filter_2_resp = QtWid.QWidget()
        self.tab_mcu_board_info = QtWid.QWidget()
        
        self.tabs.addTab(self.tab_main          , "Main")
        self.tabs.addTab(self.tab_power_spectrum, "Power spectrum")
        self.tabs.addTab(self.tab_filter_1_resp , "Filter response: band-stop")
        self.tabs.addTab(self.tab_filter_2_resp , "Filter response: low-pass")
        self.tabs.addTab(self.tab_mcu_board_info, "MCU board info")

        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        #
        #   TAB PAGE: Main
        #
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------

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
        self.pi_refsig.setLabel('bottom', text='time (ms)', **p)
        self.pi_refsig.setLabel('left', text='voltage (V)', **p)
        self.pi_refsig.setXRange(-lockin.config.BUFFER_SIZE * 
                                 lockin.config.ISR_CLOCK * 1e3,
                                 0, padding=0.01)
        self.pi_refsig.setYRange(1, 3, padding=0.05)
        self.pi_refsig.setAutoVisible(x=True, y=True)
        self.pi_refsig.setClipToView(True)

        self.CH_ref_X = ChartHistory(lockin.config.BUFFER_SIZE,
                                     self.pi_refsig.plot(pen=PEN_01))
        self.CH_ref_Y = ChartHistory(lockin.config.BUFFER_SIZE,
                                     self.pi_refsig.plot(pen=PEN_02))
        self.CH_sig_I = ChartHistory(lockin.config.BUFFER_SIZE,
                                     self.pi_refsig.plot(pen=PEN_03))
        self.CH_ref_X.x_axis_divisor = 1000     # From [us] to [ms]
        self.CH_ref_Y.x_axis_divisor = 1000     # From [us] to [ms]
        self.CH_sig_I.x_axis_divisor = 1000     # From [us] to [ms]
        self.CHs_refsig = [self.CH_ref_X, self.CH_ref_Y, self.CH_sig_I]

        # Legend
        vb = self.gw_refsig.addViewBox(enableMenu=False)
        vb.setMaximumWidth(80)
        #vb.setSizePolicy(QtWid.QSizePolicy.Minimum, QtWid.QSizePolicy.Minimum)
        legend = pg.LegendItem()
        legend.setParentItem(vb)
        legend.anchor((0,0), (0,0), offset=(1, 10))
        legend.setFixedWidth(75)
        legend.setScale(1)
        legend.addItem(self.CH_ref_X.curve, name='ref_X')
        legend.addItem(self.CH_ref_Y.curve, name='ref_Y')
        legend.addItem(self.CH_sig_I.curve, name='sig_I')

        # On/off
        self.qpbt_ENA_lockin = create_Toggle_button("lock-in OFF")
        self.qpbt_ENA_lockin.clicked.connect(self.process_qpbt_ENA_lockin)

        # QGROUP: Reference signal
        num_chars = 8  # Limit the width of the textboxes to N characters wide
        e = QtGui.QLineEdit()
        w = (8 + num_chars * e.fontMetrics().width('x') + 
             e.textMargins().left()     + e.textMargins().right() + 
             e.contentsMargins().left() + e.contentsMargins().right())
        del e
        
        p1 = {'maximumWidth': w}
        p2 = {**p1, 'readOnly': True}
        self.qlin_set_ref_freq = (
                QtWid.QLineEdit("%.2f" % lockin.config.ref_freq, **p1))
        self.qlin_read_ref_freq = (
                QtWid.QLineEdit("%.2f" % lockin.config.ref_freq, **p2))
        self.qlin_set_ref_V_offset = (
                QtWid.QLineEdit("%.2f" % lockin.config.ref_V_offset, **p1))
        self.qlin_read_ref_V_offset = (
                QtWid.QLineEdit("%.2f" % lockin.config.ref_V_offset, **p2))
        self.qlin_set_ref_V_ampl = (
                QtWid.QLineEdit("%.2f" % lockin.config.ref_V_ampl, **p1))
        self.qlin_read_ref_V_ampl = (
                QtWid.QLineEdit("%.2f" % lockin.config.ref_V_ampl, **p2))

        self.qlin_set_ref_V_offset.editingFinished.connect(
                self.process_qlin_set_ref_V_offset)
        self.qlin_set_ref_V_ampl.editingFinished.connect(
                self.process_qlin_set_ref_V_ampl)
        
        p  = {'alignment': QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight}
        p2 = {'alignment': QtCore.Qt.AlignVCenter | QtCore.Qt.AlignHCenter}
        i = 0;
        grid = QtWid.QGridLayout()
        grid.setVerticalSpacing(4)
        grid.addWidget(QtWid.QLabel("ref_X: cosine  |  ref_Y: sine", **p2)
                                                     , i, 0, 1, 4); i+=1
        grid.addWidget(QtWid.QLabel("Analog voltage limits: [0, %.2f] V" %
                                    lockin.config.A_REF, **p2), i, 0, 1, 4); i+=1
        grid.addItem(QtWid.QSpacerItem(0, 4)         , i, 0); i+=1
        grid.addWidget(QtWid.QLabel("Set")           , i, 1)
        grid.addWidget(QtWid.QLabel("Read")          , i, 2); i+=1
        grid.addWidget(QtWid.QLabel("freq:", **p)    , i, 0)
        grid.addWidget(self.qlin_set_ref_freq        , i, 1)
        grid.addWidget(self.qlin_read_ref_freq       , i, 2)
        grid.addWidget(QtWid.QLabel("Hz")            , i, 3); i+=1
        grid.addWidget(QtWid.QLabel("V_offset:", **p), i, 0)
        grid.addWidget(self.qlin_set_ref_V_offset    , i, 1)
        grid.addWidget(self.qlin_read_ref_V_offset   , i, 2)
        grid.addWidget(QtWid.QLabel("V")             , i, 3); i+=1
        grid.addWidget(QtWid.QLabel("V_ampl:", **p)  , i, 0)
        grid.addWidget(self.qlin_set_ref_V_ampl      , i, 1)
        grid.addWidget(self.qlin_read_ref_V_ampl     , i, 2)
        grid.addWidget(QtWid.QLabel("V")             , i, 3)
        
        qgrp_refsig = QtWid.QGroupBox("Reference signal")
        qgrp_refsig.setStyleSheet(SS_GROUP)
        qgrp_refsig.setLayout(grid)

        # QGROUP: Readings
        p = {'layoutDirection': QtCore.Qt.LeftToRight}
        self.chkbs_refsig = [
                QtWid.QCheckBox("ref_X [0]:", **p, checked=True),
                QtWid.QCheckBox("ref_Y [0]:", **p, checked=False),
                QtWid.QCheckBox("sig_I [0]:", **p, checked=True)]
        ([chkb.clicked.connect(self.process_chkbs_refsig) for chkb
          in self.chkbs_refsig])
        
        p = {'readOnly': True, 'maximumWidth': 100}
        self.qlin_time  = QtWid.QLineEdit(**p)
        self.qlin_ref_X = QtWid.QLineEdit(**p)
        self.qlin_ref_Y = QtWid.QLineEdit(**p)
        self.qlin_sig_I = QtWid.QLineEdit(**p)
        
        self.qpbt_full_axes = QtWid.QPushButton("full axes")
        self.qpbt_full_axes.clicked.connect(self.process_qpbt_full_axes)
        self.qpbt_autoscale_y = QtWid.QPushButton("autoscale y-axis")
        self.qpbt_autoscale_y.clicked.connect(self.process_qpbtn_autoscale_y)

        i = 0
        grid = QtWid.QGridLayout()
        grid.setVerticalSpacing(4)
        grid.addWidget(QtWid.QLabel("time [0]:"), i, 0)
        grid.addWidget(self.qlin_time           , i, 1)
        grid.addWidget(QtWid.QLabel("us")       , i, 2); i+=1
        grid.addWidget(self.chkbs_refsig[0]     , i, 0)
        grid.addWidget(self.qlin_ref_X          , i, 1)
        grid.addWidget(QtWid.QLabel("V")        , i, 2); i+=1
        grid.addWidget(self.chkbs_refsig[1]     , i, 0)
        grid.addWidget(self.qlin_ref_Y          , i, 1)
        grid.addWidget(QtWid.QLabel("V")        , i, 2); i+=1
        grid.addWidget(self.chkbs_refsig[2]     , i, 0)
        grid.addWidget(self.qlin_sig_I          , i, 1)
        grid.addWidget(QtWid.QLabel("V")        , i, 2); i+=1
        grid.addItem(QtWid.QSpacerItem(0, 4)    , i, 0); i+=1
        grid.addWidget(self.qpbt_full_axes      , i, 0, 1, 2); i+=1
        grid.addWidget(self.qpbt_autoscale_y    , i, 0, 1, 2)
        grid.setAlignment(QtCore.Qt.AlignTop)

        qgrp_readings = QtWid.QGroupBox("Readings")
        qgrp_readings.setStyleSheet(SS_GROUP)
        qgrp_readings.setLayout(grid)
        
        # Round up frame
        vbox_refsig = QtWid.QVBoxLayout()
        vbox_refsig.addWidget(self.qpbt_ENA_lockin)
        vbox_refsig.addWidget(qgrp_refsig)
        vbox_refsig.addWidget(qgrp_readings)
        vbox_refsig.addStretch()

        hbox_refsig = QtWid.QHBoxLayout()
        hbox_refsig.addWidget(self.gw_refsig, stretch=1)
        hbox_refsig.addLayout(vbox_refsig)
        
        # -----------------------------------
        # -----------------------------------
        #   FRAME: Mixer and filters
        # -----------------------------------
        # -----------------------------------
        
        # Chart: Filter
        self.gw_filt_BS = pg.GraphicsWindow()
        self.gw_filt_BS.setBackground([20, 20, 20])
        self.pi_filt_BS = self.gw_filt_BS.addPlot()
        
        p = {'color': '#BBB', 'font-size': '10pt'}
        self.pi_filt_BS.showGrid(x=1, y=1)
        self.pi_filt_BS.setTitle('Band-stop filter acting on sig_I', **p)
        self.pi_filt_BS.setLabel('bottom', text='time (ms)', **p)
        self.pi_filt_BS.setLabel('left', text='voltage (V)', **p)
        self.pi_filt_BS.setXRange(-lockin.config.BUFFER_SIZE *
                                    lockin.config.ISR_CLOCK * 1e3,
                                    0, padding=0.01)
        self.pi_filt_BS.setYRange(-1, 1, padding=0.05)
        self.pi_filt_BS.setAutoVisible(x=True, y=True)
        self.pi_filt_BS.setClipToView(True)
        
        self.CH_filt_BS_in  = ChartHistory(lockin.config.BUFFER_SIZE,
                                           self.pi_filt_BS.plot(pen=PEN_03))
        self.CH_filt_BS_out = ChartHistory(lockin.config.BUFFER_SIZE,
                                           self.pi_filt_BS.plot(pen=PEN_04))
        self.CH_filt_BS_in.x_axis_divisor = 1000     # From [us] to [ms]
        self.CH_filt_BS_out.x_axis_divisor = 1000    # From [us] to [ms]
        self.CHs_filt_BS = [self.CH_filt_BS_in, self.CH_filt_BS_out]
        
        # Legend
        vb = self.gw_filt_BS.addViewBox(enableMenu=False)
        vb.setMaximumWidth(80)
        #vb.setSizePolicy(QtWid.QSizePolicy.Minimum, QtWid.QSizePolicy.Minimum)
        legend = pg.LegendItem()
        legend.setParentItem(vb)
        legend.anchor((0,0), (0,0), offset=(1, 10))
        legend.setFixedWidth(75)
        legend.setScale(1)
        legend.addItem(self.CH_filt_BS_in.curve, name='sig_I')
        legend.addItem(self.CH_filt_BS_out.curve, name='out')
        
         # Chart: Mixer
        self.gw_mixer = pg.GraphicsWindow()
        self.gw_mixer.setBackground([20, 20, 20])
        self.pi_mixer = self.gw_mixer.addPlot()
        
        p = {'color': '#BBB', 'font-size': '10pt'}
        self.pi_mixer.showGrid(x=1, y=1)
        self.pi_mixer.setTitle('Mixer', **p)
        self.pi_mixer.setLabel('bottom', text='time (ms)', **p)
        self.pi_mixer.setLabel('left', text='AC ampl. (V%s)' % chr(0xb2), 
                               **p)
        self.pi_mixer.setXRange(-lockin.config.BUFFER_SIZE *
                                lockin.config.ISR_CLOCK * 1e3,
                                 0, padding=0.01)
        self.pi_mixer.setYRange(-1, 1, padding=0.05)
        self.pi_mixer.setAutoVisible(x=True, y=True)
        self.pi_mixer.setClipToView(True)
        
        self.CH_mix_X = ChartHistory(lockin.config.BUFFER_SIZE,
                                     self.pi_mixer.plot(pen=PEN_01))
        self.CH_mix_Y = ChartHistory(lockin.config.BUFFER_SIZE,
                                     self.pi_mixer.plot(pen=PEN_02))
        self.CH_mix_X.x_axis_divisor = 1000     # From [us] to [ms]
        self.CH_mix_Y.x_axis_divisor = 1000     # From [us] to [ms]
        self.CHs_mixer = [self.CH_mix_X, self.CH_mix_Y]
        
        # Legend
        vb = self.gw_mixer.addViewBox(enableMenu=False)
        vb.setMaximumWidth(80)
        #vb.setSizePolicy(QtWid.QSizePolicy.Minimum, QtWid.QSizePolicy.Minimum)
        legend = pg.LegendItem()
        legend.setParentItem(vb)
        legend.anchor((0,0), (0,0), offset=(1, 10))
        legend.setFixedWidth(75)
        legend.setScale(1)
        legend.addItem(self.CH_mix_X.curve, name='mix_X')
        legend.addItem(self.CH_mix_Y.curve, name='mix_Y')
        
        # Round up frame
        hbox_mixer = QtWid.QHBoxLayout()
        hbox_mixer.addWidget(self.gw_filt_BS, stretch=1)
        hbox_mixer.addWidget(self.gw_mixer, stretch=1)

        # -----------------------------------
        # -----------------------------------
        #   FRAME: LIA output amplitude and phase
        # -----------------------------------
        # -----------------------------------
        
        # Chart: Amplitude
        self.gw_LIA_output = pg.GraphicsWindow()
        self.gw_LIA_output.setBackground([20, 20, 20])
        self.pi_LIA_amp = self.gw_LIA_output.addPlot()
        
        p = {'color': '#BBB', 'font-size': '10pt'}
        self.pi_LIA_amp.showGrid(x=1, y=1)
        self.pi_LIA_amp.setTitle('Lock-in output X: amplitude', **p)
        self.pi_LIA_amp.setLabel('bottom', text='time (ms)', **p)
        self.pi_LIA_amp.setLabel('left', text='voltage (V)', **p)
        self.pi_LIA_amp.setXRange(-lockin.config.BUFFER_SIZE *
                                  lockin.config.ISR_CLOCK * 1e3,
                                  0, padding=0.01)
        self.pi_LIA_amp.setYRange(-1, 1, padding=0.05)
        self.pi_LIA_amp.setAutoVisible(x=True, y=True)
        self.pi_LIA_amp.setClipToView(True)
        
        self.CH_LIA_amp = ChartHistory(lockin.config.BUFFER_SIZE,
                                       self.pi_LIA_amp.plot(pen=PEN_03))
        self.CH_LIA_amp.x_axis_divisor = 1000     # From [us] to [ms]
        
        # Chart: phase
        self.pi_LIA_phi = self.gw_LIA_output.addPlot()
        
        p = {'color': '#BBB', 'font-size': '10pt'}
        self.pi_LIA_phi.showGrid(x=1, y=1)
        self.pi_LIA_phi.setTitle('Lock-in output Y: phase', **p)
        self.pi_LIA_phi.setLabel('bottom', text='time (ms)', **p)
        self.pi_LIA_phi.setLabel('left', text='phase (deg)', **p)
        self.pi_LIA_phi.setXRange(-lockin.config.BUFFER_SIZE *
                                   lockin.config.ISR_CLOCK * 1e3,
                                   0, padding=0.01)
        self.pi_LIA_phi.setYRange(-1, 1, padding=0.05)
        self.pi_LIA_phi.setAutoVisible(x=True, y=True)
        self.pi_LIA_phi.setClipToView(True)
        
        self.CH_LIA_phi = ChartHistory(lockin.config.BUFFER_SIZE,
                                       self.pi_LIA_phi.plot(pen=PEN_03))
        self.CH_LIA_phi.x_axis_divisor = 1000     # From [us] to [ms]
        self.CHs_LIA_output = [self.CH_LIA_amp, self.CH_LIA_phi]
        
        # Round up frame
        hbox_LIA_output = QtWid.QHBoxLayout()
        hbox_LIA_output.addWidget(self.gw_LIA_output, stretch=1)

        # -----------------------------------
        # -----------------------------------
        #   Round up tab page 'Main'
        # -----------------------------------
        # -----------------------------------
        
        vbox = QtWid.QVBoxLayout()
        vbox.addLayout(hbox_refsig, stretch=1)
        vbox.addLayout(hbox_mixer, stretch=1)
        vbox.addLayout(hbox_LIA_output, stretch=1)
        self.tab_main.setLayout(vbox)
        
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
        self.pi_filt_resp_BS.setLabel('bottom', text='frequency (Hz)', **p)
        self.pi_filt_resp_BS.setLabel('left', text='attenuation (dB)', **p)
        self.pi_filt_resp_BS.setAutoVisible(x=True, y=True)
        self.pi_filt_resp_BS.enableAutoRange('x', False)
        self.pi_filt_resp_BS.enableAutoRange('y', True)
        self.pi_filt_resp_BS.setClipToView(True)
        
        self.curve_filt_resp_BS = pg.PlotCurveItem(pen=PEN_03, brush=BRUSH_03)
        self.pi_filt_resp_BS.addItem(self.curve_filt_resp_BS)
        
        # -----------------------------------
        # -----------------------------------
        #   Round up tab page 'Filter response: band-stop'
        # -----------------------------------
        # -----------------------------------
        
        hbox = QtWid.QHBoxLayout()
        hbox.addWidget(self.gw_filt_resp_BS, stretch=1)
        self.tab_filter_1_resp.setLayout(hbox)
        
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
        self.pi_filt_resp_LP.setLabel('bottom', text='frequency (Hz)', **p)
        self.pi_filt_resp_LP.setLabel('left', text='attenuation (dB)', **p)
        self.pi_filt_resp_LP.setAutoVisible(x=True, y=True)
        self.pi_filt_resp_LP.enableAutoRange('x', False)
        self.pi_filt_resp_LP.enableAutoRange('y', True)
        self.pi_filt_resp_LP.setClipToView(True)
        
        self.curve_filt_resp_LP = pg.PlotCurveItem(pen=PEN_03, brush=BRUSH_03)
        self.pi_filt_resp_LP.addItem(self.curve_filt_resp_LP)
        
        # -----------------------------------
        # -----------------------------------
        #   Round up tab page 'Filter response: band-stop'
        # -----------------------------------
        # -----------------------------------
        
        hbox = QtWid.QHBoxLayout()
        hbox.addWidget(self.gw_filt_resp_LP, stretch=1)
        self.tab_filter_2_resp.setLayout(hbox)
        
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        #
        #   Round up full window
        #
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------

        vbox = QtWid.QVBoxLayout(self)
        vbox.addLayout(hbox_header)
        vbox.addWidget(self.tabs)
        
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
        
        self.lockin_pyqt.signal_ref_freq_is_set.connect(
                self.update_qlin_read_ref_freq)
        self.lockin_pyqt.signal_ref_V_offset_is_set.connect(
                self.update_qlin_read_ref_V_offset)
        self.lockin_pyqt.signal_ref_V_ampl_is_set.connect(
                self.update_qlin_read_ref_V_ampl)
    
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
    def process_qpbt_ENA_lockin(self):
        if self.qpbt_ENA_lockin.isChecked():
            self.lockin_pyqt.turn_on()
            self.qpbt_ENA_lockin.setText("lock-in ON")
        else:
            self.lockin_pyqt.turn_off()
            self.qlbl_DAQ_rate.setText("Buffers/s: paused")
            self.qpbt_ENA_lockin.setText("lock-in OFF")
        
    @QtCore.pyqtSlot()
    def process_qpbt_record(self):
        if self.qpbt_record.isChecked():
            self.file_logger.starting = True
            self.file_logger.stopping = False
        else:
            self.file_logger.starting = False
            self.file_logger.stopping = True

    @QtCore.pyqtSlot(str)
    def set_text_qpbt_record(self, text_str):
        self.qpbt_record.setText(text_str)
            
    @QtCore.pyqtSlot()
    def process_qlin_set_ref_V_offset(self):
        try:
            ref_V_offset = float(self.qlin_set_ref_V_offset.text())
        except ValueError:
            ref_V_offset = self.lockin.config.ref_V_offset
        
        # Clip between 0 and the analog voltage reference
        ref_V_offset = np.clip(ref_V_offset, 0, self.lockin.config.A_REF)
        
        self.qlin_set_ref_V_offset.setText("%.2f" % ref_V_offset)
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
        
        self.qlin_set_ref_V_ampl.setText("%.2f" % ref_V_ampl)
        if ref_V_ampl != self.lockin.config.ref_V_ampl:
            self.lockin_pyqt.set_ref_V_ampl(ref_V_ampl)
            QtWid.QApplication.processEvents()
        
    @QtCore.pyqtSlot()
    def update_qlin_read_ref_freq(self):
        self.qlin_read_ref_freq.setText("%.2f" %
                                        self.lockin.config.ref_freq)
        
    @QtCore.pyqtSlot()
    def update_qlin_read_ref_V_offset(self):
        self.qlin_read_ref_V_offset.setText("%.2f" %
                                            self.lockin.config.ref_V_offset)
        
    @QtCore.pyqtSlot()
    def update_qlin_read_ref_V_ampl(self):
        self.qlin_read_ref_V_ampl.setText("%.2f" %
                                          self.lockin.config.ref_V_ampl)

    @QtCore.pyqtSlot()
    def process_chkbs_refsig(self):
        if self.lockin.lockin_paused:
            self.update_chart_refsig()  # Force update graph

    @QtCore.pyqtSlot()
    def process_qpbt_full_axes(self):
        min_time = -(self.lockin.config.BUFFER_SIZE * 
                     self.lockin.config.ISR_CLOCK * 1e3)
        plot_items = [self.pi_refsig,
                      self.pi_filt_BS,
                      self.pi_mixer,
                      self.pi_LIA_amp,
                      self.pi_LIA_phi]
        for pi in plot_items:
            pi.setXRange(min_time, 0, padding=0.01)
        self.process_qpbtn_autoscale_y()

    @QtCore.pyqtSlot()
    def process_qpbtn_autoscale_y(self):
        plot_items = [self.pi_refsig,
                      self.pi_filt_BS,
                      self.pi_mixer,
                      self.pi_LIA_amp,
                      self.pi_LIA_phi]
        for pi in plot_items:
            pi.enableAutoRange('y', True)
            pi.enableAutoRange('y', False)
        
    
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    #   Update chart routines
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    

    @QtCore.pyqtSlot()
    def update_chart_refsig(self):
        [CH.update_curve() for CH in self.CHs_refsig]
        for i in range(3):
            self.CHs_refsig[i].curve.setVisible(
                    self.chkbs_refsig[i].isChecked())
            
    @QtCore.pyqtSlot()
    def update_chart_filt_BS(self):
        [CH.update_curve() for CH in self.CHs_filt_BS]
            
    @QtCore.pyqtSlot()
    def update_chart_mixer(self):
        [CH.update_curve() for CH in self.CHs_mixer]
        
    @QtCore.pyqtSlot()
    def update_chart_LIA_output(self):
        [CH.update_curve() for CH in self.CHs_LIA_output]
        
    def construct_title_plot_filt_resp(self, firf):
        __tmp1 = 'N_taps = %i' % firf.N_taps
        if isinstance(firf.firwin_window, str):
            __tmp2 = '%s' % firf.firwin_window
        else:
            __tmp2 = '%s' % [x for x in firf.firwin_window]
        __tmp3 = '%s Hz' % [round(x, 1) for x in firf.firwin_cutoff]        
        
        return ('%s, %s, %s' % (__tmp1, __tmp2, __tmp3))
    
    def update_plot_filt_resp_BS(self, firf):
        self.curve_filt_resp_BS.setFillLevel(np.min(firf.resp_ampl_dB))
        self.curve_filt_resp_BS.setData(firf.resp_freq_Hz,
                                        firf.resp_ampl_dB)
        self.pi_filt_resp_BS.setXRange(firf.resp_freq_Hz__ROI_start,
                                       firf.resp_freq_Hz__ROI_end,
                                       padding=0.01)
        self.pi_filt_resp_BS.setTitle('Filter response: band-stop<br/>%s' %
                                      self.construct_title_plot_filt_resp(firf))
        
    def update_plot_filt_resp_LP(self, firf):
        self.curve_filt_resp_LP.setFillLevel(np.min(firf.resp_ampl_dB))
        self.curve_filt_resp_LP.setData(firf.resp_freq_Hz,
                                        firf.resp_ampl_dB)
        self.pi_filt_resp_LP.setXRange(firf.resp_freq_Hz__ROI_start,
                                       firf.resp_freq_Hz__ROI_end,
                                       padding=0.01)
        self.pi_filt_resp_LP.setTitle('Filter response: low-pass<br/>%s' %
                                      self.construct_title_plot_filt_resp(firf))
        
if __name__ == "__main__":
    exec(open("DvG_Arduino_lockin_amp.py").read())