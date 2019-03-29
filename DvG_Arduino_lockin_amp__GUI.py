#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Arduino lock-in amplifier
"""
__author__      = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__         = "https://github.com/Dennis-van-Gils/DvG_Arduino_lock-in_amp"
__date__        = "28-03-2019"
__version__     = "1.0.0"

from PyQt5 import QtCore, QtGui
from PyQt5 import QtWidgets as QtWid
from PyQt5.QtCore import QDateTime
import pyqtgraph as pg
import numpy as np

from DvG_pyqt_ChartHistory import ChartHistory
from DvG_pyqt_controls     import (create_Toggle_button,
                                   create_LED_indicator_rect,
                                   SS_GROUP,
                                   SS_GROUP_BORDERLESS,
                                   SS_TEXTBOX_READ_ONLY)
from DvG_pyqt_FileLogger   import FileLogger
from DvG_debug_functions   import dprint

import DvG_dev_Arduino_lockin_amp__fun_serial as lockin_functions
import DvG_dev_Arduino_lockin_amp__pyqt_lib   as lockin_pyqt_lib

# Constants
UPDATE_INTERVAL_WALL_CLOCK = 50  # 50 [ms]

try:
    import OpenGL.GL as gl
    pg.setConfigOptions(useOpenGL=True)
    pg.setConfigOptions(enableExperimental=True)
    pg.setConfigOptions(antialias=False)
    
    # Monkey patch error in pyqtgraph
    import DvG_fix_pyqtgraph_PlotCurveItem
    pg.PlotCurveItem.paintGL = DvG_fix_pyqtgraph_PlotCurveItem.paintGL
    
    print("OpenGL hardware acceleration enabled.")
    USING_OPENGL = True
except:
    pg.setConfigOptions(useOpenGL=False)
    pg.setConfigOptions(enableExperimental=False)
    pg.setConfigOptions(antialias=False)
    print("WARNING: Could not enable OpenGL hardware acceleration.")
    print("Check if prerequisite 'PyOpenGL' library is installed.")
    USING_OPENGL = False

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
        
        self.setGeometry(50, 50, 1098, 960)
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
        self.tab_main  = QtWid.QWidget()
        self.tab_mixer = QtWid.QWidget()
        self.tab_power_spectrum = QtWid.QWidget()
        self.tab_filter_1_response = QtWid.QWidget()
        self.tab_filter_2_response = QtWid.QWidget()
        self.tab_mcu_board_info = QtWid.QWidget()
        
        self.tabs.addTab(self.tab_main             , "Main")
        self.tabs.addTab(self.tab_mixer            , "Mixer")
        self.tabs.addTab(self.tab_power_spectrum   , "Power spectrum")
        self.tabs.addTab(self.tab_filter_1_response, "Filter response: band-stop")
        self.tabs.addTab(self.tab_filter_2_response, "Filter response: low-pass")
        self.tabs.addTab(self.tab_mcu_board_info   , "MCU board info")

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
        #   FRAME: LIA output
        # -----------------------------------
        # -----------------------------------
        
        # Chart: X or R
        self.gw_XR = pg.GraphicsWindow()
        self.gw_XR.setBackground([20, 20, 20])
        self.pi_XR = self.gw_XR.addPlot()
        
        p = {'color': '#BBB', 'font-size': '10pt'}
        self.pi_XR.showGrid(x=1, y=1)
        self.pi_XR.setTitle('Amplitude R', **p)
        self.pi_XR.setLabel('bottom', text='time (ms)', **p)
        self.pi_XR.setLabel('left', text='voltage (V)', **p)
        self.pi_XR.setXRange(-lockin.config.BUFFER_SIZE *
                             lockin.config.ISR_CLOCK * 1e3,
                             0, padding=0.01)
        self.pi_XR.setYRange(-1, 1, padding=0.05)
        self.pi_XR.setAutoVisible(x=True, y=True)
        self.pi_XR.setClipToView(True)
        
        self.CH_LIA_XR = ChartHistory(lockin.config.BUFFER_SIZE,
                                      self.pi_XR.plot(pen=PEN_03))
        self.CH_LIA_XR.x_axis_divisor = 1000     # From [us] to [ms]
        
        # Chart: Y or T
        self.pi_YT = self.gw_XR.addPlot()
        
        p = {'color': '#BBB', 'font-size': '10pt'}
        self.pi_YT.showGrid(x=1, y=1)
        self.pi_YT.setTitle('Phase %s' % chr(0x398), **p)
        self.pi_YT.setLabel('bottom', text='time (ms)', **p)
        self.pi_YT.setLabel('left', text='phase (deg)', **p)
        self.pi_YT.setXRange(-lockin.config.BUFFER_SIZE *
                             lockin.config.ISR_CLOCK * 1e3,
                             0, padding=0.01)
        self.pi_YT.setYRange(-1, 1, padding=0.05)
        self.pi_YT.setAutoVisible(x=True, y=True)
        self.pi_YT.setClipToView(True)
        
        self.CH_LIA_YT = ChartHistory(lockin.config.BUFFER_SIZE,
                                      self.pi_YT.plot(pen=PEN_03))
        self.CH_LIA_YT.x_axis_divisor = 1000     # From [us] to [ms]
        self.CHs_LIA_output = [self.CH_LIA_XR, self.CH_LIA_YT]
        
        # QGROUP: Show X-Y or R-Theta
        self.qrbt_XR_X = QtWid.QRadioButton("X")
        self.qrbt_XR_R = QtWid.QRadioButton("R", checked=True)
        self.qrbt_YT_Y = QtWid.QRadioButton("Y")
        self.qrbt_YT_T = QtWid.QRadioButton("%s" % chr(0x398), checked=True)
        
        vbox = QtWid.QVBoxLayout(spacing=4)
        vbox.addWidget(self.qrbt_XR_X)
        vbox.addWidget(self.qrbt_XR_R)
        qgrp_XR = QtWid.QGroupBox()
        qgrp_XR.setStyleSheet(SS_GROUP_BORDERLESS)
        qgrp_XR.setLayout(vbox)
        
        vbox = QtWid.QVBoxLayout(spacing=4)
        vbox.addWidget(self.qrbt_YT_Y)
        vbox.addWidget(self.qrbt_YT_T)
        qgrp_YT = QtWid.QGroupBox()
        qgrp_YT.setStyleSheet(SS_GROUP_BORDERLESS)
        qgrp_YT.setLayout(vbox)
        
        grid = QtWid.QGridLayout(spacing=0)
        grid.addWidget(qgrp_XR, 0, 0, QtCore.Qt.AlignCenter)
        grid.addWidget(qgrp_YT, 0, 1, QtCore.Qt.AlignCenter)
        grid.setContentsMargins(0, 0, 0, 0)
        
        qgrp_XRYT = QtWid.QGroupBox("Show output")
        qgrp_XRYT.setStyleSheet(SS_GROUP)
        qgrp_XRYT.setLayout(grid)
        
        # QGROUP: Filters settled?
        self.LED_settled_BG_filter = create_LED_indicator_rect(False, 'NO')
        self.LED_settled_LP_filter = create_LED_indicator_rect(False, 'NO')
        
        grid = QtWid.QGridLayout(spacing=4)
        grid.addWidget(QtWid.QLabel("1: band-stop"), 0, 0)
        grid.addWidget(self.LED_settled_BG_filter  , 0, 1)
        grid.addWidget(QtWid.QLabel("2: low-pass") , 1, 0)
        grid.addWidget(self.LED_settled_LP_filter  , 1, 1)
        
        qgrp_settling = QtWid.QGroupBox("Filters settled?")
        qgrp_settling.setStyleSheet(SS_GROUP)
        qgrp_settling.setLayout(grid)
        
        # Combine QGroups
        vbox_qgrps = QtWid.QVBoxLayout()
        vbox_qgrps.addWidget(qgrp_XRYT, stretch=0)
        vbox_qgrps.addWidget(qgrp_settling, stretch=0)
        vbox_qgrps.addStretch(1)
        
        # Round up frame
        hbox_LIA_output = QtWid.QHBoxLayout()
        hbox_LIA_output.addWidget(self.gw_XR, stretch=1)
        hbox_LIA_output.addLayout(vbox_qgrps, stretch=0)

        # -----------------------------------
        # -----------------------------------
        #   Round up tab page 'Main'
        # -----------------------------------
        # -----------------------------------
        
        vbox = QtWid.QVBoxLayout()
        vbox.addLayout(hbox_refsig, stretch=1)
        vbox.addLayout(hbox_LIA_output, stretch=1)
        self.tab_main.setLayout(vbox)
        
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        #
        #   TAB PAGE: Mixer
        #
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        
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
        vbox_mixer = QtWid.QVBoxLayout()
        vbox_mixer.addWidget(self.gw_filt_BS, stretch=1)
        vbox_mixer.addWidget(self.gw_mixer, stretch=1)
        
        # -----------------------------------
        # -----------------------------------
        #   Round up tab page 'Mixer'
        # -----------------------------------
        # -----------------------------------
        
        #vbox = QtWid.QVBoxLayout()
        #vbox.addLayout(hbox_mixer, stretch=1)
        self.tab_mixer.setLayout(vbox_mixer)
        
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
        self.update_plot_filt_resp_BS()
        
        # -----------------------------------
        # -----------------------------------
        #   Round up tab page 'Filter response: band-stop'
        # -----------------------------------
        # -----------------------------------
        
        hbox = QtWid.QHBoxLayout()
        hbox.addWidget(self.gw_filt_resp_BS, stretch=1)
        self.tab_filter_1_response.setLayout(hbox)
        
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
        self.update_plot_filt_resp_LP()
        
        # -----------------------------------
        # -----------------------------------
        #   Round up tab page 'Filter response: band-stop'
        # -----------------------------------
        # -----------------------------------
        
        hbox = QtWid.QHBoxLayout()
        hbox.addWidget(self.gw_filt_resp_LP, stretch=1)
        self.tab_filter_2_response.setLayout(hbox)
        
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
        
        self.lockin_pyqt.signal_DAQ_updated.connect(
                self.update_GUI)
        
        self.lockin_pyqt.signal_DAQ_suspended.connect(
                self.update_GUI_paused)
        
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
        
        if not LIA_pyqt.worker_DAQ.suspended:
            self.qlbl_DAQ_rate.setText("Buffers/s: %.1f" % 
                                       LIA_pyqt.obtained_DAQ_rate_Hz)
            self.qlin_time.setText("%i"    % LIA_pyqt.state.time[0])
            self.qlin_ref_X.setText("%.4f" % LIA_pyqt.state.ref_X[0])
            self.qlin_ref_Y.setText("%.4f" % LIA_pyqt.state.ref_Y[0])
            self.qlin_sig_I.setText("%.4f" % LIA_pyqt.state.sig_I[0])
            
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
            
    @QtCore.pyqtSlot()
    def update_GUI_paused(self):
        #dprint("Update GUI paused")
        self.qlbl_DAQ_rate.setText("Buffers/s: paused")
    
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
            self.qpbt_ENA_lockin.setText("lock-in ON")
            
            self.lockin_pyqt.state.reset()
            self.clear_chart_histories_stage_1_and_2()
        else:
            self.lockin_pyqt.turn_off()
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
        
        self.lockin_pyqt.firf_LP_mix_X.update_firwin_cutoff([0, f_cutoff])
        self.lockin_pyqt.firf_LP_mix_Y.update_firwin_cutoff([0, f_cutoff])
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
                      self.pi_XR,
                      self.pi_YT]
        for pi in plot_items:
            pi.setXRange(min_time, 0, padding=0.01)
        self.process_qpbtn_autoscale_y()

    @QtCore.pyqtSlot()
    def process_qpbtn_autoscale_y(self):
        plot_items = [self.pi_refsig,
                      self.pi_filt_BS,
                      self.pi_mixer,
                      self.pi_XR,
                      self.pi_YT]
        for pi in plot_items:
            pi.enableAutoRange('y', True)
            pi.enableAutoRange('y', False)
        
    
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
        
if __name__ == "__main__":
    exec(open("DvG_Arduino_lockin_amp.py").read())