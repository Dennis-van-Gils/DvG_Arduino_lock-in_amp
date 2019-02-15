#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Arduino lock-in amplifier
"""
__author__      = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__         = "https://github.com/Dennis-van-Gils/DvG_Arduino_lock-in_amp"
__date__        = "21-01-2019"
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
    print("Graphing will not be hardware accelerated.")
    print("Check if prerequisite 'PyOpenGL' library is installed.")
    print("Also, the videocard might not support stencil buffers.\n")

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

        # -----------------------------------
        # -----------------------------------
        #   Frame top
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
        hbox_top = QtWid.QHBoxLayout()
        hbox_top.addLayout(vbox_left)
        hbox_top.addStretch(1)
        hbox_top.addLayout(vbox_middle)
        hbox_top.addStretch(1)
        hbox_top.addLayout(vbox_right)

        # -----------------------------------
        # -----------------------------------
        #   Frame 'Reference and signal'
        # -----------------------------------
        # -----------------------------------
        
        # Chart 'Readings'
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

        PEN_01 = pg.mkPen(color=[255, 0  , 0  ], width=3)
        PEN_02 = pg.mkPen(color=[255, 125, 0  ], width=3)
        PEN_03 = pg.mkPen(color=[0  , 255, 255], width=3)
        PEN_04 = pg.mkPen(color=[255, 255, 255], width=3)
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

        # 'On/off'
        self.qpbt_ENA_lockin = create_Toggle_button("lock-in OFF")
        self.qpbt_ENA_lockin.clicked.connect(self.process_qpbt_ENA_lockin)

        # 'Reference signal'
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
        self.qlin_set_ref_V_center = (
                QtWid.QLineEdit("%.2f" % lockin.config.ref_V_center, **p1))
        self.qlin_read_ref_V_center = (
                QtWid.QLineEdit("%.2f" % lockin.config.ref_V_center, **p2))
        self.qlin_set_ref_V_p2p = (
                QtWid.QLineEdit("%.2f" % lockin.config.ref_V_p2p, **p1))
        self.qlin_read_ref_V_p2p = (
                QtWid.QLineEdit("%.2f" % lockin.config.ref_V_p2p, **p2))

        self.qlin_set_ref_freq.editingFinished.connect(
                self.process_qlin_set_ref_freq)
        self.qlin_set_ref_V_center.editingFinished.connect(
                self.process_qlin_set_ref_V_center)
        self.qlin_set_ref_V_p2p.editingFinished.connect(
                self.process_qlin_set_ref_V_p2p)
        
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
        grid.addWidget(QtWid.QLabel("V_center:", **p), i, 0)
        grid.addWidget(self.qlin_set_ref_V_center    , i, 1)
        grid.addWidget(self.qlin_read_ref_V_center   , i, 2)
        grid.addWidget(QtWid.QLabel("V")             , i, 3); i+=1
        grid.addWidget(QtWid.QLabel("V_p2p:", **p)   , i, 0)
        grid.addWidget(self.qlin_set_ref_V_p2p       , i, 1)
        grid.addWidget(self.qlin_read_ref_V_p2p      , i, 2)
        grid.addWidget(QtWid.QLabel("V")             , i, 3)
        
        qgrp_ref_freq = QtWid.QGroupBox("Reference signal")
        qgrp_ref_freq.setStyleSheet(SS_GROUP)
        qgrp_ref_freq.setLayout(grid)

        # 'Reference and signal' readings
        p = {'layoutDirection': QtCore.Qt.LeftToRight}
        self.chkbs_refsig = [
                QtWid.QCheckBox("(red) ref_X [0]:", **p, checked=True),
                QtWid.QCheckBox("(ora) ref_Y [0]:", **p, checked=False),
                QtWid.QCheckBox("(cya) sig_I [0]:", **p, checked=True)]
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

        qgrp_refsig = QtWid.QGroupBox("Readings")
        qgrp_refsig.setStyleSheet(SS_GROUP)
        qgrp_refsig.setLayout(grid)
        
        # Round up frame
        vbox_refsig = QtWid.QVBoxLayout()
        vbox_refsig.addWidget(self.qpbt_ENA_lockin)
        vbox_refsig.addWidget(qgrp_ref_freq)
        vbox_refsig.addWidget(qgrp_refsig)
        vbox_refsig.addStretch()

        hbox_refsig = QtWid.QHBoxLayout()
        hbox_refsig.addWidget(self.gw_refsig, stretch=1)
        hbox_refsig.addLayout(vbox_refsig)
        
        # -----------------------------------
        # -----------------------------------
        #   Frame 'Mixer and filters'
        # -----------------------------------
        # -----------------------------------
        
        # Chart 'Mixer and filters'
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
                                     self.pi_mixer.plot(pen=PEN_03))
        self.CH_mix_Y = ChartHistory(lockin.config.BUFFER_SIZE,
                                     self.pi_mixer.plot(pen=PEN_04))
        self.CH_mix_X.x_axis_divisor = 1000     # From [us] to [ms]
        self.CH_mix_Y.x_axis_divisor = 1000     # From [us] to [ms]
        self.CHs_mixer = [self.CH_mix_X, self.CH_mix_Y]
        
        # Round up frame
        """
        vbox_mixer = QtWid.QVBoxLayout()
        vbox_mixer.addWidget(self.qpbt_ENA_lockin)
        vbox_mixer.addWidget(qgrp_ref_freq)
        vbox_mixer.addWidget(qgrp_refsig)
        vbox_mixer.addStretch()
        """

        hbox_mixer = QtWid.QHBoxLayout()
        hbox_mixer.addWidget(self.gw_mixer, stretch=1)
        #hbox_mixer.addLayout(vbox_mixer)
        
        """
        # Chart 'Mixer and filters'
        self.qpbt_clear_chart = QtWid.QPushButton("Clear")
        self.qpbt_clear_chart.clicked.connect(self.process_qpbt_clear_chart)

        grid = QtWid.QGridLayout()
        grid.addWidget(self.qpbt_clear_chart, 0, 0)
        grid.setAlignment(QtCore.Qt.AlignTop)

        qgrp_chart = QtWid.QGroupBox("Chart")
        qgrp_chart.setStyleSheet(SS_GROUP)
        qgrp_chart.setLayout(grid)
        """

        # -----------------------------------
        # -----------------------------------
        #   Round up full window
        # -----------------------------------
        # -----------------------------------
        
        vbox = QtWid.QVBoxLayout(self)
        vbox.addLayout(hbox_top)
        vbox.addSpacerItem(QtWid.QSpacerItem(0, 20))
        vbox.addLayout(hbox_refsig, stretch=1)
        vbox.addLayout(hbox_mixer, stretch=1)
        
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
        self.lockin_pyqt.signal_ref_V_center_is_set.connect(
                self.update_qlin_read_ref_V_center)
        self.lockin_pyqt.signal_ref_V_p2p_is_set.connect(
                self.update_qlin_read_ref_V_p2p)
    
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
    def process_qlin_set_ref_freq(self):
        try:
            ref_freq = float(self.qlin_set_ref_freq.text())
        except ValueError:
            ref_freq = self.lockin.config.ref_freq
        
        # Clip between 0 and the Nyquist frequency of the lock-in sampling rate
        ref_freq = np.clip(ref_freq, 0, 1/self.lockin.config.ISR_CLOCK/2)
        
        self.qlin_set_ref_freq.setText("%.2f" % ref_freq)
        if ref_freq != self.lockin.config.ref_freq:
            self.lockin_pyqt.set_ref_freq(ref_freq)
            QtWid.QApplication.processEvents()
            
    @QtCore.pyqtSlot()
    def process_qlin_set_ref_V_center(self):
        try:
            ref_V_center = float(self.qlin_set_ref_V_center.text())
        except ValueError:
            ref_V_center = self.lockin.config.ref_V_center
        
        # Clip between 0 and the analog voltage reference
        ref_V_center = np.clip(ref_V_center, 0, self.lockin.config.A_REF)
        
        self.qlin_set_ref_V_center.setText("%.2f" % ref_V_center)
        if ref_V_center != self.lockin.config.ref_V_center:
            self.lockin_pyqt.set_ref_V_center(ref_V_center)            
            QtWid.QApplication.processEvents()
            
    @QtCore.pyqtSlot()
    def process_qlin_set_ref_V_p2p(self):
        try:
            ref_V_p2p = float(self.qlin_set_ref_V_p2p.text())
        except ValueError:
            ref_V_p2p = self.lockin.config.ref_V_p2p
        
        # Clip between 0 and the analog voltage reference
        ref_V_p2p = np.clip(ref_V_p2p, 0, self.lockin.config.A_REF)
        
        self.qlin_set_ref_V_p2p.setText("%.2f" % ref_V_p2p)
        if ref_V_p2p != self.lockin.config.ref_V_p2p:
            self.lockin_pyqt.set_ref_V_p2p(ref_V_p2p)
            QtWid.QApplication.processEvents()
        
    @QtCore.pyqtSlot()
    def update_qlin_read_ref_freq(self):
        self.qlin_read_ref_freq.setText("%.2f" % self.lockin.config.ref_freq)
        
    @QtCore.pyqtSlot()
    def update_qlin_read_ref_V_center(self):
        self.qlin_read_ref_V_center.setText("%.2f" %
                                            self.lockin.config.ref_V_center)
        
    @QtCore.pyqtSlot()
    def update_qlin_read_ref_V_p2p(self):
        self.qlin_read_ref_V_p2p.setText("%.2f" % self.lockin.config.ref_V_p2p)

    @QtCore.pyqtSlot()
    def process_chkbs_refsig(self):
        if self.lockin.lockin_paused:
            self.update_chart_refsig()  # Force update graph

    @QtCore.pyqtSlot()
    def process_qpbt_full_axes(self):
        min_time = -(self.lockin.config.BUFFER_SIZE * 
                     self.lockin.config.ISR_CLOCK * 1e3)
        self.pi_refsig.setXRange(min_time, 0, padding=0.01)
        self.pi_mixer.setXRange(min_time, 0, padding=0.01)
        self.process_qpbtn_autoscale_y()

    @QtCore.pyqtSlot()
    def process_qpbtn_autoscale_y(self):
        self.pi_refsig.enableAutoRange('y', True)
        self.pi_refsig.enableAutoRange('y', False)
        self.pi_mixer.enableAutoRange('y', True)
        self.pi_mixer.enableAutoRange('y', False)
        
    @QtCore.pyqtSlot()
    def process_qpbt_clear_chart(self):
        str_msg = "Are you sure you want to clear the chart?"
        reply = QtWid.QMessageBox.warning(self, "Clear chart", str_msg,
                                          QtWid.QMessageBox.Yes |
                                          QtWid.QMessageBox.No,
                                          QtWid.QMessageBox.No)

        if reply == QtWid.QMessageBox.Yes:
            """Placeholder
            """
            pass
        
        
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
    def update_chart_mixer(self):
        [CH.update_curve() for CH in self.CHs_mixer]