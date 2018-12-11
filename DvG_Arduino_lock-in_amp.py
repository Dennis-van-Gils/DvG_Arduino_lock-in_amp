#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Arduino lock-in amplifier
"""
__author__      = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__         = "https://github.com/Dennis-van-Gils/DvG_Arduino_lock-in_amp"
__date__        = "09-12-2018"
__version__     = "1.0.0"

import os
import sys
import struct
from pathlib import Path

import psutil

from PyQt5 import QtCore, QtGui
from PyQt5 import QtWidgets as QtWid
from PyQt5.QtCore import QDateTime
import pyqtgraph as pg
import numpy as np

from collections import deque
import time as Time

from DvG_pyqt_FileLogger   import FileLogger
from DvG_pyqt_ChartHistory import ChartHistory
from DvG_pyqt_controls     import (create_Toggle_button,
                                   SS_GROUP,
                                   SS_TEXTBOX_READ_ONLY)
from DvG_debug_functions   import dprint, print_fancy_traceback as pft

import DvG_dev_Arduino_lockin_amp__fun_serial as lockin_functions
import DvG_dev_Arduino_lockin_amp__pyqt_lib   as lockin_pyqt_lib

# Constants
UPDATE_INTERVAL_GUI_WALL_CLOCK = 50 # 100 [ms]
CHART_HISTORY_TIME = 10  # 10  [s]

# Show debug info in terminal? Warning: Slow! Do not leave on unintentionally.
DEBUG = False

# ------------------------------------------------------------------------------
#   Arduino state
# ------------------------------------------------------------------------------

class State(object):
    """Reflects the actual readings, parsed into separate variables, of the
    Arduino(s). There should only be one instance of the State class.
    """
    def __init__(self):
        self.buffers_received = 0
        
        self.time  = np.array([], int)      # [ms]
        self.ref_X = np.array([], float)
        self.ref_Y = np.array([], float)
        self.sig_I = np.array([], float)

        """
        These arrays are N times the buffer size of the Arduino in order to
        facilitate preventing start/end effects due to FIR filtering.
        A smaller shifting window can walk over these arrays and a FIR filter
        using convolution can be applied, where only the valid samples are kept
        (no zero padding).
        
        Each time a complete buffer of BLOCK_SIZE samples is received from the
        Arduino, it is appended to the end of these arrays (FIFO shift buffer).
        
            i.e. N = 3    
                hist_time = [buffer_1; received_buffer_2; buffer_3]
                hist_time = [buffer_2; received_buffer_3; buffer_4]
                hist_time = [buffer_3; received_buffer_4; buffer_5]
                etc...
        """
        self.hist_time  =  deque()
        self.hist_ref_X = deque()
        self.hist_ref_Y = deque()
        self.hist_sig_I = deque()
        self.hist_mix_X = deque()
        self.hist_mix_Y = deque()
        self.hist_out_amp = deque()
        self.hist_out_phi = deque()

        # Mutex for proper multithreading. If the state variables are not
        # atomic or thread-safe, you should lock and unlock this mutex for each
        # read and write operation. In this demo we don't need it, but I keep it
        # as reminder.
        self.mutex = QtCore.QMutex()
        
    def init_shift_buffers(self, N):
        self.hist_time  = deque(maxlen=N * lockin.BUFFER_SIZE)
        self.hist_ref_X = deque(maxlen=N * lockin.BUFFER_SIZE)
        self.hist_ref_Y = deque(maxlen=N * lockin.BUFFER_SIZE)
        self.hist_sig_I = deque(maxlen=N * lockin.BUFFER_SIZE)
        self.hist_mix_X = deque(maxlen=N * lockin.BUFFER_SIZE)
        self.hist_mix_Y = deque(maxlen=N * lockin.BUFFER_SIZE)
        self.hist_out_amp = deque(maxlen=N * lockin.BUFFER_SIZE)
        self.hist_out_phi = deque(maxlen=N * lockin.BUFFER_SIZE)

# ------------------------------------------------------------------------------
#   MainWindow
# ------------------------------------------------------------------------------

class MainWindow(QtWid.QWidget):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

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
                                             (1/lockin.ISR_CLOCK))
        self.qlbl_buffer_size = QtWid.QLabel("BUFFER SIZE  : %i" %
                                             lockin.BUFFER_SIZE)
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
        vbox_right.addWidget(QtWid.QLabel("Dennis van Gils", **p))
        vbox_right.addWidget(QtWid.QLabel("08-12-2018", **p))

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
        
        # Chart 'Reference and signal'
        self.gw_refsig = pg.GraphicsWindow()
        self.gw_refsig.setBackground([20, 20, 20])
        self.pi_refsig = self.gw_refsig.addPlot()

        p = {'color': '#BBB', 'font-size': '10pt'}
        self.pi_refsig.showGrid(x=1, y=1)
        self.pi_refsig.setTitle('Reference and signal', **p)
        self.pi_refsig.setLabel('bottom', text='time (ms)', **p)
        self.pi_refsig.setLabel('left', text='voltage (V)', **p)
        self.pi_refsig.setXRange(-lockin.BUFFER_SIZE * lockin.ISR_CLOCK * 1e3,
                                 0, padding=0)
        self.pi_refsig.setYRange(1, 3, padding=0.05)
        self.pi_refsig.setAutoVisible(x=True, y=True)
        self.pi_refsig.setClipToView(True)

        PEN_01 = pg.mkPen(color=[255, 0  , 0  ], width=3)
        PEN_02 = pg.mkPen(color=[255, 125, 0  ], width=3)
        PEN_03 = pg.mkPen(color=[0  , 255, 255], width=3)
        PEN_04 = pg.mkPen(color=[255, 255, 255], width=3)
        self.CH_ref_X = ChartHistory(lockin.BUFFER_SIZE,
                                     self.pi_refsig.plot(pen=PEN_01))
        self.CH_ref_Y = ChartHistory(lockin.BUFFER_SIZE,
                                     self.pi_refsig.plot(pen=PEN_02))
        self.CH_sig_I = ChartHistory(lockin.BUFFER_SIZE,
                                     self.pi_refsig.plot(pen=PEN_03))
        self.CH_ref_X.x_axis_divisor = 1000     # From [us] to [ms]
        self.CH_ref_Y.x_axis_divisor = 1000     # From [us] to [ms]
        self.CH_sig_I.x_axis_divisor = 1000     # From [us] to [ms]
        self.CHs_refsig = [self.CH_ref_X, self.CH_ref_Y, self.CH_sig_I]

        # 'On/off'
        self.qpbt_ENA_lockin = create_Toggle_button("lock-in OFF")
        self.qpbt_ENA_lockin.clicked.connect(self.process_qpbt_ENA_lockin)

        # 'Reference frequency'
        self.qlin_set_ref_freq  = QtWid.QLineEdit("%.2f" % lockin.ref_freq)
        self.qlin_read_ref_freq = QtWid.QLineEdit("%.2f" % lockin.ref_freq, 
                                                  readOnly=True)
        self.qlin_set_ref_freq.editingFinished.connect(
                self.process_qlin_set_ref_freq)
        
        p = {'alignment': QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight}
        grid = QtWid.QGridLayout()
        grid.addWidget(QtWid.QLabel("Set f_ref:", **p) , 0, 1)
        grid.addWidget(self.qlin_set_ref_freq          , 0, 2)
        grid.addWidget(QtWid.QLabel("Hz")              , 0, 3)
        grid.addWidget(QtWid.QLabel("Read f_ref:", **p), 1, 1)
        grid.addWidget(self.qlin_read_ref_freq         , 1, 2)
        grid.addWidget(QtWid.QLabel("Hz")              , 1, 3)
        
        qgrp_ref_freq = QtWid.QGroupBox("Reference frequency")
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

        grid = QtWid.QGridLayout()
        grid.addWidget(QtWid.QLabel("time [0]:"), 0, 0)
        grid.addWidget(self.qlin_time           , 0, 1)
        grid.addWidget(QtWid.QLabel("us")       , 0, 2)
        grid.addWidget(self.chkbs_refsig[0]     , 1, 0)
        grid.addWidget(self.qlin_ref_X          , 1, 1)
        grid.addWidget(QtWid.QLabel("V")        , 1, 2)
        grid.addWidget(self.chkbs_refsig[1]     , 2, 0)
        grid.addWidget(self.qlin_ref_Y          , 2, 1)
        grid.addWidget(QtWid.QLabel("V")        , 2, 2)
        grid.addWidget(self.chkbs_refsig[2]     , 3, 0)
        grid.addWidget(self.qlin_sig_I          , 3, 1)
        grid.addWidget(QtWid.QLabel("V")        , 3, 2)
        grid.addWidget(self.qpbt_full_axes      , 4, 0, 1, 2)
        grid.addWidget(self.qpbt_autoscale_y    , 5, 0, 1, 2)
        grid.setAlignment(QtCore.Qt.AlignTop)

        qgrp_refsig = QtWid.QGroupBox("Reference and signal")
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
        self.pi_mixer.setXRange(-lockin.BUFFER_SIZE * lockin.ISR_CLOCK * 1e3,
                                 0, padding=0)
        self.pi_mixer.setYRange(-1, 1, padding=0.05)
        self.pi_mixer.setAutoVisible(x=True, y=True)
        self.pi_mixer.setClipToView(True)
        
        self.CH_mix_X = ChartHistory(lockin.BUFFER_SIZE,
                                     self.pi_mixer.plot(pen=PEN_03))
        self.CH_mix_Y = ChartHistory(lockin.BUFFER_SIZE,
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

    # --------------------------------------------------------------------------
    #   Handle controls
    # --------------------------------------------------------------------------

    @QtCore.pyqtSlot()
    def process_qpbt_ENA_lockin(self):
        if self.qpbt_ENA_lockin.isChecked():
            if lockin.turn_on():
                self.qpbt_ENA_lockin.setText("lock-in ON")
                lockin.lockin_paused = False
                #lockin.ser.flush()
                lockin_pyqt.worker_DAQ.unpause()
                #app.processEvents()
        else:
            window.qlbl_DAQ_rate.setText("Buffers/s: paused")
            self.qpbt_ENA_lockin.setText("lock-in OFF")
            lockin.lockin_paused = True
            lockin_pyqt.worker_DAQ.pause()
            lockin.turn_off()
            #lockin.ser.flush()
            #app.processEvents()

    @QtCore.pyqtSlot()
    def process_qpbt_clear_chart(self):
        str_msg = "Are you sure you want to clear the chart?"
        reply = QtWid.QMessageBox.warning(window, "Clear chart", str_msg,
                                          QtWid.QMessageBox.Yes |
                                          QtWid.QMessageBox.No,
                                          QtWid.QMessageBox.No)

        if reply == QtWid.QMessageBox.Yes:
            """Placeholder
            """
            pass
        
    @QtCore.pyqtSlot()
    def process_qpbt_record(self):
        if self.qpbt_record.isChecked():
            file_logger.starting = True
            file_logger.stopping = False
        else:
            file_logger.starting = False
            file_logger.stopping = True

    @QtCore.pyqtSlot(str)
    def set_text_qpbt_record(self, text_str):
        self.qpbt_record.setText(text_str)
            
    @QtCore.pyqtSlot()
    def process_qlin_set_ref_freq(self):
        try:
            ref_freq = float(self.qlin_set_ref_freq.text())
        except ValueError:
            ref_freq = lockin.ref_freq
        
        # Clip between 0 and the Nyquist frequency of the lock-in sampling rate
        ref_freq = np.clip(ref_freq, 0, 1/lockin.ISR_CLOCK/2)        
        
        self.qlin_set_ref_freq.setText("%.2f" % ref_freq)
        lockin_pyqt.worker_send.queued_instruction("set_ref_freq", ref_freq)
        app.processEvents()
        
    @QtCore.pyqtSlot()
    def update_qlin_read_ref_freq(self):
        self.qlin_read_ref_freq.setText("%.2f" % lockin.ref_freq)    

    @QtCore.pyqtSlot()
    def process_chkbs_refsig(self):
        if lockin.lockin_paused:
            update_chart_refsig()  # Force update graph

    @QtCore.pyqtSlot()
    def process_qpbt_full_axes(self):
        self.pi_refsig.setXRange(-lockin.BUFFER_SIZE * lockin.ISR_CLOCK * 1e3, 0,
                                 padding=0)
        self.process_qpbtn_autoscale_y()

    @QtCore.pyqtSlot()
    def process_qpbtn_autoscale_y(self):
        self.pi_refsig.enableAutoRange('y', True)
        self.pi_refsig.enableAutoRange('y', False)

# ------------------------------------------------------------------------------
#   Update GUI routines
# ------------------------------------------------------------------------------

def current_date_time_strings():
    cur_date_time = QDateTime.currentDateTime()
    return (cur_date_time.toString("dd-MM-yyyy"),
            cur_date_time.toString("HH:mm:ss"))

@QtCore.pyqtSlot()
def update_GUI_wall_clock():
    str_cur_date, str_cur_time = current_date_time_strings()
    window.qlbl_cur_date_time.setText("%s    %s" % (str_cur_date, str_cur_time))

@QtCore.pyqtSlot()
def update_GUI():
    window.qlbl_update_counter.setText("%i" % lockin_pyqt.DAQ_update_counter)
    
    if not lockin.lockin_paused:
        window.qlbl_DAQ_rate.setText("Buffers/s: %.1f" % 
                                     lockin_pyqt.obtained_DAQ_rate_Hz)
        window.qlin_time.setText("%i" % state.time[0])
        window.qlin_ref_X.setText("%.4f" % state.ref_X[0])
        window.qlin_ref_Y.setText("%.4f" % state.ref_Y[0])
        window.qlin_sig_I.setText("%.4f" % state.sig_I[0])
        
        update_chart_refsig()
        
        [CH.update_curve() for CH in window.CHs_mixer]

@QtCore.pyqtSlot()
def update_chart_refsig():
    [CH.update_curve() for CH in window.CHs_refsig]
    for i in range(3):
        window.CHs_refsig[i].curve.setVisible(
                window.chkbs_refsig[i].isChecked())
        
# ------------------------------------------------------------------------------
#   Program termination routines
# ------------------------------------------------------------------------------

def stop_running():
    app.processEvents()
    lockin.lockin_paused = True
    lockin_pyqt.worker_DAQ.pause()
    lockin.turn_off()
    lockin_pyqt.close_all_threads()
    file_logger.close_log()

    print("Stopping timers: ", end='')
    timer_GUI_wall_clock.stop()
    print("done.")

@QtCore.pyqtSlot()
def notify_connection_lost():
    stop_running()

    excl = "    ! ! ! ! ! ! ! !    "
    window.qlbl_title.setText("%sLOST CONNECTION%s" % (excl, excl))

    str_cur_date, str_cur_time = current_date_time_strings()
    str_msg = (("%s %s\n"
                "Lost connection to Arduino(s).\n"
                "  '%s', '%s': %salive") %
               (str_cur_date, str_cur_time,
                lockin.name, lockin.identity, '' if lockin.is_alive else "not "))
    print("\nCRITICAL ERROR @ %s" % str_msg)
    reply = QtWid.QMessageBox.warning(window, "CRITICAL ERROR", str_msg,
                                      QtWid.QMessageBox.Ok)

    if reply == QtWid.QMessageBox.Ok:
        pass    # Leave the GUI open for read-only inspection by the user

@QtCore.pyqtSlot()
def about_to_quit():
    print("\nAbout to quit")
    stop_running()
    lockin.close()

# ------------------------------------------------------------------------------
#   Lock-in amplifier data-acquisition update function
# ------------------------------------------------------------------------------

def lockin_DAQ_update():
    str_cur_date, str_cur_time = current_date_time_strings()
    
    [success, ans_bytes] = lockin.listen_to_lockin_amp()
    if lockin.lockin_paused:     # Prevent throwings errors if just paused
        return False
    
    if not(success):
        dprint("'%s' ERROR I/O       @ %s %s" %
               (lockin.name, str_cur_date, str_cur_time))
        return False
    
    state.buffers_received += 1
    N_samples = int(len(ans_bytes) / struct.calcsize('LHH'))
    if not(N_samples == lockin.BUFFER_SIZE):
        dprint("'%s' ERROR N_samples @ %s %s" %
               (lockin.name, str_cur_date, str_cur_time))
        return False
    
    e_byte_time  = N_samples * struct.calcsize('L');
    e_byte_ref_X = e_byte_time + N_samples * struct.calcsize('H')
    e_byte_sig_I = e_byte_ref_X + N_samples * struct.calcsize('H')
    bytes_time  = ans_bytes[0            : e_byte_time]
    bytes_ref_X = ans_bytes[e_byte_time  : e_byte_ref_X]
    bytes_sig_I = ans_bytes[e_byte_ref_X : e_byte_sig_I]
    try:
        time        = np.array(struct.unpack('<' + 'L'*N_samples, bytes_time))
        phase_ref_X = np.array(struct.unpack('<' + 'H'*N_samples, bytes_ref_X))
        sig_I       = np.array(struct.unpack('<' + 'H'*N_samples, bytes_sig_I))
    except Exception:
        return False
    
    ref_X = np.cos(2*np.pi*phase_ref_X/12288)
    ref_X = 2 + ref_X                   # [V]
    ref_Y = np.sin(2*np.pi*phase_ref_X/12288)
    ref_Y = 2 + ref_Y                   # [V]
    sig_I = sig_I / (2**12 - 1)*3.3     # [V]
    
    mix_X = (ref_X - 2) * (sig_I - 2)
    mix_Y = (ref_Y - 2) * (sig_I - 2)
    
    out_amp = 2 * np.sqrt(mix_X**2 + mix_Y**2)
    # NOTE: Because 'mix_X' and 'mix_Y' are both of type 'numpy.array', a division
    # by (mix_X = 0) is handled correctly due to 'numpy.inf'.
    # Likewise, 'numpy.arctan(numpy.inf)' will result in pi/2.
    # We suppress the RuntimeWarning: divide by zero encountered in true_divide.
    np.seterr(divide='ignore')
    out_phi = np.arctan(mix_Y / mix_X)
    np.seterr(divide='warn')
    
    state.time  = time
    state.ref_X = ref_X
    state.ref_Y = ref_Y
    state.sig_I = sig_I
    
    state.hist_time.extend(time)
    state.hist_ref_X.extend(ref_X)
    state.hist_ref_Y.extend(ref_Y)
    state.hist_sig_I.extend(sig_I)
    state.hist_mix_X.extend(mix_X)
    state.hist_mix_Y.extend(mix_Y)
    state.hist_out_amp.extend(out_amp)
    state.hist_out_phi.extend(out_phi)
    
    window.CH_ref_X.add_new_readings(time, ref_X)
    window.CH_ref_Y.add_new_readings(time, ref_Y)
    window.CH_sig_I.add_new_readings(time, sig_I)
    
    window.CH_mix_X.add_new_readings(time, mix_X)
    window.CH_mix_Y.add_new_readings(time, mix_Y)
    
    # Logging to file
    if file_logger.starting:
        fn_log = QDateTime.currentDateTime().toString("yyMMdd_HHmmss") + ".txt"
        if file_logger.create_log(state.time, fn_log, mode='w'):
            file_logger.signal_set_recording_text.emit(
                "Recording to file: " + fn_log)
            file_logger.write("time[us]\tref_X[V]\tref_Y[V]\tsig_I[V]\n")

    if file_logger.stopping:
        file_logger.signal_set_recording_text.emit(
            "Click to start recording to file")
        file_logger.close_log()

    if file_logger.is_recording:
        for i in range(N_samples):
            file_logger.write("%i\t%.4f\t%.4f\t%.4f\n" % 
                              (time[i], ref_X[i], ref_Y[i], sig_I[i]))

    return True

# ------------------------------------------------------------------------------
#   Main
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    # Set priority of this process to maximum in the operating system
    print("PID: %s\n" % os.getpid())
    try:
        proc = psutil.Process(os.getpid())
        if os.name == "nt": proc.nice(psutil.REALTIME_PRIORITY_CLASS) # Windows
        else: proc.nice(-20)                                          # Other
    except:
        print("Warning: Could not set process to maximum priority.\n")

    # --------------------------------------------------------------------------
    #   Connect to Arduino
    # --------------------------------------------------------------------------

    lockin = lockin_functions.Arduino_lockin_amp(baudrate=1.5e6, read_timeout=.5)
    if not lockin.auto_connect(Path("port_data.txt"), "Arduino lock-in amp"):
        sys.exit(0)
        
    lockin.begin()
    lockin.set_ref_freq(100)
    
    state = State()
    state.init_shift_buffers(N=3)

    """if not(lockin.is_alive):
        print("\nCheck connection and try resetting the Arduino.")
        print("Exiting...\n")
        sys.exit(0)
    """

    # --------------------------------------------------------------------------
    #   Create application and main window
    # --------------------------------------------------------------------------
    QtCore.QThread.currentThread().setObjectName('MAIN')    # For DEBUG info

    app = 0    # Work-around for kernel crash when using Spyder IDE
    app = QtWid.QApplication(sys.argv)
    app.aboutToQuit.connect(about_to_quit)

    window = MainWindow()

    # --------------------------------------------------------------------------
    #   File logger
    # --------------------------------------------------------------------------

    file_logger = FileLogger()
    file_logger.signal_set_recording_text.connect(window.set_text_qpbt_record)

    # --------------------------------------------------------------------------
    #   Set up communication threads for the Arduino
    # --------------------------------------------------------------------------

    # Create workers and threads
    lockin_pyqt = lockin_pyqt_lib.Arduino_lockin_amp_pyqt(
            dev=lockin,
            DAQ_function_to_run_each_update=lockin_DAQ_update,
            DAQ_critical_not_alive_count=np.nan,
            calc_DAQ_rate_every_N_iter=10)

    # Connect signals to slots
    lockin_pyqt.signal_DAQ_updated.connect(update_GUI)
    lockin_pyqt.signal_connection_lost.connect(notify_connection_lost)
    lockin_pyqt.signal_ref_freq_is_set.connect(window.update_qlin_read_ref_freq)

    # Start threads
    lockin_pyqt.start_thread_worker_DAQ(QtCore.QThread.TimeCriticalPriority)
    lockin_pyqt.start_thread_worker_send()

    # --------------------------------------------------------------------------
    #   Create timers
    # --------------------------------------------------------------------------

    timer_GUI_wall_clock = QtCore.QTimer()
    timer_GUI_wall_clock.timeout.connect(update_GUI_wall_clock)
    timer_GUI_wall_clock.start(UPDATE_INTERVAL_GUI_WALL_CLOCK)

    # --------------------------------------------------------------------------
    #   Start the main GUI event loop
    # --------------------------------------------------------------------------

    window.show()
    sys.exit(app.exec_())