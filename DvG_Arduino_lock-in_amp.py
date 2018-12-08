#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Arduino lock-in amplifier
"""
__author__      = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__         = "https://github.com/Dennis-van-Gils/DvG_Arduino_lock-in_amp"
__date__        = "08-12-2018"
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

from DvG_pyqt_FileLogger   import FileLogger
from DvG_pyqt_ChartHistory import ChartHistory
from DvG_pyqt_controls     import create_Toggle_button, SS_GROUP
from DvG_debug_functions   import dprint, print_fancy_traceback as pft

import DvG_dev_Arduino_lockin_amp__fun_serial as lockin_functions
import DvG_dev_Arduino__pyqt_lib as Arduino_pyqt_lib

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
        self.lockin_paused = True
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
        self.hist_time  = deque(maxlen=N * ard.BUFFER_SIZE)
        self.hist_ref_X = deque(maxlen=N * ard.BUFFER_SIZE)
        self.hist_ref_Y = deque(maxlen=N * ard.BUFFER_SIZE)
        self.hist_sig_I = deque(maxlen=N * ard.BUFFER_SIZE)
        self.hist_mix_X = deque(maxlen=N * ard.BUFFER_SIZE)
        self.hist_mix_Y = deque(maxlen=N * ard.BUFFER_SIZE)
        self.hist_out_amp = deque(maxlen=N * ard.BUFFER_SIZE)
        self.hist_out_phi = deque(maxlen=N * ard.BUFFER_SIZE)

state = State()

# ------------------------------------------------------------------------------
#   current_date_time_strings()
# ------------------------------------------------------------------------------

def current_date_time_strings():
    cur_date_time = QDateTime.currentDateTime()
    return (cur_date_time.toString("dd-MM-yyyy"),
            cur_date_time.toString("HH:mm:ss"))
    
# ------------------------------------------------------------------------------
#   MainWindow
# ------------------------------------------------------------------------------

class MainWindow(QtWid.QWidget):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        self.setGeometry(50, 50, 800, 660)
        self.setWindowTitle("Arduino lock-in amplifier")

        # -------------------------
        #   Top frame
        # -------------------------

        # Left box
        self.qlbl_update_counter = QtWid.QLabel("0")
        self.qlbl_sample_rate = QtWid.QLabel("SAMPLE RATE: nan")
        self.qlbl_buffer_size = QtWid.QLabel("BUFFER SIZE: nan")
        self.qlbl_DAQ_rate = QtWid.QLabel("Buffers/s: nan")
        self.qlbl_DAQ_rate.setMinimumWidth(100)

        vbox_left = QtWid.QVBoxLayout()
        #vbox_left.addWidget(self.qlbl_update_counter, stretch=0)
        #vbox_left.addStretch(1)
        vbox_left.addWidget(self.qlbl_sample_rate, stretch=0)
        vbox_left.addWidget(self.qlbl_buffer_size, stretch=0)
        vbox_left.addStretch(1)
        vbox_left.addWidget(self.qlbl_DAQ_rate, stretch=0)
        vbox_left.addWidget(self.qlbl_update_counter, stretch=0)

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
        vbox_right.addWidget(self.qpbt_exit, stretch=0)
        vbox_right.addStretch(1)
        vbox_right.addWidget(QtWid.QLabel("Dennis van Gils", **p), stretch=0)
        vbox_right.addWidget(QtWid.QLabel("08-12-2018", **p), stretch=0)
        vbox_right.addStretch(1)

        # Round up top frame
        hbox_top = QtWid.QHBoxLayout()
        hbox_top.addLayout(vbox_left, stretch=0)
        hbox_top.addStretch(1)
        hbox_top.addLayout(vbox_middle, stretch=0)
        hbox_top.addStretch(1)
        hbox_top.addLayout(vbox_right, stretch=0)

        # -------------------------
        #   Bottom frame
        # -------------------------

        # Create PlotItem
        self.gw_chart = pg.GraphicsWindow()
        self.gw_chart.setBackground([20, 20, 20])
        self.pi_chart = self.gw_chart.addPlot()

        p = {'color': '#BBB', 'font-size': '10pt'}
        self.pi_chart.showGrid(x=1, y=1)
        self.pi_chart.setTitle('Arduino timeseries', **p)
        self.pi_chart.setLabel('bottom', text='time (ms)', **p)
        self.pi_chart.setLabel('left', text='voltage (V)', **p)
        #self.pi_chart.setRange(xRange=[-40, 0],
        #                       yRange=[.9, 3.1],
        #                       disableAutoRange=True)
        self.pi_chart.setXRange(-ard.BUFFER_SIZE * ard.ISR_CLOCK * 1e3, 0,
                                padding=0)
        self.pi_chart.setYRange(1, 3, padding=0.05)

        # Create ChartHistory and PlotDataItem and link them together
        PEN_01 = pg.mkPen(color=[255, 0  , 0], width=3)
        PEN_02 = pg.mkPen(color=[255, 125, 0], width=3)
        PEN_03 = pg.mkPen(color=[0  , 255, 255], width=3)
        self.CH_ref_X = ChartHistory(ard.BUFFER_SIZE,
                                     self.pi_chart.plot(pen=PEN_01))
        self.CH_ref_Y = ChartHistory(ard.BUFFER_SIZE,
                                     self.pi_chart.plot(pen=PEN_02))
        self.CH_sig_I = ChartHistory(ard.BUFFER_SIZE,
                                     self.pi_chart.plot(pen=PEN_03))
        self.CH_ref_X.x_axis_divisor = 1000     # From [us] to [ms]
        self.CH_ref_Y.x_axis_divisor = 1000     # From [us] to [ms]
        self.CH_sig_I.x_axis_divisor = 1000     # From [us] to [ms]

        # 'On/off'
        self.qpbt_ENA_lockin = create_Toggle_button("lock-in OFF")
        self.qpbt_ENA_lockin.clicked.connect(self.process_qpbt_ENA_lockin)

        # 'Readings'
        p = {'readOnly': True}
        self.qlin_time  = QtWid.QLineEdit(**p)
        self.qlin_ref_X = QtWid.QLineEdit(**p)
        self.qlin_ref_Y = QtWid.QLineEdit(**p)
        self.qlin_sig_I = QtWid.QLineEdit(**p)

        grid = QtWid.QGridLayout()
        grid.addWidget(QtWid.QLabel("time [0]:") , 0, 0)
        grid.addWidget(self.qlin_time            , 0, 1)
        grid.addWidget(QtWid.QLabel("us")        , 0, 2)
        grid.addWidget(QtWid.QLabel("ref_X [0]:"), 1, 0)
        grid.addWidget(self.qlin_ref_X           , 1, 1)
        grid.addWidget(QtWid.QLabel("V")         , 1, 2)
        grid.addWidget(QtWid.QLabel("ref_Y [0]:"), 2, 0)
        grid.addWidget(self.qlin_ref_Y           , 2, 1)
        grid.addWidget(QtWid.QLabel("V")         , 2, 2)
        grid.addWidget(QtWid.QLabel("sig_I [0]:"), 3, 0)
        grid.addWidget(self.qlin_sig_I           , 3, 1)
        grid.addWidget(QtWid.QLabel("V")         , 3, 2)
        grid.setAlignment(QtCore.Qt.AlignTop)

        qgrp_readings = QtWid.QGroupBox("Readings")
        qgrp_readings.setStyleSheet(SS_GROUP)
        qgrp_readings.setLayout(grid)

        """
        # 'Chart'
        self.qpbt_clear_chart = QtWid.QPushButton("Clear")
        self.qpbt_clear_chart.clicked.connect(self.process_qpbt_clear_chart)

        grid = QtWid.QGridLayout()
        grid.addWidget(self.qpbt_clear_chart, 0, 0)
        grid.setAlignment(QtCore.Qt.AlignTop)

        qgrp_chart = QtWid.QGroupBox("Chart")
        qgrp_chart.setStyleSheet(SS_GROUP)
        qgrp_chart.setLayout(grid)
        """

        vbox = QtWid.QVBoxLayout()
        vbox.addWidget(self.qpbt_ENA_lockin)
        vbox.addWidget(qgrp_readings)
        #vbox.addWidget(qgrp_chart)
        vbox.addStretch()

        # Round up bottom frame
        hbox_bot = QtWid.QHBoxLayout()
        hbox_bot.addWidget(self.gw_chart, 1)
        hbox_bot.addLayout(vbox, 0)

        # -------------------------
        #   Round up full window
        # -------------------------

        vbox = QtWid.QVBoxLayout(self)
        vbox.addLayout(hbox_top, stretch=0)
        vbox.addSpacerItem(QtWid.QSpacerItem(0, 20))
        vbox.addLayout(hbox_bot, stretch=1)

    # --------------------------------------------------------------------------
    #   Handle controls
    # --------------------------------------------------------------------------

    @QtCore.pyqtSlot()
    def process_qpbt_ENA_lockin(self):
        if self.qpbt_ENA_lockin.isChecked():
            if ard.turn_on():
                self.qpbt_ENA_lockin.setText("lock-in ON")
                state.lockin_paused = False
                #ard.ser.flush()
                ard_pyqt.worker_DAQ.unpause()
                #app.processEvents()
        else:
            self.qpbt_ENA_lockin.setText("lock-in OFF")
            state.lockin_paused = True
            ard_pyqt.worker_DAQ.pause()
            ard.turn_off()
            #ard.ser.flush()
            #app.processEvents()
            
            window.qlbl_DAQ_rate.setText("Buffers/s: paused")

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

# ------------------------------------------------------------------------------
#   update_GUI
# ------------------------------------------------------------------------------

@QtCore.pyqtSlot()
def update_GUI_wall_clock():
    str_cur_date, str_cur_time = current_date_time_strings()
    window.qlbl_cur_date_time.setText("%s    %s" % (str_cur_date, str_cur_time))

@QtCore.pyqtSlot()
def update_GUI():
    window.qlbl_update_counter.setText("%i" % ard_pyqt.DAQ_update_counter)
    
    if not state.lockin_paused:
        window.qlbl_DAQ_rate.setText("Buffers/s: %.1f" % 
                                     ard_pyqt.obtained_DAQ_rate_Hz)
        window.qlin_time.setText("%i" % state.time[0])
        window.qlin_ref_X.setText("%.4f" % state.ref_X[0])
        window.qlin_ref_Y.setText("%.4f" % state.ref_Y[0])
        window.qlin_sig_I.setText("%.4f" % state.sig_I[0])
        
        window.CH_ref_X.update_curve()
        window.CH_ref_Y.update_curve()
        window.CH_sig_I.update_curve()
    
# ------------------------------------------------------------------------------
#   update_chart
# ------------------------------------------------------------------------------

@QtCore.pyqtSlot()
def update_chart():
    if DEBUG:
        tick = QDateTime.currentDateTime()

    # For future charts

    if DEBUG:
        tack = QDateTime.currentDateTime()
        dprint("  update_curve done in %d ms" % tick.msecsTo(tack))

# ------------------------------------------------------------------------------
#   Program termination routines
# ------------------------------------------------------------------------------

def stop_running():
    app.processEvents()
    ard_pyqt.close_all_threads()
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
                ard.name, ard.identity, '' if ard.is_alive else "not "))
    print("\nCRITICAL ERROR @ %s" % str_msg)
    reply = QtWid.QMessageBox.warning(window, "CRITICAL ERROR", str_msg,
                                      QtWid.QMessageBox.Ok)

    if reply == QtWid.QMessageBox.Ok:
        pass    # Leave the GUI open for read-only inspection by the user

@QtCore.pyqtSlot()
def about_to_quit():
    print("\nAbout to quit")
    stop_running()
    ard.close()

# ------------------------------------------------------------------------------
#   Lock-in amplifier data-acquisition update function
# ------------------------------------------------------------------------------

def lockin_DAQ_update():
    str_cur_date, str_cur_time = current_date_time_strings()
    
    [success, ans_bytes] = ard.listen_to_lockin_amp()

    # Prevent throwings errors when the lock-in was just set to pause
    if state.lockin_paused:
        return False
    
    if not(success):
        dprint("'%s' reports IOError @ %s %s" %
               (ard.name, str_cur_date, str_cur_time))
        return False
    
    state.buffers_received += 1
    N_samples = int(len(ans_bytes) / struct.calcsize('LHH'))
    if not(N_samples == ard.BUFFER_SIZE):
        print("Wrong bumber of samples received")
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
    mix_X = ref_X * sig_I
    mix_Y = ref_Y * sig_I
    
    hist_out_amp = 2 * np.sqrt(mix_X**2 + mix_Y**2)
    hist_out_phi = np.arctan(mix_Y / mix_X)
    
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
    state.hist_out_amp.extend(hist_out_amp)
    state.hist_out_phi.extend(hist_out_phi)
    
    window.CH_ref_X.add_new_readings(time, ref_X)
    window.CH_ref_Y.add_new_readings(time, ref_Y)
    window.CH_sig_I.add_new_readings(time, sig_I)
    
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
        #log_elapsed_time = (state.time - file_logger.start_time)/1e3  # [sec]
        #file_logger.write("samples received: %i\n" % N_samples)
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

    ard = lockin_functions.Arduino_lockin_amp(baudrate=1.5e6, read_timeout=.5)
    if not ard.auto_connect(Path("port_data.txt"), "Arduino lock-in amp"):
        sys.exit(0)
        
    ard.begin()
    ard.set_ref_freq(100)
    
    state.init_shift_buffers(N=3)

    """if not(ard.is_alive):
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
    window.qlbl_sample_rate.setText("SAMPLE RATE: %.2f Hz" % (1/ard.ISR_CLOCK))
    window.qlbl_buffer_size.setText("BUFFER SIZE  : %i" % ard.BUFFER_SIZE)

    # --------------------------------------------------------------------------
    #   File logger
    # --------------------------------------------------------------------------

    file_logger = FileLogger()
    file_logger.signal_set_recording_text.connect(window.set_text_qpbt_record)

    # --------------------------------------------------------------------------
    #   Set up communication threads for the Arduino
    # --------------------------------------------------------------------------

    # Create workers and threads
    ard_pyqt = Arduino_pyqt_lib.Arduino_pyqt(
            dev=ard,
            DAQ_function_to_run_each_update=lockin_DAQ_update,
            DAQ_critical_not_alive_count=np.nan,
            calc_DAQ_rate_every_N_iter=10)

    # Connect signals to slots
    ard_pyqt.signal_DAQ_updated.connect(update_GUI)
    ard_pyqt.signal_connection_lost.connect(notify_connection_lost)

    # Start threads
    ard_pyqt.start_thread_worker_DAQ(QtCore.QThread.TimeCriticalPriority)
    ard_pyqt.start_thread_worker_send()

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