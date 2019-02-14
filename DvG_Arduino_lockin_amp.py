#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Arduino lock-in amplifier
"""
__author__      = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__         = "https://github.com/Dennis-van-Gils/DvG_Arduino_lock-in_amp"
__date__        = "30-01-2019"
__version__     = "1.0.0"

import os
import sys
from pathlib2 import Path

import psutil

from PyQt5 import QtCore
from PyQt5 import QtWidgets as QtWid
from PyQt5.QtCore import QDateTime
import numpy as np
import time as Time

from collections import deque

from DvG_pyqt_FileLogger import FileLogger
from DvG_debug_functions import dprint, print_fancy_traceback as pft

import DvG_Arduino_lockin_amp__GUI            as lockin_GUI
import DvG_dev_Arduino_lockin_amp__fun_serial as lockin_functions
import DvG_dev_Arduino_lockin_amp__pyqt_lib   as lockin_pyqt_lib

from DvG_Buffered_FIR_Filter import DvG_Buffered_FIR_Filter

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
        These arrays are N_buffers_in_deque times the buffer size of the Arduino
        in order to facilitate preventing start/end effects due to FIR
        filtering. A smaller shifting window can walk over these arrays and a
        FIR filter using convolution can be applied, where only the valid
        samples are kept (no zero padding).
        
        Each time a complete buffer of BLOCK_SIZE samples is received from the
        Arduino, it is appended to the end of these arrays (FIFO shift buffer).
        
            i.e. N_buffers_in_deque = 3    
                hist_time = [buffer_1; buffer_2; buffer_3]
                hist_time = [buffer_2; buffer_3; buffer_4]
                hist_time = [buffer_3; buffer_4; buffer_5]
                etc...
        """
        
        # Create deque buffers
        self.N_buffers_in_deque = 81
        self.N_deque = lockin.config.BUFFER_SIZE * self.N_buffers_in_deque # [samples]
        self.deque_time  = deque(maxlen=self.N_deque)
        self.deque_ref_X = deque(maxlen=self.N_deque)
        self.deque_ref_Y = deque(maxlen=self.N_deque)
        self.deque_sig_I = deque(maxlen=self.N_deque)
        
        """
        self.hist_time    = deque(maxlen=maxlen)
        self.hist_ref_X   = deque(maxlen=maxlen)
        self.hist_ref_Y   = deque(maxlen=maxlen)
        self.hist_sig_I   = deque(maxlen=maxlen)
        self.hist_mix_X   = deque(maxlen=maxlen)
        self.hist_mix_Y   = deque(maxlen=maxlen)
        self.hist_out_amp = deque(maxlen=maxlen)
        self.hist_out_phi = deque(maxlen=maxlen)
        """

        # Mutex for proper multithreading. If the state variables are not
        # atomic or thread-safe, you should lock and unlock this mutex for each
        # read and write operation. In this demo we don't need it, but I keep it
        # as reminder.
        self.mutex = QtCore.QMutex()

# ------------------------------------------------------------------------------
#   Update GUI routines
# ------------------------------------------------------------------------------

def current_date_time_strings():
    cur_date_time = QDateTime.currentDateTime()
    return (cur_date_time.toString("dd-MM-yyyy"),
            cur_date_time.toString("HH:mm:ss"))

@QtCore.pyqtSlot()
def update_GUI():
    window.qlbl_update_counter.setText("%i" % lockin_pyqt.DAQ_update_counter)
    
    if not lockin.lockin_paused:
        window.qlbl_DAQ_rate.setText("Buffers/s: %.1f" % 
                                     lockin_pyqt.obtained_DAQ_rate_Hz)
        window.qlin_time.setText("%i"    % state.time[0])
        window.qlin_ref_X.setText("%.4f" % state.ref_X[0])
        window.qlin_ref_Y.setText("%.4f" % state.ref_Y[0])
        window.qlin_sig_I.setText("%.4f" % state.sig_I[0])
        
        window.update_chart_refsig()
        window.update_chart_mixer()
        
# ------------------------------------------------------------------------------
#   Program termination routines
# ------------------------------------------------------------------------------

def stop_running():
    app.processEvents()
    if lockin.is_alive: lockin_pyqt.turn_off_immediately()
    lockin_pyqt.close_all_threads()
    file_logger.close_log()

@QtCore.pyqtSlot()
def notify_connection_lost():
    stop_running()

    excl = "    ! ! ! ! ! ! ! !    "
    window.qlbl_title.setText("%sLOST CONNECTION%s" % (excl, excl))

    str_cur_date, str_cur_time = current_date_time_strings()
    str_msg = ("%s %s\nLost connection to Arduino on port %s.\n" %
              (str_cur_date, str_cur_time, lockin.ser.portstr))
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
    c: lockin_functions.Arduino_lockin_amp.Config = lockin.config  # Shorthand

    if lockin.lockin_paused:  # Prevent throwings errors if just paused
        return False
    
    [success, time, ref_X, ref_Y, sig_I] = lockin.listen_to_lockin_amp()
    if not(success):
        dprint("@ %s %s" % (str_cur_date, str_cur_time))
        return False

    state.buffers_received += 1
        
    # Detect dropped buffers
    prev_last_deque_time = (state.deque_time[-1] if state.buffers_received > 1
                            else np.nan)
    dT = (time[0] - prev_last_deque_time) / 1e6 # transform [usec] to [sec]
    if dT > (c.ISR_CLOCK)*1.05:  # Allow a few percent clock jitter
        print("dropped buffer %i" % state.buffers_received)
        N_dropped_buffers = int(round(dT / c.T_SPAN_BUFFER))
        print("N_dropped_buffers %i" % N_dropped_buffers)
        
        # Replace dropped buffers with ...
        N_dropped_samples = c.BUFFER_SIZE * N_dropped_buffers
        state.deque_time.extend(prev_last_deque_time +
                                np.arange(1, N_dropped_samples + 1) *
                                c.ISR_CLOCK)
        if 1:
            """ Proper: with np.nan samples.
            As a result, the filter output will contain a continuous series
            of np.nan values in the output for up to T_settling seconds long
            after the occurance of the last dropped buffer.
            """
            state.deque_ref_X.extend(np.array([np.nan] * N_dropped_samples))
            state.deque_ref_Y.extend(np.array([np.nan] * N_dropped_samples))
            state.deque_sig_I.extend(np.array([np.nan] * N_dropped_samples))
        else:
            """ Improper: with linearly interpolated samples.
            As a result, the filter output will contain fake data where
            ever dropped buffers occured. The advantage is that, in contrast
            to using above proper technique, the filter output remains a
            continuous series of values.
            """
            state.deque_sig_I.extend(state.deque_sig[-1] +
                                     np.arange(1, N_dropped_samples + 1) *
                                     (sig_I[0] - state.deque_sig[-1]) /
                                     N_dropped_samples)
            
    state.deque_time.extend(time)
    state.deque_ref_X.extend(ref_X)
    state.deque_ref_Y.extend(ref_Y)
    state.deque_sig_I.extend(sig_I)
    
    # Perform 50 Hz bandgap filter on sig_I
    tick = Time.time()
    sig_I_filt = firf_BG_50Hz.process(state.deque_sig_I)
    print("%.2f" % ((Time.time() - tick) * 1000))
    time_filt = (np.array(state.deque_time)[firf_BG_50Hz.win_idx_valid_start:
                                            firf_BG_50Hz.win_idx_valid_end])
    sig_I_unfilt = (np.array(state.deque_sig_I)[firf_BG_50Hz.win_idx_valid_start:
                                                firf_BG_50Hz.win_idx_valid_end])
    
    mix_X = (ref_X - c.ref_V_center) * (sig_I - c.ref_V_center)
    mix_Y = (ref_Y - c.ref_V_center) * (sig_I - c.ref_V_center)
    
    out_amp = 2 * np.sqrt(mix_X**2 + mix_Y**2)
    """NOTE: Because 'mix_X' and 'mix_Y' are both of type 'numpy.array', a
    division by (mix_X = 0) is handled correctly due to 'numpy.inf'. Likewise,
    'numpy.arctan(numpy.inf)' will result in pi/2. We suppress the
    RuntimeWarning: divide by zero encountered in true_divide.
    """
    np.seterr(divide='ignore')
    out_phi = np.arctan(mix_Y / mix_X)
    np.seterr(divide='warn')
    
    # Save state
    state.time  = time
    state.ref_X = ref_X
    state.ref_Y = ref_Y
    state.sig_I = sig_I
    
    # Add new data to graphs
    window.CH_ref_X.add_new_readings(time, ref_X)
    window.CH_ref_Y.add_new_readings(time, ref_Y)
    window.CH_sig_I.add_new_readings(time, sig_I)
    
    window.CH_mix_X.add_new_readings(time_filt, sig_I_unfilt)
    window.CH_mix_Y.add_new_readings(time_filt, sig_I_filt)
    
    #print(Time.time() - tick)
    
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
        for i in range(c.BUFFER_SIZE):
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
    #   Arduino
    # --------------------------------------------------------------------------

    # Connect to Arduino
    lockin = lockin_functions.Arduino_lockin_amp(baudrate=1e6, read_timeout=5)
    if not lockin.auto_connect(Path("port_data.txt"), "Arduino lock-in amp"):
        print("\nCheck connection and try resetting the Arduino.")
        print("Exiting...\n")
        sys.exit(0)
        
    lockin.begin(ref_freq=100)
    
    # Set up state and filters
    state = State()

    #firwin_cutoff = [0.0001, 49.5, 50.5, lockin.config.Fs/2-0.0001]
    firwin_cutoff = [0.0001, 120]
    firf_BG_50Hz = DvG_Buffered_FIR_Filter(lockin.config.BUFFER_SIZE,
                                           state.N_buffers_in_deque,
                                           lockin.config.Fs,
                                           firwin_cutoff,
                                           ("chebwin", 50))
    firf_BG_50Hz.report()

    # Create workers and threads
    lockin_pyqt = lockin_pyqt_lib.Arduino_lockin_amp_pyqt(
                            dev=lockin,
                            DAQ_function_to_run_each_update=lockin_DAQ_update,
                            DAQ_critical_not_alive_count=np.nan,
                            calc_DAQ_rate_every_N_iter=10,
                            DEBUG_worker_DAQ=False,
                            DEBUG_worker_send=False)
    lockin_pyqt.signal_DAQ_updated.connect(update_GUI)
    lockin_pyqt.signal_connection_lost.connect(notify_connection_lost)
    
    # Manage logging to disk
    file_logger = FileLogger()

    # --------------------------------------------------------------------------
    #   Create application and main window
    # --------------------------------------------------------------------------
    QtCore.QThread.currentThread().setObjectName('MAIN')    # For DEBUG info

    app = 0    # Work-around for kernel crash when using Spyder IDE
    app = QtWid.QApplication(sys.argv)
    app.aboutToQuit.connect(about_to_quit)

    window = lockin_GUI.MainWindow(lockin=lockin,
                                   lockin_pyqt=lockin_pyqt,
                                   file_logger=file_logger)

    # --------------------------------------------------------------------------
    #   Start threads
    # --------------------------------------------------------------------------
    
    lockin_pyqt.start_thread_worker_DAQ(QtCore.QThread.TimeCriticalPriority)
    lockin_pyqt.start_thread_worker_send()

    # --------------------------------------------------------------------------
    #   Start the main GUI event loop
    # --------------------------------------------------------------------------

    window.show()
    sys.exit(app.exec_())