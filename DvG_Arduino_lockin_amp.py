#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Arduino lock-in amplifier
"""
__author__      = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__         = "https://github.com/Dennis-van-Gils/DvG_Arduino_lock-in_amp"
__date__        = "22-03-2019"
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

from DvG_Buffered_FIR_Filter import Buffered_FIR_Filter

# Show debug info in terminal? Warning: Slow! Do not leave on unintentionally.
DEBUG = False

# ------------------------------------------------------------------------------
#   State
# ------------------------------------------------------------------------------

class State(object):
    """ Deque arrays needed for the second FIR filter stage, i.e. the low-pass
    filter after the heterodyne mixing.
    """
    def __init__(self, buffer_size, N_buffers_in_deque=0):
        self.buffer_size        = buffer_size           # [samples]
        self.N_buffers_in_deque = N_buffers_in_deque    # [int]
        self.N_deque = buffer_size * N_buffers_in_deque # [samples]
        
        self.deque_time       = deque(maxlen=self.N_deque)
        self.deque_sig_I_filt = deque(maxlen=self.N_deque)
        self.deque_mix_X      = deque(maxlen=self.N_deque)
        self.deque_mix_Y      = deque(maxlen=self.N_deque)
        self.deque_LIA_amp    = deque(maxlen=self.N_deque)
        self.deque_LIA_phi    = deque(maxlen=self.N_deque)
        
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
        window.qlin_time.setText("%i"    % lockin_pyqt.state.time[0])
        window.qlin_ref_X.setText("%.4f" % lockin_pyqt.state.ref_X[0])
        window.qlin_ref_Y.setText("%.4f" % lockin_pyqt.state.ref_Y[0])
        window.qlin_sig_I.setText("%.4f" % lockin_pyqt.state.sig_I[0])
        
        window.update_chart_refsig()
        window.update_chart_filt_BS()
        window.update_chart_mixer()
        window.update_chart_LIA_output()
        
@QtCore.pyqtSlot()
def process_qlin_set_ref_freq():
    try:
        ref_freq = float(window.qlin_set_ref_freq.text())
    except ValueError:
        ref_freq = lockin.config.ref_freq
    
    # Clip between 0 and the Nyquist frequency of the lock-in sampling rate
    ref_freq = np.clip(ref_freq, 0, 1/lockin.config.ISR_CLOCK/2)
    
    window.qlin_set_ref_freq.setText("%.2f" % ref_freq)
    if ref_freq != lockin.config.ref_freq:
        lockin_pyqt.set_ref_freq(ref_freq)
        
        firwin_cutoff = [0, 2*ref_freq - 1]
        firf_LP_mix_X.update_firwin_cutoff(firwin_cutoff)
        firf_LP_mix_Y.update_firwin_cutoff(firwin_cutoff)
        window.update_plot_filt_resp_LP(firf_LP_mix_X)
        
        QtWid.QApplication.processEvents()
    
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
    
    # Shorthands
    c: lockin_functions.Arduino_lockin_amp.Config = lockin.config
    state_1: lockin_pyqt_lib.Arduino_lockin_amp_pyqt.State = lockin_pyqt.state

    if lockin.lockin_paused:  # Prevent throwings errors if just paused
        return False
    
    [success, time, ref_X, ref_Y, sig_I] = lockin.listen_to_lockin_amp()
    if not(success):
        dprint("@ %s %s" % (str_cur_date, str_cur_time))
        return False
    
    # HACK: hard-coded calibration correction on the ADC
    dev_sig_I = sig_I * 0.0054 + 0.0020;
    #sig_I -= dev_sig_I

    lockin_pyqt.state.buffers_received += 1
    
    # Detect dropped buffers
    prev_last_deque_time = (state_1.deque_time[-1] if
                            state_1.buffers_received > 1
                            else np.nan)
    dT = (time[0] - prev_last_deque_time) / 1e6 # transform [usec] to [sec]
    if dT > (c.ISR_CLOCK)*1.05:  # Allow a few percent clock jitter
        print("dropped buffer %i" % state_1.buffers_received)
        N_dropped_buffers = int(round(dT / c.T_SPAN_BUFFER))
        print("N_dropped_buffers %i" % N_dropped_buffers)
        
        # Replace dropped buffers with ...
        N_dropped_samples = c.BUFFER_SIZE * N_dropped_buffers
        state_1.deque_time.extend(prev_last_deque_time +
                                  np.arange(1, N_dropped_samples + 1) *
                                  c.ISR_CLOCK)
        if 1:
            """ Proper: with np.nan samples.
            As a result, the filter output will contain a continuous series
            of np.nan values in the output for up to T_settling seconds long
            after the occurance of the last dropped buffer.
            """
            state_1.deque_ref_X.extend(np.array([np.nan] * N_dropped_samples))
            state_1.deque_ref_Y.extend(np.array([np.nan] * N_dropped_samples))
            state_1.deque_sig_I.extend(np.array([np.nan] * N_dropped_samples))
        else:
            """ Improper: with linearly interpolated samples.
            As a result, the filter output will contain fake data where
            ever dropped buffers occured. The advantage is that, in contrast
            to using above proper technique, the filter output remains a
            continuous series of values.
            """
            state_1.deque_sig_I.extend(state_1.deque_sig[-1] +
                                       np.arange(1, N_dropped_samples + 1) *
                                       (sig_I[0] - state_1.deque_sig[-1]) /
                                       N_dropped_samples)
            
    # Store first stage data into deque
    state_1.deque_time.extend(time)
    state_1.deque_ref_X.extend(ref_X)
    state_1.deque_ref_Y.extend(ref_Y)
    state_1.deque_sig_I.extend(sig_I)
    
    # Perform 50 Hz bandgap filter on sig_I
    sig_I_filt = firf_BS_sig_I.process(state_1.deque_sig_I)
    
    time_filt    = (np.array(state_1.deque_time, dtype=np.int64)
                    [firf_BS_sig_I.win_idx_valid_start:
                     firf_BS_sig_I.win_idx_valid_end])
    sig_I_unfilt = (np.array(state_1.deque_sig_I, dtype=np.float64)
                    [firf_BS_sig_I.win_idx_valid_start:
                     firf_BS_sig_I.win_idx_valid_end])
    
    # Retrieve the block of original data from the past that alligns with the
    # current filter output
    old_ref_X = (np.array(state_1.deque_ref_X, dtype=np.float64)
                 [firf_BS_sig_I.win_idx_valid_start:
                  firf_BS_sig_I.win_idx_valid_end])
    old_ref_Y = (np.array(state_1.deque_ref_Y, dtype=np.float64)
                 [firf_BS_sig_I.win_idx_valid_start:
                  firf_BS_sig_I.win_idx_valid_end])
    
    #if not len(time_filt) == 0:
    #    print("%i %i: %i" % (time[-1], time_filt[-1], time[-1] - time_filt[-1]))
        
    if firf_BS_sig_I.has_settled:    
        mix_X = (old_ref_X - c.ref_V_offset) * sig_I_filt
        mix_Y = (old_ref_Y - c.ref_V_offset) * sig_I_filt
    else:
        mix_X = np.array([np.nan] * c.BUFFER_SIZE)
        mix_Y = np.array([np.nan] * c.BUFFER_SIZE)
        
    # Store second stage data into deque
    state_2.deque_time.extend(time_filt)
    state_2.deque_sig_I_filt.extend(sig_I_filt)
    state_2.deque_mix_X.extend(mix_X)
    state_2.deque_mix_Y.extend(mix_Y)
    
    # Perform low-pass filter on the mixer output
    out_X = firf_LP_mix_X.process(state_2.deque_mix_X)
    out_Y = firf_LP_mix_Y.process(state_2.deque_mix_Y)
    
    time_2_filt = (np.array(state_2.deque_time, dtype=np.int64)
                   [firf_LP_mix_X.win_idx_valid_start:
                   firf_LP_mix_X.win_idx_valid_end])
        
    out_amp = np.sqrt(out_X**2 + out_Y**2)
    """NOTE: Because 'mix_X' and 'mix_Y' are both of type 'numpy.array', a
    division by (mix_X = 0) is handled correctly due to 'numpy.inf'. Likewise,
    'numpy.arctan(numpy.inf)' will result in pi/2. We suppress the
    RuntimeWarning: divide by zero encountered in true_divide.
    """
    np.seterr(divide='ignore')
    out_phi = np.arctan(out_Y / out_X)
    out_phi = out_phi/np.pi*180     # Transform [rad] to [deg]
    np.seterr(divide='warn')
    
    # Save state
    # TO DO: might be redundant to store this into State. Investigate
    state_1.time  = time
    state_1.ref_X = ref_X
    state_1.ref_Y = ref_Y
    state_1.sig_I = sig_I
    
    # Add new data to graphs
    window.CH_ref_X.add_new_readings(time, ref_X)
    window.CH_ref_Y.add_new_readings(time, ref_Y)
    window.CH_sig_I.add_new_readings(time, sig_I)
        
    window.CH_filt_BS_in.add_new_readings(time_filt, sig_I_unfilt)
    window.CH_filt_BS_out.add_new_readings(time_filt, sig_I_filt)
    
    window.CH_mix_X.add_new_readings(time_filt, mix_X)
    window.CH_mix_Y.add_new_readings(time_filt, mix_Y)
    
    window.CH_LIA_amp.add_new_readings(time_2_filt, out_amp)
    window.CH_LIA_phi.add_new_readings(time_2_filt, out_phi)
    
    # Logging to file
    if file_logger.starting:
        fn_log = QDateTime.currentDateTime().toString("yyMMdd_HHmmss") + ".txt"
        if file_logger.create_log(time, fn_log, mode='w'):
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
        
    #lockin.begin()
    lockin.begin(ref_freq=250, ref_V_offset=2.8, ref_V_ampl=0.4)
    
    # Create workers and threads
    lockin_pyqt = lockin_pyqt_lib.Arduino_lockin_amp_pyqt(
                            dev=lockin,
                            DAQ_function_to_run_each_update=lockin_DAQ_update,
                            DAQ_critical_not_alive_count=np.nan,
                            calc_DAQ_rate_every_N_iter=10,
                            N_buffers_in_deque=41,
                            DEBUG_worker_DAQ=False,
                            DEBUG_worker_send=False)
    lockin_pyqt.signal_DAQ_updated.connect(update_GUI)
    lockin_pyqt.signal_connection_lost.connect(notify_connection_lost)
    
    # --------------------------------------------------------------------------
    #   Set up state and filters
    # --------------------------------------------------------------------------
    state_2 = State(lockin_pyqt.state.buffer_size,
                    lockin_pyqt.state.N_buffers_in_deque)
    
    # Band-stop on sig_I
    firwin_cutoff = [0.5, 49.5, 50.5, 99.5, 100.5, lockin.config.F_Nyquist]
    firwin_window = ("chebwin", 50)
    firf_BS_sig_I = Buffered_FIR_Filter(lockin_pyqt.state.buffer_size,
                                        lockin_pyqt.state.N_buffers_in_deque,
                                        lockin.config.Fs,
                                        firwin_cutoff,
                                        firwin_window)

    # Low-pass on mix_X and mix_Y
    firwin_cutoff = [0, 2*lockin.config.ref_freq - 1]
    firwin_window = "blackman"
    firf_LP_mix_X = Buffered_FIR_Filter(lockin_pyqt.state.buffer_size,
                                        lockin_pyqt.state.N_buffers_in_deque,
                                        lockin.config.Fs,
                                        firwin_cutoff,
                                        firwin_window)
    firf_LP_mix_Y = Buffered_FIR_Filter(lockin_pyqt.state.buffer_size,
                                        lockin_pyqt.state.N_buffers_in_deque,
                                        lockin.config.Fs,
                                        firwin_cutoff,
                                        firwin_window)

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

    window.qlin_set_ref_freq.editingFinished.connect(process_qlin_set_ref_freq)

    window.pi_refsig.setYRange(2.35, 3.25)
    window.pi_filt_BS.setYRange(-.7, 3.5)
    window.pi_mixer.setYRange(-0.12, 0.2)
    window.pi_LIA_amp.setYRange(0.068, 0.092)
    window.pi_LIA_phi.setYRange(-92, 92)
    window.update_plot_filt_resp_BS(firf_BS_sig_I)
    window.update_plot_filt_resp_LP(firf_LP_mix_X)

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