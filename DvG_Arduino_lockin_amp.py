#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Arduino lock-in amplifier
"""
__author__      = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__         = "https://github.com/Dennis-van-Gils/DvG_Arduino_lock-in_amp"
__date__        = "02-04-2019"
__version__     = "1.0.0"

import os
import sys
from pathlib2 import Path

import psutil

from PyQt5 import QtCore
from PyQt5 import QtWidgets as QtWid
from PyQt5.QtCore import QDateTime
import numpy as np
from scipy.signal import welch

from DvG_pyqt_FileLogger import FileLogger
from DvG_debug_functions import dprint#, print_fancy_traceback as pft

import DvG_Arduino_lockin_amp__GUI            as lockin_GUI
import DvG_dev_Arduino_lockin_amp__fun_serial as lockin_functions
import DvG_dev_Arduino_lockin_amp__pyqt_lib   as lockin_pyqt_lib

# Show debug info in terminal? Warning: Slow! Do not leave on unintentionally.
DEBUG = False

# TODO: Handle Arduino timer roll-over at t = 4294967295 us correctly. Current
# Python code will think there are dropped samples at timer roll-over.
# Happens every 71.2 minutes.
# https://arduino.stackexchange.com/questions/12587/how-can-i-handle-the-millis-rollover
        
# ------------------------------------------------------------------------------
#   Update GUI routines
# ------------------------------------------------------------------------------

def current_date_time_strings():
    cur_date_time = QDateTime.currentDateTime()
    return (cur_date_time.toString("dd-MM-yyyy"),
            cur_date_time.toString("HH:mm:ss"))
        
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
    """Performs the main mathematical operations for lock-in signal processing.
    NOTE: NO (SLOW) GUI OPERATIONS ARE ALLOWED HERE. Otherwise it will affect
    the worker_DAQ thread negatively, resulting in lost buffers.
    """
    # Shorthands
    c: lockin_functions.Arduino_lockin_amp.Config = lockin.config
    state: lockin_pyqt_lib.Arduino_lockin_amp_pyqt.State = lockin_pyqt.state

    if lockin.lockin_paused:  # Prevent throwings errors if just paused
        return False
    
    [success, time, ref_X, ref_Y, sig_I] = lockin.listen_to_lockin_amp()
    if not(success):
        dprint("@ %s %s" % current_date_time_strings())
        return False
    
    # HACK: hard-coded calibration correction on the ADC
    dev_sig_I = sig_I * 0.0054 + 0.0020;
    sig_I -= dev_sig_I
    
    # Detect dropped samples / buffers
    lockin_pyqt.state.buffers_received += 1
    prev_last_deque_time = (state.deque_time[-1] if state.buffers_received > 1
                            else np.nan)
    dT = (time[0] - prev_last_deque_time) / 1e6 # Transform [usec] to [sec]
    if dT > (c.ISR_CLOCK)*1.05: # Allow a few percent clock jitter
        N_dropped_samples = int(round(dT / c.ISR_CLOCK) - 1)
        print("Dropped samples: idx %i, %i" %
              (state.buffers_received, N_dropped_samples))
        
        # Replace dropped samples with np.nan samples.
        # As a result, the filter output will contain a continuous series of
        # np.nan values in the output for up to T_settling seconds long after
        # the occurrence of the last dropped sample.
        state.deque_time.extend(prev_last_deque_time +
                                np.arange(1, N_dropped_samples + 1) *
                                c.ISR_CLOCK)
        state.deque_ref_X.extend(np.full(N_dropped_samples, np.nan))
        state.deque_ref_Y.extend(np.full(N_dropped_samples, np.nan))
        state.deque_sig_I.extend(np.full(N_dropped_samples, np.nan))
        
    # Stage 0
    # -------
    
    state.time  = time
    state.ref_X = ref_X
    state.ref_Y = ref_Y
    state.sig_I = sig_I
    
    state.sig_I_min = np.min(sig_I)
    state.sig_I_max = np.max(sig_I)
    state.sig_I_avg = np.mean(sig_I)
    state.sig_I_std = np.std(sig_I)

    state.deque_time.extend(time)
    state.deque_ref_X.extend(ref_X)
    state.deque_ref_Y.extend(ref_Y)
    state.deque_sig_I.extend(sig_I)

    # Stage 1
    # -------
    
    # Apply band-stop filter to sig_I
    sig_I_filt = lockin_pyqt.firf_BS_sig_I.process(state.deque_sig_I)
    
    # Retrieve the block of original data from the past that alligns with
    # the current filter output
    time_1    = (np.array(state.deque_time, dtype=np.int64)
                 [lockin_pyqt.firf_BS_sig_I.win_idx_valid_start:
                  lockin_pyqt.firf_BS_sig_I.win_idx_valid_end])
    old_sig_I = (np.array(state.deque_sig_I, dtype=np.float64)
                 [lockin_pyqt.firf_BS_sig_I.win_idx_valid_start:
                  lockin_pyqt.firf_BS_sig_I.win_idx_valid_end])
    
    if lockin_pyqt.firf_BS_sig_I.has_settled:
        old_ref_X = (np.array(state.deque_ref_X, dtype=np.float64)
                     [lockin_pyqt.firf_BS_sig_I.win_idx_valid_start:
                      lockin_pyqt.firf_BS_sig_I.win_idx_valid_end])
        old_ref_Y = (np.array(state.deque_ref_Y, dtype=np.float64)
                     [lockin_pyqt.firf_BS_sig_I.win_idx_valid_start:
                      lockin_pyqt.firf_BS_sig_I.win_idx_valid_end])
        
        # Heterodyne mixing
        mix_X = (old_ref_X - c.ref_V_offset) * sig_I_filt
        mix_Y = (old_ref_Y - c.ref_V_offset) * sig_I_filt
    else:
        mix_X = np.full(c.BUFFER_SIZE, np.nan)
        mix_Y = np.full(c.BUFFER_SIZE, np.nan)
    
    state.deque_time_1.extend(time_1)
    state.deque_sig_I_filt.extend(sig_I_filt)
    state.deque_mix_X.extend(mix_X)
    state.deque_mix_Y.extend(mix_Y)
    
    # Stage 2
    # -------
    
    # Apply low-pass filter to the mixer output
    out_X = lockin_pyqt.firf_LP_mix_X.process(state.deque_mix_X)
    out_Y = lockin_pyqt.firf_LP_mix_Y.process(state.deque_mix_Y)
    
    # Retrieve the block of original data from the past that alligns with
    # the current filter output
    time_2 = (np.array(state.deque_time_1, dtype=np.int64)
              [lockin_pyqt.firf_LP_mix_X.win_idx_valid_start:
               lockin_pyqt.firf_LP_mix_X.win_idx_valid_end])
            
    if lockin_pyqt.firf_LP_mix_X.has_settled:
        # Signal amplitude and phase reconstruction
        out_R = np.sqrt(out_X**2 + out_Y**2)
        
        # NOTE: Because 'mix_X' and 'mix_Y' are both of type 'numpy.array', a
        # division by (mix_X = 0) is handled correctly due to 'numpy.inf'.
        # Likewise, 'numpy.arctan(numpy.inf)' will result in pi/2. We suppress
        # the RuntimeWarning: divide by zero encountered in true_divide.
        np.seterr(divide='ignore')
        out_T = np.arctan(out_Y / out_X)
        out_T = out_T/np.pi*180     # Transform [rad] to [deg]
        np.seterr(divide='warn')
    else:
        out_R = np.full(c.BUFFER_SIZE, np.nan)
        out_T = np.full(c.BUFFER_SIZE, np.nan)
    
    state.time2 = time_2
    state.X     = out_X
    state.Y     = out_Y
    state.R     = out_R
    state.T     = out_T
    
    state.deque_time_2.extend(time_2)
    state.deque_out_X.extend(out_X)
    state.deque_out_Y.extend(out_Y)
    state.deque_out_R.extend(out_R)
    state.deque_out_T.extend(out_T)
        
    # Power spectrum
    # --------------
    if len(state.deque_sig_I) == state.deque_sig_I.maxlen:
        [f, Pxx] = welch(state.deque_sig_I, fs=c.Fs, nperseg=10250,
                         scaling='density')
       
        window.BP_power_spectrum.set_data(f, Pxx)
    
    # Add new data to charts
    # ----------------------
    
    window.CH_ref_X.add_new_readings(time, ref_X)
    window.CH_ref_Y.add_new_readings(time, ref_Y)
    window.CH_sig_I.add_new_readings(time, sig_I)
    window.CH_filt_BS_in.add_new_readings(time_1, old_sig_I)
    window.CH_filt_BS_out.add_new_readings(time_1, sig_I_filt)
    window.CH_mix_X.add_new_readings(time_1, mix_X)
    window.CH_mix_Y.add_new_readings(time_1, mix_Y)        
    if window.qrbt_XR_X.isChecked():
        window.CH_LIA_XR.add_new_readings(time_2, out_X)
    else:
        window.CH_LIA_XR.add_new_readings(time_2, out_R)
    if window.qrbt_YT_Y.isChecked():
        window.CH_LIA_YT.add_new_readings(time_2, out_Y)
    else:
        window.CH_LIA_YT.add_new_readings(time_2, out_T)
    
    # Logging to file
    #----------------

    if file_logger.starting:
        fn_log = QDateTime.currentDateTime().toString("yyMMdd_HHmmss") + ".txt"
        if file_logger.create_log(time, fn_log, mode='w'):
            file_logger.signal_set_recording_text.emit(
                "Recording to file: " + fn_log)
            header = ("time[us]\t"
                      "ref_X[V]\t"
                      "ref_Y[V]\t"
                      "sig_I[V]\t"
                      "sig_I_BS[V]\t"
                      "mix_X[V]\t"
                      "mix_Y[V]\t"
                      "X[V]\t"
                      "Y[V]\t"
                      "R[V]\t"
                      "T[deg]\n")
            file_logger.write(header)
            #file_logger.write("time[us]\tref_X[V]\tref_Y[V]\tsig_I[V]\n")

    if file_logger.stopping:
        file_logger.signal_set_recording_text.emit(
            "Click to start recording to file")
        file_logger.close_log()

    if file_logger.is_recording:
        if lockin_pyqt.firf_LP_mix_X.has_settled:
            idx_offset = lockin_pyqt.firf_BS_sig_I.win_idx_valid_start
            for i in range(c.BUFFER_SIZE):
                data = (("%i\t" +
                         "%.5f\t" * 9 +
                         "%.4f\n") % (
                        state.deque_time[i],
                        state.deque_ref_X[i],
                        state.deque_ref_Y[i],
                        state.deque_sig_I[i],
                        state.deque_sig_I_filt[i + idx_offset],
                        state.deque_mix_X[i + idx_offset],
                        state.deque_mix_Y[i + idx_offset],
                        out_X[i],
                        out_Y[i],
                        out_R[i],
                        out_T[i]                    
                        ))
                file_logger.write(data)
            #file_logger.write("%i\t%.4f\t%.4f\t%.4f\n" % 
            #                  (time[i], ref_X[i], ref_Y[i], sig_I[i]))

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
    lockin = lockin_functions.Arduino_lockin_amp(baudrate=1e6, read_timeout=4)
    if not lockin.auto_connect(Path("port_data.txt"), "Arduino lock-in amp"):
        print("\nCheck connection and try resetting the Arduino.")
        print("Exiting...\n")
        sys.exit(0)
        
    #lockin.begin()
    lockin.begin(ref_freq=110, ref_V_offset=1.7, ref_V_ampl=1.414)
    
    # Create workers and threads
    lockin_pyqt = lockin_pyqt_lib.Arduino_lockin_amp_pyqt(
                            dev=lockin,
                            DAQ_function_to_run_each_update=lockin_DAQ_update,
                            DAQ_critical_not_alive_count=3,
                            calc_DAQ_rate_every_N_iter=10,
                            N_buffers_in_deque=41,
                            DEBUG_worker_DAQ=False,
                            DEBUG_worker_send=False)
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

    window.pi_refsig.setYRange(0.2, 3.2)
    window.pi_filt_BS.setYRange(-1.8, 3.4)
    window.pi_mixer.setYRange(-1.2, 2.2)
    window.pi_XR.setYRange(0.99, 1.01)
    window.pi_YT.setYRange(-92, 92)

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