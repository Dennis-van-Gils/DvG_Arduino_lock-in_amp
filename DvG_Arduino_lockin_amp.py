#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Arduino lock-in amplifier
"""
__author__      = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__         = "https://github.com/Dennis-van-Gils/DvG_Arduino_lock-in_amp"
__date__        = "25-03-2019"
__version__     = "1.0.0"

import os
import sys
from pathlib2 import Path

import psutil

from PyQt5 import QtCore
from PyQt5 import QtWidgets as QtWid
from PyQt5.QtCore import QDateTime
import numpy as np

from DvG_pyqt_FileLogger import FileLogger
from DvG_debug_functions import dprint#, print_fancy_traceback as pft

import DvG_Arduino_lockin_amp__GUI            as lockin_GUI
import DvG_dev_Arduino_lockin_amp__fun_serial as lockin_functions
import DvG_dev_Arduino_lockin_amp__pyqt_lib   as lockin_pyqt_lib

# Show debug info in terminal? Warning: Slow! Do not leave on unintentionally.
DEBUG = False
        
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
    str_cur_date, str_cur_time = current_date_time_strings()
    
    # Shorthands
    c: lockin_functions.Arduino_lockin_amp.Config = lockin.config
    state: lockin_pyqt_lib.Arduino_lockin_amp_pyqt.State = lockin_pyqt.state

    if lockin.lockin_paused:  # Prevent throwings errors if just paused
        return False
    
    [success, time, ref_X, ref_Y, sig_I] = lockin.listen_to_lockin_amp()
    if not(success):
        dprint("@ %s %s" % (str_cur_date, str_cur_time))
        return False
    
    # HACK: hard-coded calibration correction on the ADC
    dev_sig_I = sig_I * 0.0054 + 0.0020;
    sig_I -= dev_sig_I

    lockin_pyqt.state.buffers_received += 1
    
    # Detect dropped buffers
    prev_last_deque_time = (state.deque_time[-1] if
                            state.buffers_received > 1
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
            
    # Stage 0
    # -------
    
    state.time  = time
    state.ref_X = ref_X
    state.ref_Y = ref_Y
    state.sig_I = sig_I
    
    state.deque_time.extend(time)
    state.deque_ref_X.extend(ref_X)
    state.deque_ref_Y.extend(ref_Y)
    state.deque_sig_I.extend(sig_I)
    
    # Stage 1
    # -------
    
    # Apply band-stop filter to sig_I
    sig_I_filt = lockin_pyqt.firf_BS_sig_I.process(state.deque_sig_I)
    
    if lockin_pyqt.firf_BS_sig_I.has_settled:
        # Retrieve the block of original data from the past that alligns with
        # the current filter output
        time_1    = (np.array(state.deque_time, dtype=np.int64)
                     [lockin_pyqt.firf_BS_sig_I.win_idx_valid_start:
                      lockin_pyqt.firf_BS_sig_I.win_idx_valid_end])
        old_sig_I = (np.array(state.deque_sig_I, dtype=np.float64)
                     [lockin_pyqt.firf_BS_sig_I.win_idx_valid_start:
                      lockin_pyqt.firf_BS_sig_I.win_idx_valid_end])
        old_ref_X = (np.array(state.deque_ref_X, dtype=np.float64)
                     [lockin_pyqt.firf_BS_sig_I.win_idx_valid_start:
                      lockin_pyqt.firf_BS_sig_I.win_idx_valid_end])
        old_ref_Y = (np.array(state.deque_ref_Y, dtype=np.float64)
                     [lockin_pyqt.firf_BS_sig_I.win_idx_valid_start:
                      lockin_pyqt.firf_BS_sig_I.win_idx_valid_end])
        
        #if not len(time_filt) == 0:
        #    print("%i %i: %i" % (time[-1], time_filt[-1], time[-1] - time_filt[-1]))
    
        # Heterodyne mixing
        mix_X = (old_ref_X - c.ref_V_offset) * sig_I_filt
        mix_Y = (old_ref_Y - c.ref_V_offset) * sig_I_filt
        
        state.deque_time_1.extend(time_1)
        state.deque_sig_I_filt.extend(sig_I_filt)
        state.deque_mix_X.extend(mix_X)
        state.deque_mix_Y.extend(mix_Y)
    
        # Stage 2
        # -------
        
        # Apply low-pass filter to the mixer output
        out_X = lockin_pyqt.firf_LP_mix_X.process(state.deque_mix_X)
        out_Y = lockin_pyqt.firf_LP_mix_Y.process(state.deque_mix_Y)
            
        if lockin_pyqt.firf_LP_mix_X.has_settled:
            # Signal amplitude and phase reconstruction
            LIA_amp = np.sqrt(out_X**2 + out_Y**2)
            """NOTE: Because 'mix_X' and 'mix_Y' are both of type 'numpy.array', a
            division by (mix_X = 0) is handled correctly due to 'numpy.inf'. Likewise,
            'numpy.arctan(numpy.inf)' will result in pi/2. We suppress the
            RuntimeWarning: divide by zero encountered in true_divide.
            """
            np.seterr(divide='ignore')
            LIA_phi = np.arctan(out_Y / out_X)
            LIA_phi = LIA_phi/np.pi*180     # Transform [rad] to [deg]
            np.seterr(divide='warn')
            
            # Retrieve the block of original data from the past that alligns with the
            # current filter output
            time_2 = (np.array(state.deque_time_1, dtype=np.int64)
                      [lockin_pyqt.firf_LP_mix_X.win_idx_valid_start:
                       lockin_pyqt.firf_LP_mix_X.win_idx_valid_end])
            
            state.deque_time_2.extend(time_2)
            state.deque_LIA_amp.extend(LIA_amp)
            state.deque_LIA_phi.extend(LIA_phi)
    
    # Add new data to graphs
    # ----------------------
    
    window.CH_ref_X.add_new_readings(time, ref_X)
    window.CH_ref_Y.add_new_readings(time, ref_Y)
    window.CH_sig_I.add_new_readings(time, sig_I)
        
    if lockin_pyqt.firf_BS_sig_I.has_settled:
        window.CH_filt_BS_in.add_new_readings(time_1, old_sig_I)
        window.CH_filt_BS_out.add_new_readings(time_1, sig_I_filt)
    
        window.CH_mix_X.add_new_readings(time_1, mix_X)
        window.CH_mix_Y.add_new_readings(time_1, mix_Y)
    
        if lockin_pyqt.firf_LP_mix_X.has_settled:
            window.CH_LIA_amp.add_new_readings(time_2, LIA_amp)
            window.CH_LIA_phi.add_new_readings(time_2, LIA_phi)
            #window.CH_LIA_amp.add_new_readings(time_2, out_X)
            #window.CH_LIA_phi.add_new_readings(time_2, out_Y)
    
    # Logging to file
    #----------------
    
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

    window.pi_refsig.setYRange(2.35, 3.25)
    window.pi_filt_BS.setYRange(-.7, 3.5)
    window.pi_mixer.setYRange(-0.12, 0.2)
    window.pi_LIA_amp.setYRange(0.068, 0.092)
    window.pi_LIA_phi.setYRange(-92, 92)

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