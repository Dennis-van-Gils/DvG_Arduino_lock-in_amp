#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Arduino lock-in amplifier
"""
__author__      = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__         = "https://github.com/Dennis-van-Gils/DvG_Arduino_lock-in_amp"
__date__        = "05-06-2019"
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
    """Listen for new data buffers send by the lock-in amplifier and perform the
    main mathematical operations for signal processing. This function will run
    in a dedicated thread (i.e. worker_DAQ), separated from the main program
    thread that handles the GUI.
    NOTE: NO (SLOW) GUI OPERATIONS ARE ALLOWED HERE. Otherwise it will affect
    the worker_DAQ thread negatively, resulting in lost buffers.
    """
    # Shorthands
    c: lockin_functions.Arduino_lockin_amp.Config = lockin.config
    state: lockin_pyqt_lib.Arduino_lockin_amp_pyqt.State = lockin_pyqt.state

    # Prevent throwings errors if just paused
    if lockin.lockin_paused:
        return False
    
    if not(window.boost_fps_graphing):
        # Prevent possible concurrent pyqtgraph.GraphicsWindow() redraws and GUI
        # events when doing heavy calculations to unburden the CPU and prevent
        # dropped buffers. Dropped graphing frames are prefereable to dropped
        # data buffers.
        for gw in window.gws_all:
            gw.setUpdatesEnabled(False)
    
    # Listen for data buffers send by the lock-in
    [success, time, ref_X, ref_Y, sig_I] = lockin.listen_to_lockin_amp()
    if not(success):
        dprint("@ %s %s" % current_date_time_strings())
        return False
    
    if window.boost_fps_graphing:
        # Prevent possible concurrent pyqtgraph.GraphicsWindow() redraws and GUI
        # events when doing heavy calculations to unburden the CPU and prevent
        # dropped buffers. Dropped graphing frames are prefereable to dropped
        # data buffers.
        for gw in window.gws_all:
            gw.setUpdatesEnabled(False)
    
    # HACK: hard-coded calibration correction on the ADC
    # TODO: make a self-calibration procedure and store correction results
    # on non-volatile memory of the microprocessor.
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
        # np.nan values in the output for up to DvG_Buffered_FIR_Filter.
        # Buffered_FIR_Filter().T_settle_deque seconds long after the occurrence
        # of the last dropped sample.
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
    
    # Apply filter 1 to sig_I
    filt_I = lockin_pyqt.firf_1_sig_I.process(state.deque_sig_I)
    
    # Retrieve the block of original data from the past that aligns with
    # the current filter output
    time_1    = (np.array(state.deque_time, dtype=c.return_type_time)
                 [lockin_pyqt.firf_1_sig_I.win_idx_valid_start:
                  lockin_pyqt.firf_1_sig_I.win_idx_valid_end])
    old_sig_I = (np.array(state.deque_sig_I, dtype=c.return_type_sig_I)
                 [lockin_pyqt.firf_1_sig_I.win_idx_valid_start:
                  lockin_pyqt.firf_1_sig_I.win_idx_valid_end])
    
    if lockin_pyqt.firf_1_sig_I.deque_has_settled:
        old_ref_X = (np.array(state.deque_ref_X, dtype=c.return_type_ref_X)
                     [lockin_pyqt.firf_1_sig_I.win_idx_valid_start:
                      lockin_pyqt.firf_1_sig_I.win_idx_valid_end])
        old_ref_Y = (np.array(state.deque_ref_Y, dtype=c.return_type_ref_X)
                     [lockin_pyqt.firf_1_sig_I.win_idx_valid_start:
                      lockin_pyqt.firf_1_sig_I.win_idx_valid_end])
        
        # Heterodyne mixing
        # TODO: use np.multiply(x, 10, out=y), see np.ufunc. Speeds up by non buffering
        # See https://jakevdp.github.io/PythonDataScienceHandbook/02.03-computation-on-arrays-ufuncs.html
        mix_X = (old_ref_X - c.ref_V_offset) * filt_I  
        mix_Y = (old_ref_Y - c.ref_V_offset) * filt_I
    else:
        mix_X = np.full(c.BUFFER_SIZE, np.nan)
        mix_Y = np.full(c.BUFFER_SIZE, np.nan)
    
    state.deque_time_1.extend(time_1)
    state.deque_filt_I.extend(filt_I)
    state.deque_mix_X.extend(mix_X)
    state.deque_mix_Y.extend(mix_Y)
    
    # Stage 2
    # -------
    
    # Apply filter 2 to the mixer output
    out_X = lockin_pyqt.firf_2_mix_X.process(state.deque_mix_X)
    out_Y = lockin_pyqt.firf_2_mix_Y.process(state.deque_mix_Y)
    
    # Retrieve the block of original data from the past that aligns with
    # the current filter output
    time_2 = (np.array(state.deque_time_1, dtype=c.return_type_time)
              [lockin_pyqt.firf_2_mix_X.win_idx_valid_start:
               lockin_pyqt.firf_2_mix_X.win_idx_valid_end])
            
    if lockin_pyqt.firf_2_mix_X.deque_has_settled:
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

    # Power spectra
    # -------------
    # Will only compute the power spectrum if the checkbox is checked in the
    # legend. Calculating power spectra is a heavy burden for the CPU and
    # slower computers will suffer by this. Hence, only compute when requested
    # by the user, instead of always computing.

    if window.legend_box_PS.chkbs[0].isChecked():
        [f, P_dB] = lockin_pyqt.compute_power_spectrum(state.deque_sig_I)
        if len(f) > 0: window.BP_PS_1.set_data(f, P_dB)
            
    if window.legend_box_PS.chkbs[1].isChecked():
        [f, P_dB] = lockin_pyqt.compute_power_spectrum(state.deque_filt_I)
        if len(f) > 0: window.BP_PS_2.set_data(f, P_dB)
        
    if window.legend_box_PS.chkbs[2].isChecked():
        [f, P_dB] = lockin_pyqt.compute_power_spectrum(state.deque_mix_X)
        if len(f) > 0: window.BP_PS_3.set_data(f, P_dB)
        
    if window.legend_box_PS.chkbs[3].isChecked():
        [f, P_dB] = lockin_pyqt.compute_power_spectrum(state.deque_mix_Y)
        if len(f) > 0: window.BP_PS_4.set_data(f, P_dB)
        
    if window.legend_box_PS.chkbs[4].isChecked():
        [f, P_dB] = lockin_pyqt.compute_power_spectrum(state.deque_out_R)
        if len(f) > 0: window.BP_PS_5.set_data(f, P_dB)
    
    # Add new data to charts
    # ----------------------
    
    window.CH_ref_X.add_new_readings(time, ref_X)
    window.CH_ref_Y.add_new_readings(time, ref_Y)
    window.CH_sig_I.add_new_readings(time, sig_I)
    window.CH_filt_1_in.add_new_readings(time_1, old_sig_I)
    window.CH_filt_1_out.add_new_readings(time_1, filt_I)
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
                      "ref_X*[V]\t"
                      "ref_Y*[V]\t"
                      "sig_I[V]\t"
                      "filt_I[V]\t"
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
        if lockin_pyqt.firf_2_mix_X.deque_has_settled:
            idx_offset = lockin_pyqt.firf_1_sig_I.win_idx_valid_start
            for i in range(c.BUFFER_SIZE):
                data = (("%i\t" +
                         "%.5f\t" * 9 +
                         "%.4f\n") % (
                        state.deque_time[i],
                        state.deque_ref_X[i],
                        state.deque_ref_Y[i],
                        state.deque_sig_I[i],
                        state.deque_filt_I[i + idx_offset],
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
    
    # Re-enable pyqtgraph.GraphicsWindow() redraws and GUI events
    for gw in window.gws_all:
        gw.setUpdatesEnabled(True)
    
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
        
    lockin.begin()
    #lockin.begin(ref_freq=110, ref_V_offset=1.7, ref_V_ampl=1.414)
    
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