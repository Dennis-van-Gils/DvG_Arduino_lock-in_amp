#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Arduino lock-in amplifier
"""
__author__      = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__         = "https://github.com/Dennis-van-Gils/DvG_Arduino_lock-in_amp"
__date__        = "12-08-2019"
__version__     = "1.0.0"

import os
import sys
from pathlib2 import Path
#import time as Time

import psutil

from PyQt5 import QtCore
from PyQt5 import QtWidgets as QtWid
from PyQt5.QtCore import QDateTime
import numpy as np

from DvG_pyqt_FileLogger import FileLogger
from DvG_debug_functions import dprint#, print_fancy_traceback as pft
from DvG_FFTW_WelchPowerSpectrum import FFTW_WelchPowerSpectrum

import DvG_Arduino_lockin_amp__GUI            as lockin_GUI
import DvG_dev_Arduino_lockin_amp__fun_serial as lockin_functions
import DvG_dev_Arduino_lockin_amp__pyqt_lib   as lockin_pyqt_lib

# Show debug info in terminal? Warning: Slow! Do not leave on unintentionally.
DEBUG = False

# Enable GPU-accelerated computations on an NVIDIA videocard with CUDA support?
# Will handle fftconvolve (FIR filters).
USE_CUDA = False

# SNIPPET: To quickly enable FFTW running inside numpy and (partially) scipy
# Used for debugging only
"""
import pyfftw
np.fft = pyfftw.interfaces.numpy_fft  # Monkey patch fftpack
pyfftw.interfaces.cache.enable()      # Turn on cache for optimum performance
"""

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
        # Prevent possible concurrent pyqtgraph.PlotWidget() redraws and GUI
        # events when doing heavy calculations to unburden the CPU and prevent
        # dropped buffers. Dropped graphing frames are prefereable to dropped
        # data buffers.
        for gw in window.all_charts:
            gw.setUpdatesEnabled(False)
    
    # Listen for data buffers send by the lock-in
    [success,
     state.time,
     state.ref_X,
     state.ref_Y,
     state.sig_I] = lockin.listen_to_lockin_amp()
    
    if not(success):
        dprint("@ %s %s" % current_date_time_strings())
        return False
    
    if window.boost_fps_graphing:
        # Prevent possible concurrent pyqtgraph.PlotWidget() redraws and GUI
        # events when doing heavy calculations to unburden the CPU and prevent
        # dropped buffers. Dropped graphing frames are prefereable to dropped
        # data buffers.
        for gw in window.all_charts:
            gw.setUpdatesEnabled(False)
    
    # HACK: hard-coded calibration correction on the ADC
    # TODO: make a self-calibration procedure and store correction results
    # on non-volatile memory of the microprocessor.
    dev_sig_I = state.sig_I * 0.0054 + 0.0020;
    state.sig_I -= dev_sig_I
    
    # Detect dropped samples / buffers
    lockin_pyqt.state.buffers_received += 1

    prev_last_deque_time = (state.deque_time[-1] if state.buffers_received > 1
                            else np.nan)
    dT = (state.time[0] - prev_last_deque_time) / 1e6 # Transform [usec] to [sec]
    if dT > (c.ISR_CLOCK)*1.10: # Allow a few percent clock jitter
        N_dropped_samples = int(round(dT / c.ISR_CLOCK) - 1)
        dprint("Dropped samples: %i" % N_dropped_samples)
        dprint("@ %s %s" % current_date_time_strings())
        
        # Replace dropped samples with np.nan samples.
        # As a result, the filter output will contain a continuous series of
        # np.nan values in the output for up to DvG_Buffered_FIR_Filter.
        # Buffered_FIR_Filter().T_settle_deque seconds long after the occurrence
        # of the last dropped sample.
        state.deque_time .extend(prev_last_deque_time + 
                np.arange(1, N_dropped_samples + 1) * c.ISR_CLOCK * 1e6)
        state.deque_ref_X.extend(np.full(N_dropped_samples, np.nan))
        state.deque_ref_Y.extend(np.full(N_dropped_samples, np.nan))
        state.deque_sig_I.extend(np.full(N_dropped_samples, np.nan))
    
    
    # Stage 0
    # -------
    
    
    state.sig_I_min = np.min (state.sig_I)
    state.sig_I_max = np.max (state.sig_I)
    state.sig_I_avg = np.mean(state.sig_I)
    state.sig_I_std = np.std (state.sig_I)

    state.deque_time .extend(state.time)
    state.deque_ref_X.extend(state.ref_X)
    state.deque_ref_Y.extend(state.ref_Y)
    state.deque_sig_I.extend(state.sig_I)

    window.CH_ref_X.add_new_readings(state.time, state.ref_X)
    window.CH_ref_Y.add_new_readings(state.time, state.ref_Y)
    window.CH_sig_I.add_new_readings(state.time, state.sig_I)


    # Stage 1
    # -------

    
    # Apply filter 1 to sig_I
    state.filt_I = lockin_pyqt.firf_1_sig_I.process(state.deque_sig_I)
    
    if lockin_pyqt.firf_1_sig_I.has_deque_settled:
        # Retrieve the block of original data from the past that aligns with
        # the current filter output
        valid_slice = slice(lockin_pyqt.firf_1_sig_I.win_idx_valid_start,
                            lockin_pyqt.firf_1_sig_I.win_idx_valid_end)
        
        state.time_1 = state.deque_time [valid_slice]        
        old_sig_I    = state.deque_sig_I[valid_slice]        
        old_ref_X    = state.deque_ref_X[valid_slice]
        old_ref_Y    = state.deque_ref_Y[valid_slice]
        
        # Heterodyne mixing       
        # Equivalent to:
        #   mix_X = (old_ref_X - c.ref_V_offset) * filt_I  # SLOW code
        #   mix_Y = (old_ref_Y - c.ref_V_offset) * filt_I  # SLOW code
        np.subtract(old_ref_X, c.ref_V_offset, out=old_ref_X)
        np.subtract(old_ref_Y, c.ref_V_offset, out=old_ref_Y)
        np.multiply(old_ref_X, state.filt_I, out=state.mix_X)
        np.multiply(old_ref_Y, state.filt_I, out=state.mix_Y)
    else:
        state.time_1 = np.full(c.BUFFER_SIZE, np.nan)
        old_sig_I    = np.full(c.BUFFER_SIZE, np.nan)
        state.mix_X  = np.full(c.BUFFER_SIZE, np.nan)
        state.mix_Y  = np.full(c.BUFFER_SIZE, np.nan)
        
    state.deque_time_1.extend(state.time_1)
    state.deque_filt_I.extend(state.filt_I)
    state.deque_mix_X .extend(state.mix_X)
    state.deque_mix_Y .extend(state.mix_Y)
    
    window.CH_filt_1_in .add_new_readings(state.time_1, old_sig_I)
    window.CH_filt_1_out.add_new_readings(state.time_1, state.filt_I)
    window.CH_mix_X     .add_new_readings(state.time_1, state.mix_X)
    window.CH_mix_Y     .add_new_readings(state.time_1, state.mix_Y)
    
    
    # Stage 2
    # -------
    
    
    # Apply filter 2 to the mixer output
    state.X = lockin_pyqt.firf_2_mix_X.process(state.deque_mix_X)
    state.Y = lockin_pyqt.firf_2_mix_Y.process(state.deque_mix_Y)
    
    if lockin_pyqt.firf_2_mix_X.has_deque_settled:            
        # Retrieve the block of time data from the past that aligns with
        # the current filter output
        state.time_2 = state.deque_time_1[
                            lockin_pyqt.firf_1_sig_I.win_idx_valid_start:
                            lockin_pyqt.firf_1_sig_I.win_idx_valid_end]
        
        # Signal amplitude and phase reconstruction
        np.sqrt(state.X**2 + state.Y**2, out=state.R)
        
        # NOTE: Because 'mix_X' and 'mix_Y' are both of type 'numpy.array', a
        # division by (mix_X = 0) is handled correctly due to 'numpy.inf'.
        # Likewise, 'numpy.arctan(numpy.inf)' will result in pi/2. We suppress
        # the RuntimeWarning: divide by zero encountered in true_divide.
        np.seterr(divide='ignore')
        np.divide(state.Y, state.X, out=state.T)
        np.arctan(state.T, out=state.T)
        np.multiply(state.T, 180/np.pi, out=state.T) # Transform [rad] to [deg]
        np.seterr(divide='warn')
    else:
        state.time_2 = np.full(c.BUFFER_SIZE, np.nan)
        state.R      = np.full(c.BUFFER_SIZE, np.nan)
        state.T      = np.full(c.BUFFER_SIZE, np.nan)
        
    state.deque_time_2.extend(state.time_2)
    state.deque_X.extend(state.X)
    state.deque_Y.extend(state.Y)
    state.deque_R.extend(state.R)
    state.deque_T.extend(state.T)
    
    if window.qrbt_XR_X.isChecked():
        window.CH_LIA_XR.add_new_readings(state.time_2, state.X)
    else:
        window.CH_LIA_XR.add_new_readings(state.time_2, state.R)
    if window.qrbt_YT_Y.isChecked():
        window.CH_LIA_YT.add_new_readings(state.time_2, state.Y)
    else:
        window.CH_LIA_YT.add_new_readings(state.time_2, state.T)
    
    
    # Check if memory address of underlying buffer is still unchanged
    """
    test = np.asarray(state.deque_X)
    print("%6i, mem: %i, cont?: %i, rb buf mem: %i, full? %i" % (
            state.buffers_received,
            test.__array_interface__['data'][0],
            test.flags['C_CONTIGUOUS'],
            state.deque_X._unwrap_buffer.__array_interface__['data'][0],
            state.deque_X.is_full))
    """
    

    # Power spectra
    # -------------

    if (window.legend_box_PS.chkbs[0].isChecked() and
        state.deque_sig_I.is_full):
        window.BP_PS_1.set_data(
                lockin_pyqt.fftw_PS_sig_I.freqs,
                lockin_pyqt.fftw_PS_sig_I.process_dB(state.deque_sig_I))
            
    if (window.legend_box_PS.chkbs[1].isChecked() and
        state.deque_filt_I.is_full):
        window.BP_PS_2.set_data(
                lockin_pyqt.fftw_PS_filt_I.freqs,
                lockin_pyqt.fftw_PS_filt_I.process_dB(state.deque_filt_I))
        
    if (window.legend_box_PS.chkbs[2].isChecked() and
        state.deque_mix_X.is_full):
        window.BP_PS_3.set_data(
                lockin_pyqt.fftw_PS_mix_X.freqs,
                lockin_pyqt.fftw_PS_mix_X.process_dB(state.deque_mix_X))
        
    if (window.legend_box_PS.chkbs[3].isChecked() and
        state.deque_mix_Y.is_full):
        window.BP_PS_4.set_data(
                lockin_pyqt.fftw_PS_mix_Y.freqs,
                lockin_pyqt.fftw_PS_mix_Y.process_dB(state.deque_mix_Y))
        
    if (window.legend_box_PS.chkbs[4].isChecked() and
        state.deque_R.is_full):
        window.BP_PS_5.set_data(
                lockin_pyqt.fftw_PS_R.freqs, 
                lockin_pyqt.fftw_PS_R.process_dB(state.deque_R))

    
    # Logging to file
    #----------------


    if file_logger.starting:
        fn_log = QDateTime.currentDateTime().toString("yyMMdd_HHmmss") + ".txt"
        if file_logger.create_log(state.time, fn_log, mode='w'):
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
        if lockin_pyqt.firf_2_mix_X.has_deque_settled: # All lights green!
            idx_offset = lockin_pyqt.firf_1_sig_I.win_idx_valid_start
            
            for i in range(c.BUFFER_SIZE):
                data = (("%i\t" +
                         "%.5f\t" * 9 +
                         "%.4f\n") % (
                         #"%.4f\t%i\t%i\n") % (
                        state.deque_time  [i],
                        state.deque_ref_X [i],
                        state.deque_ref_Y [i],
                        state.deque_sig_I [i],
                        state.deque_filt_I[i + idx_offset],
                        state.deque_mix_X [i + idx_offset],
                        state.deque_mix_Y [i + idx_offset],
                        state.X[i],
                        state.Y[i],
                        state.R[i],
                        state.T[i]#,
                        #state.deque_time_1[i + idx_offset],
                        #state.time_2[i]
                        ))
                file_logger.write(data)
            #file_logger.write("%i\t%.4f\t%.4f\t%.4f\n" % 
            #                  (time[i], ref_X[i], ref_Y[i], sig_I[i]))
    
    # Re-enable pyqtgraph.PlotWidget() redraws and GUI events
    for gw in window.all_charts:
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
        if os.name == "nt": proc.nice(psutil.HIGH_PRIORITY_CLASS) # Windows
        else: proc.nice(-20)                                      # Other
    except:
        print("Warning: Could not set process to high priority.\n")

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
                            DEBUG_worker_send=False,
                            use_CUDA=USE_CUDA)
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
    #   Create power spectrum FFTW objects as members of lockin_pyqt
    # --------------------------------------------------------------------------

    p = {'len_data': lockin_pyqt.state.N_deque, 'fs': lockin.config.Fs,
         'nperseg': lockin.config.Fs}
    lockin_pyqt.fftw_PS_sig_I  = FFTW_WelchPowerSpectrum(**p)
    lockin_pyqt.fftw_PS_filt_I = FFTW_WelchPowerSpectrum(**p)
    lockin_pyqt.fftw_PS_mix_X  = FFTW_WelchPowerSpectrum(**p)
    lockin_pyqt.fftw_PS_mix_Y  = FFTW_WelchPowerSpectrum(**p)
    lockin_pyqt.fftw_PS_R      = FFTW_WelchPowerSpectrum(**p)

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