#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Arduino lock-in amplifier
"""
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/DvG_Arduino_lock-in_amp"
__date__ = "13-05-2021"
__version__ = "2.0.0"
# pylint: disable=invalid-name

import os
import sys

import time as Time
import psutil

from PyQt5 import QtCore
from PyQt5 import QtWidgets as QtWid
from PyQt5.QtCore import QDateTime
import numpy as np

from dvg_pyqt_filelogger import FileLogger
from dvg_debug_functions import dprint  # , print_fancy_traceback as pft
from DvG_FFTW_WelchPowerSpectrum import FFTW_WelchPowerSpectrum

from Alia_protocol_serial import Alia, Waveform
from Alia_qdev import Alia_qdev
from Alia_gui import MainWindow

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


# ------------------------------------------------------------------------------
#   current_date_time_strings
# ------------------------------------------------------------------------------


def current_date_time_strings():
    cur_date_time = QDateTime.currentDateTime()
    return (
        cur_date_time.toString("dd-MM-yyyy"),
        cur_date_time.toString("HH:mm:ss"),
    )


# ------------------------------------------------------------------------------
#   Program termination routines
# ------------------------------------------------------------------------------


def stop_running():
    app.processEvents()
    alia_qdev.turn_off()
    alia_qdev.quit()
    logger.close()


@QtCore.pyqtSlot()
def notify_connection_lost():
    stop_running()

    excl = "    ! ! ! ! ! ! ! !    "
    window.qlbl_title.setText("%sLOST CONNECTION%s" % (excl, excl))

    str_cur_date, str_cur_time = current_date_time_strings()
    str_msg = "%s %s\nLost connection to Arduino on port %s.\n" % (
        str_cur_date,
        str_cur_time,
        alia.ser.portstr,
    )
    print("\nCRITICAL ERROR @ %s" % str_msg)
    reply = QtWid.QMessageBox.warning(
        window, "CRITICAL ERROR", str_msg, QtWid.QMessageBox.Ok
    )

    if reply == QtWid.QMessageBox.Ok:
        pass  # Leave the GUI open for read-only inspection by the user


@QtCore.pyqtSlot()
def about_to_quit():
    print("\nAbout to quit")
    stop_running()
    alia.close()


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
    c: Alia.Config = alia.config
    state: Alia_qdev.State = alia_qdev.state

    # Prevent throwings errors if just paused
    if alia.lockin_paused:
        return False

    if not window.boost_fps_graphing:
        # Prevent possible concurrent pyqtgraph.PlotWidget() redraws and GUI
        # events when doing heavy calculations to unburden the CPU and prevent
        # dropped buffers. Dropped graphing frames are prefereable to dropped
        # data buffers.
        for graph in window.all_graphs:
            graph.setUpdatesEnabled(False)

    # Listen for data buffers send by the lock-in
    (
        success,
        state.time,
        state.ref_X,
        state.ref_Y,
        state.sig_I,
    ) = alia.listen_to_lockin_amp()

    if not success:
        dprint("@ %s %s" % current_date_time_strings())
        return False

    if window.boost_fps_graphing:
        # Prevent possible concurrent pyqtgraph.PlotWidget() redraws and GUI
        # events when doing heavy calculations to unburden the CPU and prevent
        # dropped buffers. Dropped graphing frames are prefereable to dropped
        # data buffers.
        for graph in window.all_graphs:
            graph.setUpdatesEnabled(False)

    # HACK: hard-coded calibration correction on the ADC
    # TODO: make a self-calibration procedure and store correction results
    # on non-volatile memory of the microprocessor.
    # dev_sig_I = state.sig_I * 0.0054 + 0.0020
    # state.sig_I += 0.01

    # Detect dropped samples / buffers
    alia_qdev.state.buffers_received += 1

    prev_last_deque_time = (
        state.deque_time[-1] if state.buffers_received > 1 else np.nan
    )
    dT = (
        state.time[0] - prev_last_deque_time
    ) / 1e6  # Transform [usec] to [sec]
    if (
        dT > (c.SAMPLING_PERIOD * 1e6) * 1.10
    ):  # Allow a few percent clock jitter
        N_dropped_samples = int(round(dT / c.SAMPLING_PERIOD) - 1)
        dprint("Dropped samples: %i" % N_dropped_samples)
        dprint("@ %s %s" % current_date_time_strings())

        # Replace dropped samples with np.nan samples.
        # As a result, the filter output will contain a continuous series of
        # np.nan values in the output for up to DvG_Buffered_FIR_Filter.
        # Buffered_FIR_Filter().T_settle_deque seconds long after the occurrence
        # of the last dropped sample.
        state.deque_time.extend(
            prev_last_deque_time
            + np.arange(1, N_dropped_samples + 1) * c.SAMPLING_PERIOD * 1e6
        )
        state.deque_ref_X.extend(np.full(N_dropped_samples, np.nan))
        state.deque_ref_Y.extend(np.full(N_dropped_samples, np.nan))
        state.deque_sig_I.extend(np.full(N_dropped_samples, np.nan))

    # Stage 0
    # -------

    state.sig_I_min = np.min(state.sig_I)
    state.sig_I_max = np.max(state.sig_I)
    state.sig_I_avg = np.mean(state.sig_I)
    state.sig_I_std = np.std(state.sig_I)

    state.deque_time.extend(state.time)
    state.deque_ref_X.extend(state.ref_X)
    state.deque_ref_Y.extend(state.ref_Y)
    state.deque_sig_I.extend(state.sig_I)

    window.hcc_ref_X.extendData(state.time, state.ref_X)
    window.hcc_ref_Y.extendData(state.time, state.ref_Y)
    window.hcc_sig_I.extendData(state.time, state.sig_I)

    # Stage 1
    # -------

    # Apply filter 1 to sig_I
    state.filt_I = alia_qdev.firf_1_sig_I.process(state.deque_sig_I)

    # fmt: off
    if alia_qdev.firf_1_sig_I.has_deque_settled:
        # Retrieve the block of original data from the past that aligns with
        # the current filter output
        valid_slice = slice(
            alia_qdev.firf_1_sig_I.win_idx_valid_start,
            alia_qdev.firf_1_sig_I.win_idx_valid_end,
        )

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
        np.multiply(old_ref_X, state.filt_I  , out=state.mix_X)
        np.multiply(old_ref_Y, state.filt_I  , out=state.mix_Y)
    else:
        state.time_1 = np.full(c.BLOCK_SIZE, np.nan)
        old_sig_I    = np.full(c.BLOCK_SIZE, np.nan)
        state.mix_X  = np.full(c.BLOCK_SIZE, np.nan)
        state.mix_Y  = np.full(c.BLOCK_SIZE, np.nan)

    state.deque_time_1.extend(state.time_1)
    state.deque_filt_I.extend(state.filt_I)
    state.deque_mix_X .extend(state.mix_X)
    state.deque_mix_Y .extend(state.mix_Y)

    window.hcc_filt_1_in .extendData(state.time_1, old_sig_I)
    window.hcc_filt_1_out.extendData(state.time_1, state.filt_I)
    window.hcc_mix_X     .extendData(state.time_1, state.mix_X)
    window.hcc_mix_Y     .extendData(state.time_1, state.mix_Y)
    # fmt: on

    # Stage 2
    # -------

    # Apply filter 2 to the mixer output
    state.X = alia_qdev.firf_2_mix_X.process(state.deque_mix_X)
    state.Y = alia_qdev.firf_2_mix_Y.process(state.deque_mix_Y)

    if alia_qdev.firf_2_mix_X.has_deque_settled:
        # Retrieve the block of time data from the past that aligns with
        # the current filter output
        # fmt: off
        state.time_2 = state.deque_time_1[
            alia_qdev.firf_1_sig_I.win_idx_valid_start :
            alia_qdev.firf_1_sig_I.win_idx_valid_end
        ]
        # fmt: on

        # Signal amplitude and phase reconstruction
        np.sqrt(state.X ** 2 + state.Y ** 2, out=state.R)

        # NOTE: Because 'mix_X' and 'mix_Y' are both of type 'numpy.array', a
        # division by (mix_X = 0) is handled correctly due to 'numpy.inf'.
        # Likewise, 'numpy.arctan(numpy.inf)' will result in pi/2. We suppress
        # the RuntimeWarning: divide by zero encountered in true_divide.
        np.seterr(divide="ignore")
        np.divide(state.Y, state.X, out=state.T)
        np.arctan(state.T, out=state.T)
        np.multiply(
            state.T, 180 / np.pi, out=state.T
        )  # Transform [rad] to [deg]
        np.seterr(divide="warn")
    else:
        state.time_2 = np.full(c.BLOCK_SIZE, np.nan)
        state.R = np.full(c.BLOCK_SIZE, np.nan)
        state.T = np.full(c.BLOCK_SIZE, np.nan)

    state.deque_time_2.extend(state.time_2)
    state.deque_X.extend(state.X)
    state.deque_Y.extend(state.Y)
    state.deque_R.extend(state.R)
    state.deque_T.extend(state.T)

    window.hcc_LIA_XR.extendData(
        state.time_2, state.X if window.qrbt_XR_X.isChecked() else state.R
    )
    window.hcc_LIA_YT.extendData(
        state.time_2, state.Y if window.qrbt_YT_Y.isChecked() else state.T
    )

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

    calculate_PS_sig_I()
    calculate_PS_filt_I()
    calculate_PS_mix_X()
    calculate_PS_mix_Y()
    calculate_PS_R()

    # Logging to file
    logger.update(mode="w")

    # Re-enable pyqtgraph.PlotWidget() redraws and GUI events
    for graph in window.all_graphs:
        graph.setUpdatesEnabled(True)

    # Return success
    return True


# ------------------------------------------------------------------------------
#   Log functions
# ------------------------------------------------------------------------------


def write_header_to_log():
    header = (
        "time[s]\t"
        "ref_X*[V]\t"
        "ref_Y*[V]\t"
        "sig_I[V]\t"
        "filt_I[V]\t"
        "mix_X[V]\t"
        "mix_Y[V]\t"
        "X[V]\t"
        "Y[V]\t"
        "R[V]\t"
        "T[deg]\n"
    )
    logger.write(header)


def write_data_to_log():
    # Try: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html

    if alia_qdev.firf_2_mix_X.has_deque_settled:  # All lights green!
        idx_offset = alia_qdev.firf_1_sig_I.win_idx_valid_start
        N = alia.config.BLOCK_SIZE
        state = alia_qdev.state

        tick = Time.perf_counter()
        data = np.asmatrix(
            [
                state.deque_time[:N] / 1e6,
                state.deque_ref_X[:N],
                state.deque_ref_Y[:N],
                state.deque_sig_I[:N],
                state.deque_filt_I[idx_offset : idx_offset + N],
                state.deque_mix_X[idx_offset : idx_offset + N],
                state.deque_mix_Y[idx_offset : idx_offset + N],
                state.X[:N],
                state.Y[:N],
                state.R[:N],
                state.T[:N],
                # state.deque_time_1[idx_offset : idx_offset + N] / 1e6,
                # state.time_2[:N] / 1e6,
            ]
        )
        data = np.ma.transpose(data)

        tock = Time.perf_counter()
        # print("%.4f" % (tock - tick), end=", ")

        # TODO: incorporate `np.savetxt()` into `dvg_pyqt_filelogger`
        for row in data:
            np.savetxt(
                logger._filehandle, row, fmt="%.5f", delimiter="\t",
            )

        # print("%.4f" % (Time.perf_counter() - tock))


# ------------------------------------------------------------------------------
#   Main
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # Set priority of this process to maximum in the operating system
    print("PID: %s\n" % os.getpid())
    try:
        proc = psutil.Process(os.getpid())
        if os.name == "nt":
            proc.nice(psutil.HIGH_PRIORITY_CLASS)  # Windows
        else:
            proc.nice(-20)  # Other
    except:  # pylint: disable=bare-except
        print("Warning: Could not set process to high priority.\n")

    # --------------------------------------------------------------------------
    #   Arduino
    # --------------------------------------------------------------------------

    # Connect to Arduino
    alia = Alia(read_timeout=4)
    alia.auto_connect()

    if not alia.is_alive:
        print("\nCheck connection and try resetting the Arduino.")
        print("Exiting...\n")
        sys.exit(0)

    # alia.begin()
    alia.begin(
        ref_freq=250,
        ref_V_offset=2.0,
        ref_V_ampl=1.0,
        ref_waveform=Waveform.Cosine,
    )

    # Create workers and threads
    alia_qdev = Alia_qdev(
        dev=alia,
        DAQ_function=lockin_DAQ_update,
        critical_not_alive_count=3,
        N_buffers_in_deque=21,
        use_CUDA=USE_CUDA,
        debug=DEBUG,
    )
    alia_qdev.signal_connection_lost.connect(notify_connection_lost)

    # --------------------------------------------------------------------------
    #   Create application and main window
    # --------------------------------------------------------------------------
    QtCore.QThread.currentThread().setObjectName("MAIN")  # For DEBUG info

    app = 0  # Work-around for kernel crash when using Spyder IDE
    app = QtWid.QApplication(sys.argv)
    app.aboutToQuit.connect(about_to_quit)

    window = MainWindow(alia, alia_qdev)

    # --------------------------------------------------------------------------
    #   File logger
    # --------------------------------------------------------------------------

    logger = FileLogger(
        write_header_function=write_header_to_log,
        write_data_function=write_data_to_log,
    )
    logger.signal_recording_started.connect(
        lambda filepath: window.qpbt_record.setText(
            "Recording to file: %s" % filepath
        )
    )
    logger.signal_recording_stopped.connect(
        lambda: window.qpbt_record.setText("Click to start recording to file")
    )

    window.qpbt_record.clicked.connect(
        lambda state: logger.record(state)  # pylint: disable=unnecessary-lambda
    )

    # --------------------------------------------------------------------------
    #   Create power spectrum FFTW objects
    # --------------------------------------------------------------------------

    p = {
        "len_data": alia_qdev.state.N_deque,
        "fs": alia.config.Fs,
        "nperseg": alia.config.Fs,
    }
    # fmt: off
    alia_qdev.fftw_PS_sig_I  = FFTW_WelchPowerSpectrum(**p)
    alia_qdev.fftw_PS_filt_I = FFTW_WelchPowerSpectrum(**p)
    alia_qdev.fftw_PS_mix_X  = FFTW_WelchPowerSpectrum(**p)
    alia_qdev.fftw_PS_mix_Y  = FFTW_WelchPowerSpectrum(**p)
    alia_qdev.fftw_PS_R      = FFTW_WelchPowerSpectrum(**p)
    # fmt: on

    # Only calculate the power spectrum when the curve is visible. Calculating
    # spectra is CPU intensive and might impact the responsiveness of the GUI
    # or, in the extreme case, cause dropped buffers.

    def calculate_PS_sig_I():
        state = alia_qdev.state
        if window.pc_PS_sig_I.isVisible() and state.deque_sig_I.is_full:
            window.pc_PS_sig_I.setData(
                alia_qdev.fftw_PS_sig_I.freqs,
                alia_qdev.fftw_PS_sig_I.process_dB(state.deque_sig_I),
            )

    def calculate_PS_filt_I():
        state = alia_qdev.state
        if window.pc_PS_filt_I.isVisible() and state.deque_filt_I.is_full:
            window.pc_PS_filt_I.setData(
                alia_qdev.fftw_PS_filt_I.freqs,
                alia_qdev.fftw_PS_filt_I.process_dB(state.deque_filt_I),
            )

    def calculate_PS_mix_X():
        state = alia_qdev.state
        if window.pc_PS_mix_X.isVisible() and state.deque_mix_X.is_full:
            window.pc_PS_mix_X.setData(
                alia_qdev.fftw_PS_mix_X.freqs,
                alia_qdev.fftw_PS_mix_X.process_dB(state.deque_mix_X),
            )

    def calculate_PS_mix_Y():
        state = alia_qdev.state
        if window.pc_PS_mix_Y.isVisible() and state.deque_mix_Y.is_full:
            window.pc_PS_mix_Y.setData(
                alia_qdev.fftw_PS_mix_Y.freqs,
                alia_qdev.fftw_PS_mix_Y.process_dB(state.deque_mix_Y),
            )

    def calculate_PS_R():
        state = alia_qdev.state
        if window.pc_PS_R.isVisible() and state.deque_R.is_full:
            window.pc_PS_R.setData(
                alia_qdev.fftw_PS_R.freqs,
                alia_qdev.fftw_PS_R.process_dB(state.deque_R),
            )

    # Special case where the lock-in is paused: Clicking the legend checkboxes
    # to unhide the PS curves should recalculate the PS based on the last known
    # data. We must check if the lock-in is paused before calculating, because
    # we might otherwise interfere with the other possible PS calculation
    # already happening in the worker_DAQ thread if the lock-in is actually
    # running at the moment of toggling the checkboxes.
    #
    # Ugly workaround, I know. All because we want the PS to be calculated only
    # when the curve will be shown in order to reduce the cpu load.

    @QtCore.pyqtSlot()
    def update_paused_PS_sig_I():
        if alia.lockin_paused:
            calculate_PS_sig_I()
            window.pc_PS_sig_I.update()

    @QtCore.pyqtSlot()
    def update_paused_PS_filt_I():
        if alia.lockin_paused:
            calculate_PS_filt_I()
            window.pc_PS_filt_I.update()

    @QtCore.pyqtSlot()
    def update_paused_PS_mix_X():
        if alia.lockin_paused:
            calculate_PS_mix_X()
            window.pc_PS_mix_X.update()

    @QtCore.pyqtSlot()
    def update_paused_PS_mix_Y():
        if alia.lockin_paused:
            calculate_PS_mix_Y()
            window.pc_PS_mix_Y.update()

    @QtCore.pyqtSlot()
    def update_paused_PS_R():
        if alia.lockin_paused:
            calculate_PS_R()
            window.pc_PS_R.update()

    window.legend_PS.chkbs[0].clicked.connect(update_paused_PS_sig_I)
    window.legend_PS.chkbs[1].clicked.connect(update_paused_PS_filt_I)
    window.legend_PS.chkbs[2].clicked.connect(update_paused_PS_mix_X)
    window.legend_PS.chkbs[3].clicked.connect(update_paused_PS_mix_Y)
    window.legend_PS.chkbs[4].clicked.connect(update_paused_PS_R)

    # --------------------------------------------------------------------------
    #   Start threads
    # --------------------------------------------------------------------------

    alia_qdev.start(DAQ_priority=QtCore.QThread.TimeCriticalPriority)

    # --------------------------------------------------------------------------
    #   Start the main GUI event loop
    # --------------------------------------------------------------------------

    window.show()
    sys.exit(app.exec_())
