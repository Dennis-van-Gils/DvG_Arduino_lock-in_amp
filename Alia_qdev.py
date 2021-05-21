#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PyQt5 module to provide multithreaded communication and periodical data
acquisition for an Arduino based lock-in amplifier.
"""
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/DvG_dev_Arduino"
__date__ = "21-05-2021"
__version__ = "2.0.0"
# pylint: disable=invalid-name

import numpy as np
from PyQt5 import QtCore

from dvg_qdeviceio import QDeviceIO, DAQ_TRIGGER
from dvg_ringbuffer import RingBuffer

from Alia_protocol_serial import Alia
from DvG_Buffered_FIR_Filter import Buffered_FIR_Filter

# ------------------------------------------------------------------------------
#   Alia_qdev
# ------------------------------------------------------------------------------


class Alia_qdev(QDeviceIO):
    """Manages multithreaded communication and periodical data acquisition for
    an Arduino(-like) lock-in amplifier device.

    All device I/O operations will be offloaded to 'workers', each running in
    a newly created thread.

    (*): See 'dvg_qdeviceio.QDeviceIO()' for details.

    Args:
        (*) dev:
            Reference to an 'Alia_serial_protocol.Alia()' instance.

        (*) DAQ_function
        (*) critical_not_alive_count

        N_buffers_in_deque:

        use_CUDA:

        (*) debug:
            Show debug info in terminal? Warning: Slow! Do not leave on
            unintentionally.

    Signals:
        (*) signal_DAQ_updated()
        (*) signal_connection_lost()
        signal_ref_freq_is_set
        signal_ref_V_offset_is_set
        signal_ref_V_ampl_is_set
    """

    signal_ref_freq_is_set = QtCore.pyqtSignal()
    signal_ref_V_offset_is_set = QtCore.pyqtSignal()
    signal_ref_V_ampl_is_set = QtCore.pyqtSignal()

    class State:
        def __init__(self, buffer_size, N_buffers_in_deque=0):
            """Reflects the actual readings, parsed into separate variables, of
            the lock-in amplifier. There should only be one instance of the
            State class.
            """

            # fmt: off
            self.buffers_received   = 0
            self.buffer_size        = buffer_size           # [samples]
            self.N_buffers_in_deque = N_buffers_in_deque    # [int]
            self.N_deque = buffer_size * N_buffers_in_deque # [samples]

            # Predefine arrays for clarity
            # Keep .time as dtype=np.float64, because it can contain np.nan
            self.time   = np.full(buffer_size, np.nan, dtype=np.float64) # [ms]
            self.ref_X  = np.full(buffer_size, np.nan, dtype=np.float64)
            self.ref_Y  = np.full(buffer_size, np.nan, dtype=np.float64)
            self.sig_I  = np.full(buffer_size, np.nan, dtype=np.float64)

            self.time_1 = np.full(buffer_size, np.nan, dtype=np.float64) # [ms]
            self.filt_I = np.full(buffer_size, np.nan, dtype=np.float64)
            self.mix_X  = np.full(buffer_size, np.nan, dtype=np.float64)
            self.mix_Y  = np.full(buffer_size, np.nan, dtype=np.float64)

            self.time_2 = np.full(buffer_size, np.nan, dtype=np.float64) # [ms]
            self.X      = np.full(buffer_size, np.nan, dtype=np.float64)
            self.Y      = np.full(buffer_size, np.nan, dtype=np.float64)
            self.R      = np.full(buffer_size, np.nan, dtype=np.float64)
            self.T      = np.full(buffer_size, np.nan, dtype=np.float64)
            # fmt: on

            self.sig_I_min = np.nan
            self.sig_I_max = np.nan
            self.sig_I_avg = np.nan
            self.sig_I_std = np.nan
            self.filt_I_min = np.nan
            self.filt_I_max = np.nan
            self.filt_I_avg = np.nan
            self.filt_I_std = np.nan
            self.X_avg = np.nan
            self.Y_avg = np.nan
            self.R_avg = np.nan
            self.T_avg = np.nan

            """ Deque arrays needed for proper FIR filtering.
            Each time a complete buffer of BUFFER_SIZE samples is received from
            the lock-in, it will extend the deque array (a thread-safe FIFO
            shift buffer).

                i.e. N_buffers_in_deque = 3
                    startup          : deque = [no value; no value ; no value]
                    received buffer 1: deque = [buffer_1; no value ; no value]
                    received buffer 2: deque = [buffer_1; buffer_2 ; no value]
                    received buffer 3: deque = [buffer_1; buffer_2 ; buffer_3]
                    received buffer 4: deque = [buffer_2; buffer_3 ; buffer_4]
                    received buffer 5: deque = [buffer_3; buffer_4 ; buffer_5]
                    etc...
            """

            # Create deques
            if self.N_buffers_in_deque > 0:
                # fmt: off
                # Stage 0: unprocessed data
                p = {'capacity': self.N_deque, 'dtype': np.float64}
                self.deque_time   = RingBuffer(**p)
                self.deque_ref_X  = RingBuffer(**p)
                self.deque_ref_Y  = RingBuffer(**p)
                self.deque_sig_I  = RingBuffer(**p)
                # Stage 1: apply band-stop filter and heterodyne mixing
                self.deque_time_1 = RingBuffer(**p)
                self.deque_filt_I = RingBuffer(**p)
                self.deque_mix_X  = RingBuffer(**p)
                self.deque_mix_Y  = RingBuffer(**p)
                # Stage 2: apply low-pass filter and signal reconstruction
                self.deque_time_2 = RingBuffer(**p)
                self.deque_X      = RingBuffer(**p)
                self.deque_Y      = RingBuffer(**p)
                self.deque_R      = RingBuffer(**p)
                self.deque_T      = RingBuffer(**p)
                # fmt: on

                self.deques = [
                    self.deque_time,
                    self.deque_ref_X,
                    self.deque_ref_Y,
                    self.deque_sig_I,
                    self.deque_time_1,
                    self.deque_filt_I,
                    self.deque_mix_X,
                    self.deque_mix_Y,
                    self.deque_time_2,
                    self.deque_X,
                    self.deque_Y,
                    self.deque_R,
                    self.deque_T,
                ]

            # Mutex for proper multithreading. If the state variables are not
            # atomic or thread-safe, you should lock and unlock this mutex for
            # each read and write operation.
            self.mutex = QtCore.QMutex()

        def reset(self):
            """Clears the received buffer counter and clears all deques."""
            locker = QtCore.QMutexLocker(self.mutex)

            self.buffers_received = 0
            if self.N_buffers_in_deque > 0:
                for this_deque in self.deques:
                    this_deque.clear()

            locker.unlock()

    def __init__(
        self,
        dev: Alia,
        DAQ_function=None,
        critical_not_alive_count=np.nan,  # np.nan goes on indefinitely
        N_buffers_in_deque=21,
        use_CUDA=False,
        debug=False,
        **kwargs,
    ):
        super().__init__(dev, **kwargs)  # Pass kwargs onto QtCore.QObject()

        self.create_worker_DAQ(
            DAQ_trigger=DAQ_TRIGGER.CONTINUOUS,
            DAQ_function=DAQ_function,
            critical_not_alive_count=critical_not_alive_count,
            debug=debug,
        )

        self.create_worker_jobs(
            jobs_function=self.jobs_function, debug=debug,
        )

        self.state = self.State(dev.config.BLOCK_SIZE, N_buffers_in_deque)
        self.use_CUDA = use_CUDA

        # Create FIR filter: Band-stop on sig_I
        # TODO: turn 'use_narrower_filter' into a toggle button UI
        # fmt: off
        use_narrower_filter = False
        if use_narrower_filter:
            firwin_cutoff = [  0.5,
                              49.5,  50.5,
                              99.5, 100.5,
                             149.5, 150.5]
            firwin_window = ("chebwin", 50)
        else:
            firwin_cutoff = [2.0,]
            """firwin_cutoff = [  1.0,
                              49.0,  51.0,
                              99.0, 101.0,
                             149.0, 151.0]
            """
            firwin_window = "blackmanharris"
        # fmt: on
        self.firf_1_sig_I = Buffered_FIR_Filter(
            self.state.buffer_size,
            self.state.N_buffers_in_deque,
            dev.config.Fs,
            firwin_cutoff,
            firwin_window,
            pass_zero=False,
            display_name="firf_1_sig_I",
            use_CUDA=self.use_CUDA,
        )

        # Create FIR filter: Low-pass on mix_X and mix_Y
        # TODO: the extra distance 'roll_off_width' to stay away from
        # f_cutoff should be calculated based on the roll-off width of the
        # filter, instead of hard-coded
        roll_off_width = 5  # [Hz]
        firwin_cutoff = 2 * dev.config.ref_freq - roll_off_width
        firwin_window = "blackmanharris"
        self.firf_2_mix_X = Buffered_FIR_Filter(
            self.state.buffer_size,
            self.state.N_buffers_in_deque,
            dev.config.Fs,
            firwin_cutoff,
            firwin_window,
            pass_zero=True,
            display_name="firf_2_mix_X",
            use_CUDA=self.use_CUDA,
        )
        self.firf_2_mix_Y = Buffered_FIR_Filter(
            self.state.buffer_size,
            self.state.N_buffers_in_deque,
            dev.config.Fs,
            firwin_cutoff,
            firwin_window,
            pass_zero=True,
            display_name="firf_2_mix_Y",
            use_CUDA=self.use_CUDA,
        )

    def turn_on(self):
        self.send("turn_on")

    def turn_off(self):
        self.send("turn_off")

    def set_ref_freq(self, value: float):
        self.send("set_ref_freq", value)

    def set_ref_V_offset(self, value: float):
        self.send("set_ref_V_offset", value)

    def set_ref_V_ampl(self, value: float):
        self.send("set_ref_V_ampl", value)

    # -------------------------------------------------------------------------
    #   jobs_function
    # -------------------------------------------------------------------------

    def jobs_function(self, func, args):
        if func[:8] == "set_ref_":
            set_value = args[0]

            if func == "set_ref_freq":
                current_value = self.dev.config.ref_freq
            elif func == "set_ref_V_offset":
                current_value = self.dev.config.ref_V_offset
            elif func == "set_ref_V_ampl":
                current_value = self.dev.config.ref_V_ampl
            else:
                current_value = 0

            if not (set_value == current_value):
                was_paused = self.dev.lockin_paused

                if not was_paused:
                    self.pause_DAQ()

                if func == "set_ref_freq":
                    self.dev.set_ref(freq=set_value)
                    self.signal_ref_freq_is_set.emit()
                elif func == "set_ref_V_offset":
                    self.dev.set_ref(V_offset=set_value)
                    self.signal_ref_V_offset_is_set.emit()
                elif func == "set_ref_V_ampl":
                    self.dev.set_ref(V_ampl=set_value)
                    self.signal_ref_V_ampl_is_set.emit()

                if not was_paused:
                    self.unpause_DAQ()

        elif func == "turn_on":
            self.state.reset()
            if self.dev.turn_on(reset_timer=True):
                self.unpause_DAQ()

        elif func == "turn_off":
            self.pause_DAQ()
            self.dev.turn_off()

        else:
            # Default job handling
            func(*args)
