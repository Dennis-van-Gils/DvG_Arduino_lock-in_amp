#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PyQt5 module to provide multithreaded communication and periodical data
acquisition with an Arduino(-like) microcontroller board that is flashed with
specific firmware to turn it into a lock-in amplifier.
"""
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/DvG_dev_Arduino"
__date__ = "02-06-2021"
__version__ = "2.0.0"
# pylint: disable=invalid-name, missing-function-docstring

# Quick and dirty toggle to either allow:
# 1) SAMD21, Arduino M0 Zero Pro, with firmware v2.0 Microchip Studio
# 2) SAMD51, Adafruit Feather M4 Express, with outdated firmware
SAMD51 = True

import numpy as np
from PyQt5 import QtCore

from dvg_qdeviceio import QDeviceIO, DAQ_TRIGGER
from dvg_ringbuffer import RingBuffer
from dvg_ringbuffer_fir_filter import (
    RingBuffer_FIR_Filter,
    RingBuffer_FIR_Filter_Config,
)

from Alia_protocol_serial import Alia

# ------------------------------------------------------------------------------
#   Alia_qdev
# ------------------------------------------------------------------------------


class Alia_qdev(QDeviceIO):
    """Manages multithreaded communication and periodical data acquisition with
    an Arduino(-like) lock-in amplifier device.

    All device I/O operations will be offloaded to 'workers', each running in
    a newly created thread.

    (*): See 'dvg_qdeviceio.QDeviceIO()' for details.

    Args:
        (*) dev:
            Reference to an 'Alia_serial_protocol.Alia()' instance. I.e. the
            serial communication layer with the Arduino lock-in amplifier.

        (*) DAQ_function
        (*) critical_not_alive_count

        N_blocks (int):
            Number of blocks to make up a full ring buffer. A block is defined
            as the number of samples per quantity that are send in bursts by the
            lock-in amplifier over the serial port, i.e. `block_size`.
            `block_size` is determined by and received from the microcontroller
            board running the lock-in amplifier firmware.

        use_CUDA (bool):
            See the header description in module `dvg_ringbuffer_fir_filter.py`.

        (*) debug:
            Show debug info in terminal? Warning: Slow! Do not leave on
            unintentionally.

    Attributes:
        Many...

    Signals:
        (*) signal_DAQ_updated()
        (*) signal_connection_lost()
        signal_ref_freq_is_set()
        signal_ref_V_offset_is_set()
        signal_ref_V_ampl_is_set()
    """

    signal_ref_freq_is_set = QtCore.pyqtSignal()
    signal_ref_V_offset_is_set = QtCore.pyqtSignal()
    signal_ref_V_ampl_is_set = QtCore.pyqtSignal()

    class State:
        def __init__(self, block_size: int, N_blocks: int):
            """Reflects the actual readings, parsed into separate variables, of
            the lock-in amplifier. There should only be one instance of the
            State class.
            """

            # fmt: off
            self.block_size  = block_size
            self.N_blocks    = N_blocks
            self.rb_capacity = block_size * N_blocks
            self.blocks_received = 0

            # Arrays to hold the block data coming from the lock-in amplifier
            # Keep `time` as `dtype=np.float64`, because it can contain `np.nan`
            self.time   = np.full(block_size, np.nan, dtype=np.float64) # [ms]
            self.ref_X  = np.full(block_size, np.nan, dtype=np.float64)
            self.ref_Y  = np.full(block_size, np.nan, dtype=np.float64)
            self.sig_I  = np.full(block_size, np.nan, dtype=np.float64)

            self.time_1 = np.full(block_size, np.nan, dtype=np.float64) # [ms]
            self.filt_I = np.full(block_size, np.nan, dtype=np.float64)
            self.mix_X  = np.full(block_size, np.nan, dtype=np.float64)
            self.mix_Y  = np.full(block_size, np.nan, dtype=np.float64)

            self.time_2 = np.full(block_size, np.nan, dtype=np.float64) # [ms]
            self.X      = np.full(block_size, np.nan, dtype=np.float64)
            self.Y      = np.full(block_size, np.nan, dtype=np.float64)
            self.R      = np.full(block_size, np.nan, dtype=np.float64)
            self.T      = np.full(block_size, np.nan, dtype=np.float64)
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

            # Ring buffers (rb) for performing FIR filtering and power spectra
            # fmt: off
            """
            Each time a complete block of `block_size` samples is received from
            the lock-in, it will extend the ring buffer array (FIFO shift
            buffer) with that block.

                i.e. N_blocks = 3
                    startup         : rb = [no value; no value ; no value]
                    received block 1: rb = [block_1 ; no value ; no value]
                    received block 2: rb = [block_1 ; block_2  ; no value]
                    received block 3: rb = [block_1 ; block_2  ; block_3]
                    received block 4: rb = [block_2 ; block_3  ; block_4]
                    received block 5: rb = [block_3 ; block_4  ; block_5]
                    etc...
            """
            p = {'capacity': self.rb_capacity, 'dtype': np.float64}

            # Stage 0: unprocessed data
            self.rb_time   = RingBuffer(**p)
            self.rb_ref_X  = RingBuffer(**p)
            self.rb_ref_Y  = RingBuffer(**p)
            self.rb_sig_I  = RingBuffer(**p)

            # Stage 1: AC-coupling and band-stop filter and heterodyne mixing
            self.rb_time_1 = RingBuffer(**p)
            self.rb_filt_I = RingBuffer(**p)
            self.rb_mix_X  = RingBuffer(**p)
            self.rb_mix_Y  = RingBuffer(**p)

            # Stage 2: low-pass filter and signal reconstruction
            self.rb_time_2 = RingBuffer(**p)
            self.rb_X      = RingBuffer(**p)
            self.rb_Y      = RingBuffer(**p)
            self.rb_R      = RingBuffer(**p)
            self.rb_T      = RingBuffer(**p)
            # fmt: on

            self.ringbuffers = [
                self.rb_time,
                self.rb_ref_X,
                self.rb_ref_Y,
                self.rb_sig_I,
                self.rb_time_1,
                self.rb_filt_I,
                self.rb_mix_X,
                self.rb_mix_Y,
                self.rb_time_2,
                self.rb_X,
                self.rb_Y,
                self.rb_R,
                self.rb_T,
            ]

            # Mutex for proper multithreading. If the state variables are not
            # atomic or thread-safe, you should lock and unlock this mutex for
            # each read and write operation.
            self.mutex = QtCore.QMutex()

        def reset(self):
            """Clear the received blocks counter and clear all ring buffers."""
            locker = QtCore.QMutexLocker(self.mutex)

            self.blocks_received = 0
            for rb in self.ringbuffers:
                rb.clear()

            locker.unlock()

    dev: Alia  # Type hint for Pylint/Pylance

    def __init__(
        self,
        dev: Alia,
        DAQ_function=None,
        critical_not_alive_count=np.nan,  # np.nan will make it go on indefinitely
        N_blocks=21,
        use_CUDA=False,
        debug=False,
        **kwargs,
    ):
        super().__init__(dev, **kwargs)  # Pass kwargs onto QtCore.QObject()

        self.state = self.State(dev.config.BLOCK_SIZE, N_blocks)

        #  Create workers
        # --------------------

        self.create_worker_DAQ(
            DAQ_trigger=DAQ_TRIGGER.CONTINUOUS,
            DAQ_function=DAQ_function,
            critical_not_alive_count=critical_not_alive_count,
            debug=debug,
        )

        self.create_worker_jobs(
            jobs_function=self.jobs_function,
            debug=debug,
        )

        #  Create FIR filters
        # --------------------

        # AC-coupling & band-stop filter on sig_I
        firf_1_config = RingBuffer_FIR_Filter_Config(
            Fs=dev.config.Fs,
            block_size=self.state.block_size,
            N_blocks=self.state.N_blocks,
            firwin_cutoff=[1, 49, 51, 99, 101, 149, 151] if SAMD51 else [2],
            firwin_window="blackmanharris",
            firwin_pass_zero=False,
            use_CUDA=use_CUDA,
        )

        self.firf_1_sig_I = RingBuffer_FIR_Filter(
            config=firf_1_config, name="firf_1_sig_I"
        )

        # Low-pass filter on mix_X and mix_Y
        # TODO: The extra distance `roll_off_width` to stay away from
        # `f_cutoff` should be calculated based on the roll-off width of the
        # filter, instead of hard-coded.
        roll_off_width = 5  # [Hz]

        firf_2_config = RingBuffer_FIR_Filter_Config(
            Fs=dev.config.Fs,
            block_size=self.state.block_size,
            N_blocks=self.state.N_blocks,
            firwin_cutoff=2 * dev.config.ref_freq - roll_off_width,
            firwin_window="blackmanharris",
            firwin_pass_zero=True,
            use_CUDA=use_CUDA,
        )

        self.firf_2_mix_X = RingBuffer_FIR_Filter(
            config=firf_2_config, name="firf_2_mix_X"
        )

        self.firf_2_mix_Y = RingBuffer_FIR_Filter(
            config=firf_2_config, name="firf_2_mix_Y"
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

            if not set_value == current_value:
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
