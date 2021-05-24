#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dennis van Gils
24-05-2021
"""
# pylint: disable=invalid-name

import time as Time

import numpy as np

from dvg_ringbuffer import RingBuffer
from dvg_ringbuffer_fir_filter import (
    RingBuffer_FIR_Filter,
    RingBuffer_FIR_Filter_Config,
)
from dvg_fftw_welchpowerspectrum import FFTW_WelchPowerSpectrum

# Main parameters to test for
TEST_POWERSPECTRA = True
BLOCK_SIZE = 2000
N_BLOCKS = 21
Fs = 20000  # [Hz]

# Simulation vars
T_total = 120  # [s]
ref_freq_Hz = 300  # [Hz]
ref_V_offset = 1.5  # [V]
sig_I_phase = 10  # [deg]
sig_I_noise_ampl = 0.04


class State:
    def __init__(self, block_size, N_blocks):
        """Reflects the actual readings, parsed into separate variables, of
        the lock-in amplifier. There should only be one instance of the
        State class.
        """
        # fmt: off
        self.block_size  = block_size
        self.N_blocks    = N_blocks
        self.rb_capacity = block_size * N_blocks
        self.buffers_received = 0

        # Predefine arrays for clarity
        # Keep .time as dtype=np.float64, because it can contain np.nan
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

        """ Ring buffers needed for proper FIR filtering.
        Each time a complete block of `block_size` samples is received from
        the lock-in, it will extend the ring buffer array (FIFO).

            i.e. N_blocks = 3
                startup         : ringbuffer = [nan     ; nan     ; nan    ]
                received block 1: ringbuffer = [block_1 ; nan     ; nan    ]
                received block 2: ringbuffer = [block_1 ; block_2 ; nan    ]
                received block 3: ringbuffer = [block_1 ; block_2 ; block_3]
                received block 4: ringbuffer = [block_2 ; block_3 ; block_4]
                received block 5: ringbuffer = [block_3 ; block_4 ; block_5]
                etc...
        """

        # Create ring buffers
        p = {'capacity': self.rb_capacity, 'dtype': np.float64}

        # Stage 0: unprocessed data
        self.rb_time   = RingBuffer(**p)
        self.rb_ref_X  = RingBuffer(**p)
        self.rb_ref_Y  = RingBuffer(**p)
        self.rb_sig_I  = RingBuffer(**p)

        # Stage 1: apply AC-coupling and band-stop filter and heterodyne mixing
        self.rb_time_1 = RingBuffer(**p)
        self.rb_filt_I = RingBuffer(**p)
        self.rb_mix_X  = RingBuffer(**p)
        self.rb_mix_Y  = RingBuffer(**p)

        # Stage 2: apply low-pass filter and signal reconstruction
        self.rb_time_2 = RingBuffer(**p)
        self.rb_X      = RingBuffer(**p)
        self.rb_Y      = RingBuffer(**p)
        self.rb_R      = RingBuffer(**p)
        self.rb_T      = RingBuffer(**p)
        # fmt: on


# ------------------------------------------------------------------------------
#   main
# ------------------------------------------------------------------------------

if __name__ == "__main__":

    state = State(BLOCK_SIZE, N_BLOCKS)

    #  Create FIR filters
    # --------------------

    # AC-coupling & band-stop filter on sig_I
    firf_1_config = RingBuffer_FIR_Filter_Config(
        Fs=Fs,
        block_size=BLOCK_SIZE,
        N_blocks=N_BLOCKS,
        firwin_cutoff=[2.0, 48.0, 52.0],
        firwin_window="blackmanharris",
        firwin_pass_zero=False,
    )
    firf_1_sig_I = RingBuffer_FIR_Filter(
        config=firf_1_config, name="firf_1_sig_I"
    )

    # Low-pass filter on mix_X and mix_Y
    roll_off_width = 5  # [Hz]
    firf_2_config = RingBuffer_FIR_Filter_Config(
        Fs=Fs,
        block_size=BLOCK_SIZE,
        N_blocks=N_BLOCKS,
        firwin_cutoff=2 * ref_freq_Hz - roll_off_width,
        firwin_window="blackmanharris",
        firwin_pass_zero=True,
    )
    firf_2_mix_X = RingBuffer_FIR_Filter(
        config=firf_2_config, name="firf_2_mix_X"
    )
    firf_2_mix_Y = RingBuffer_FIR_Filter(
        config=firf_2_config, name="firf_2_mix_Y"
    )

    #  Create power spectrum FFTW objects
    # ------------------------------------

    if TEST_POWERSPECTRA:
        # fmt: off
        p = {"len_data": BLOCK_SIZE * N_BLOCKS, "fs": Fs, "nperseg": Fs}
        fftw_PS_sig_I  = FFTW_WelchPowerSpectrum(**p)
        fftw_PS_filt_I = FFTW_WelchPowerSpectrum(**p)
        fftw_PS_mix_X  = FFTW_WelchPowerSpectrum(**p)
        fftw_PS_mix_Y  = FFTW_WelchPowerSpectrum(**p)
        fftw_PS_R      = FFTW_WelchPowerSpectrum(**p)
        # fmt: on

    #  Generate artificial time series ahead of time
    # -----------------------------------------------

    time = np.linspace(0, T_total, T_total * Fs, endpoint=False)  # [s]
    ref_X = ref_V_offset + np.cos(2 * np.pi * ref_freq_Hz * time)
    ref_Y = ref_V_offset + np.sin(2 * np.pi * ref_freq_Hz * time)

    np.random.seed(0)
    sig_I_noise = sig_I_noise_ampl * np.random.randn(len(time))
    sig_I = ref_V_offset + np.cos(
        2 * np.pi * ref_freq_Hz * time - sig_I_phase / 180 * np.pi
    )
    np.add(sig_I, sig_I_noise, out=sig_I)

    #  Simulate incoming blocks on the fly
    # -------------------------------------

    tick = Time.perf_counter()
    N_sim_blocks = int(len(time) / BLOCK_SIZE)
    for idx_sim_block in range(N_sim_blocks):
        sim_slice = slice(
            BLOCK_SIZE * idx_sim_block, BLOCK_SIZE * (idx_sim_block + 1)
        )

        # Stage 0
        # -------

        state.time = time[sim_slice]
        state.ref_X = ref_X[sim_slice]
        state.ref_Y = ref_Y[sim_slice]
        state.sig_I = sig_I[sim_slice]

        state.sig_I_min = np.min(state.sig_I)
        state.sig_I_max = np.max(state.sig_I)
        state.sig_I_avg = np.mean(state.sig_I)
        state.sig_I_std = np.std(state.sig_I)

        state.rb_time.extend(state.time)
        state.rb_ref_X.extend(state.ref_X)
        state.rb_ref_Y.extend(state.ref_Y)
        state.rb_sig_I.extend(state.sig_I)

        # Stage 1
        # -------
        # fmt: off

        # Apply filter 1 to sig_I
        state.filt_I = firf_1_sig_I.apply_filter(state.rb_sig_I)

        if firf_1_sig_I.filter_has_settled:
            # Retrieve the block of original data from the past that aligns with
            # the current filter output
            valid_slice = firf_1_sig_I.rb_valid_slice

            state.time_1 = state.rb_time [valid_slice]
            old_sig_I    = state.rb_sig_I[valid_slice]
            old_ref_X    = state.rb_ref_X[valid_slice]
            old_ref_Y    = state.rb_ref_Y[valid_slice]

            # Heterodyne mixing
            # Equivalent to:
            #   mix_X = (old_ref_X - c.ref_V_offset) * filt_I  # SLOW code
            #   mix_Y = (old_ref_Y - c.ref_V_offset) * filt_I  # SLOW code
            np.subtract(old_ref_X, ref_V_offset, out=old_ref_X)
            np.subtract(old_ref_Y, ref_V_offset, out=old_ref_Y)
            np.multiply(old_ref_X, state.filt_I, out=state.mix_X)
            np.multiply(old_ref_Y, state.filt_I, out=state.mix_Y)
        else:
            state.time_1 = np.full(BLOCK_SIZE, np.nan)
            old_sig_I    = np.full(BLOCK_SIZE, np.nan)
            state.mix_X  = np.full(BLOCK_SIZE, np.nan)
            state.mix_Y  = np.full(BLOCK_SIZE, np.nan)

        state.filt_I_min = np.min(state.filt_I)
        state.filt_I_max = np.max(state.filt_I)
        state.filt_I_avg = np.mean(state.filt_I)
        state.filt_I_std = np.std(state.filt_I)

        state.rb_time_1.extend(state.time_1)
        state.rb_filt_I.extend(state.filt_I)
        state.rb_mix_X .extend(state.mix_X)
        state.rb_mix_Y .extend(state.mix_Y)
        # fmt: on

        # Stage 2
        # -------

        # Apply filter 2 to the mixer output
        state.X = firf_2_mix_X.apply_filter(state.rb_mix_X)
        state.Y = firf_2_mix_Y.apply_filter(state.rb_mix_Y)

        if firf_2_mix_X.filter_has_settled:
            # Retrieve the block of time data from the past that aligns with
            # the current filter output
            valid_slice = firf_1_sig_I.rb_valid_slice
            state.time_2 = state.rb_time_1[valid_slice]

            # Signal amplitude and phase reconstruction
            np.sqrt(state.X ** 2 + state.Y ** 2, out=state.R)

            # NOTE: Because `mix_X` and `mix_Y` are both of type `numpy.ndarray`, a
            # division by (mix_X = 0) is handled correctly due to `numpy.inf`.
            # Likewise, `numpy.arctan(numpy.inf)`` will result in pi/2. We suppress
            # the RuntimeWarning: divide by zero encountered in true_divide.
            np.seterr(divide="ignore")
            np.divide(state.Y, state.X, out=state.T)
            np.arctan(state.T, out=state.T)
            np.multiply(state.T, 180 / np.pi, out=state.T)  # [rad] to [deg]
            np.seterr(divide="warn")
        else:
            state.time_2 = np.full(BLOCK_SIZE, np.nan)
            state.R = np.full(BLOCK_SIZE, np.nan)
            state.T = np.full(BLOCK_SIZE, np.nan)

        state.X_avg = np.mean(state.X)
        state.Y_avg = np.mean(state.Y)
        state.R_avg = np.mean(state.R)
        state.T_avg = np.mean(state.T)

        state.rb_time_2.extend(state.time_2)
        state.rb_X.extend(state.X)
        state.rb_Y.extend(state.Y)
        state.rb_R.extend(state.R)
        state.rb_T.extend(state.T)

        # Power spectra
        # -------------
        if TEST_POWERSPECTRA:
            if state.rb_sig_I.is_full:
                fftw_PS_sig_I.process_dB(state.rb_sig_I)

            if state.rb_filt_I.is_full:
                fftw_PS_filt_I.process_dB(state.rb_filt_I)

            if state.rb_mix_X.is_full:
                fftw_PS_mix_X.process_dB(state.rb_mix_X)

            if state.rb_mix_Y.is_full:
                fftw_PS_mix_Y.process_dB(state.rb_mix_Y)

            if state.rb_R.is_full:
                fftw_PS_R.process_dB(state.rb_R)

    print("%5.3f %5.3f" % (state.sig_I_avg, state.filt_I_avg))
    print("%5.3f %5.3f" % (state.R_avg, state.T_avg))

    tock = Time.perf_counter()
    print("Number of blocks simulated: %i" % N_sim_blocks)
    print("Avg time per block: %.1f ms" % ((tock - tick) / N_sim_blocks * 1000))

