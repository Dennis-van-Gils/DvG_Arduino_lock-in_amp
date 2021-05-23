#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dennis van Gils
23-05-2021
"""
# pylint: disable=invalid-name

import os
import sys
import inspect
import time as Time
from collections import deque

from dvg_ringbuffer import RingBuffer
import numpy as np
import matplotlib as mpl

# mpl.use("qt5agg")
import matplotlib.pyplot as plt
import pylab

mpl.rcParams["lines.linewidth"] = 3.0
marker = "-"

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))
)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from dvg_ringbuffer_fir_filter import (
    RingBuffer_FIR_Filter,
    RingBuffer_FIR_Filter_Config,
)

f_SHOW_PLOT = True

# Original data series
Fs = 5000  # [Hz] sampling frequency
total_time = 8  # [s]
time = np.arange(0, total_time, 1 / Fs)  # [s]

sig1_ampl = 1
sig1_freq_Hz = 48.8

sig2_ampl = 1
sig2_freq_Hz = 50

sig3_ampl = 1
sig3_freq_Hz = 52

sig1 = sig1_ampl * np.sin(2 * np.pi * time * sig1_freq_Hz)
sig2 = sig2_ampl * np.sin(2 * np.pi * time * sig2_freq_Hz)
sig3 = sig3_ampl * np.sin(2 * np.pi * time * sig3_freq_Hz)
sig = sig1 + sig2 + sig3

# np.random.seed(0)
# sig = sig + np.random.randn(len(sig))

if f_SHOW_PLOT:
    plt.figure(1)
    ax1 = plt.subplot(3, 1, 1)
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax3 = plt.subplot(3, 1, 3)
    plt.sca(ax1)

    # plt.plot(time, sig1, marker, color='k')
    # plt.plot(time, sig2, marker, color='k')
    # plt.plot(time, sig2, marker, color='k')
    # plt.plot(time, sig, marker, color=('0.8'))
    ax1.plot(time, sig1 + sig3, marker, color=("0.8"))

# -----------------------------------------
#   FIR filter
# -----------------------------------------

# Optimal scenario:
#   Perform convolve every incoming buffer
#   Maximize N_taps given N_BLOCKS
BLOCK_SIZE = 500  # [samples]
N_BLOCKS = 41

# Ring buffers
rb_capacity = BLOCK_SIZE * N_BLOCKS

if False:
    rb_time = deque(maxlen=rb_capacity)
    rb_sig_I = deque(maxlen=rb_capacity)
else:
    rb_time = RingBuffer(capacity=rb_capacity)
    rb_sig_I = RingBuffer(capacity=rb_capacity)

config = RingBuffer_FIR_Filter_Config(
    Fs=Fs,
    block_size=BLOCK_SIZE,
    N_blocks=N_BLOCKS,
    firwin_cutoff=[49, 51],
    firwin_window="blackmanharris",
    firwin_pass_zero=True,
)

firf = RingBuffer_FIR_Filter(config=config, use_CUDA=False,)
firf.compute_firwin_and_freqz()
firf.report()

residuals = np.full(len(time), np.nan)
tick = Time.perf_counter()

# Simulate incoming blocks on the fly
for i_block in range(int(len(time) / BLOCK_SIZE)):
    block_time = time[BLOCK_SIZE * i_block : BLOCK_SIZE * (i_block + 1)]
    block_sig_I = sig[BLOCK_SIZE * i_block : BLOCK_SIZE * (i_block + 1)]

    # Simulate dropped block
    if 0:
        if i_block == 43 or i_block == 46:
            continue

    # Extend ringbufffers with incoming data
    rb_time.extend(block_time)
    rb_sig_I.extend(block_sig_I)

    filt_I = firf.apply_filter(rb_sig_I)

    # Retrieve the block of original data from the past that aligns with
    # the current filter output
    old_time = np.array(rb_time)[firf.config.rb_valid_slice]
    old_sig_I = np.array(rb_sig_I)[firf.config.rb_valid_slice]

    if firf.filter_has_settled:
        slice_orig_array = slice(
            firf.config.rb_valid_slice.start
            + BLOCK_SIZE * (i_block - N_BLOCKS + 1),
            firf.config.rb_valid_slice.stop
            + BLOCK_SIZE * (i_block - N_BLOCKS + 1),
        )

        residuals[slice_orig_array] = (
            sig1[slice_orig_array] + sig3[slice_orig_array] - filt_I
        )

        if f_SHOW_PLOT:
            color = "r" if (i_block % 2 == 0) else "g"
            ax1.plot(old_time, filt_I, marker + color)

print(
    "\nsingle process step: %.2f ms"
    % ((Time.perf_counter() - tick) / (i_block + 1) * 1000)
)

# Quality check of band-gap filter
dev = np.nanstd(residuals)

if f_SHOW_PLOT:
    # Plot vertical lines indicating each incoming buffer
    y_bars = np.array(plt.ylim()) * 1.1
    for i in range(int(len(time) / BLOCK_SIZE)):
        ax1.plot([i * time[BLOCK_SIZE], i * time[BLOCK_SIZE]], y_bars, "k")

    plt.title("N_taps = %i, dev = %.3f" % (firf.config.firwin_numtaps, dev))
    plt.xlabel("time (s)")
    plt.ylabel("signal")
    plt.xlim(
        [firf.config.T_settle_filter - 0.1, firf.config.T_settle_filter + 0.8]
    )
    plt.ylim([-2.5, 2.5])
    plt.grid()

    # Plot residuals
    plt.sca(ax2)
    plt.plot(time, residuals, ".-b")
    plt.xlabel("time (s)")
    plt.ylabel("residuals")
    plt.grid()

    # Filter frequency response
    plt.sca(ax3)
    plt.plot(firf.freqz.full_freq_Hz, firf.freqz.full_ampl_dB, ".-b")
    plt.xlabel("frequency (Hz)")
    plt.ylabel("ampl. attn. (dB)")
    plt.xlim([48, 52])
    plt.grid()

    plt.sca(ax1)
    thismanager = pylab.get_current_fig_manager()
    if hasattr(thismanager, "window"):
        thismanager.window.setGeometry(500, 120, 1200, 700)
    plt.show()
