#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dennis van Gils
23-05-2021
"""
# pylint: disable=invalid-name

import numpy as np
from scipy.signal import firwin, freqz

import matplotlib as mpl

mpl.rcParams["lines.linewidth"] = 2.0
# mpl.use("qt5agg")
import matplotlib.pyplot as plt
import pylab

# Create tap filters for upcoming convolution
windows = ("hamming", "hanning", "blackman", "blackmanharris", ("chebwin", 50))

# Fs [Hz], block_size, N_blocks, pass_zero, cutoff [Hz]
# fmt: off
tests = [
    [5000 , 500 , 21, True, [218]],
    [5000 , 500 , 41, True, [218]],

    [20000, 2000, 11, True, [218]],
    [20000, 2000, 21, True, [218]],
    [20000, 2000, 41, True, [218]],

    [5000 , 500 , 21, True, [49, 51]],
    [5000 , 500 , 41, True, [49, 51]],

    [20000, 2000, 11, True, [49, 51]],
    [20000, 2000, 21, True, [49, 51]],
    [20000, 2000, 41, True, [49, 51]],
]
# fmt: on

# Prepare plot
plt.figure(1)
ax1 = plt.subplot(2, 1, 1, label="ax1")
ax2 = plt.subplot(2, 1, 2, label="ax2")
marker = ".-"

thismanager = pylab.get_current_fig_manager()
if hasattr(thismanager, "window"):
    thismanager.window.setGeometry(500, 120, 1200, 900)

for test in tests:
    Fs, block_size, N_blocks, pass_zero, cutoff = test
    N_taps = block_size * (N_blocks - 1) + 1
    T_settle = (N_blocks - 1) * block_size / Fs  # [s]

    ax1.clear()
    ax2.clear()

    for window in windows:
        b_BG = firwin(N_taps, cutoff, window=window, fs=Fs, pass_zero=pass_zero)

        # Filter frequency response
        w, h = freqz(b_BG, worN=2 ** 18)
        freq_table_Hz = w / np.pi * Fs / 2

        np.seterr(divide="ignore")
        ax1.plot(freq_table_Hz, 20 * np.log10(abs(h)), marker, label=window)
        ax2.plot(freq_table_Hz, 20 * np.log10(abs(h)), marker, label=window)
        np.seterr(divide="warn")

    plt.sca(ax1)
    plt.title(
        "Fs = %.0f Hz, block_size = %i, N_blocks = %i, N_taps = %i\n"
        "T_settle = %.2f s, pass_zero = %s, cutoffs = %s"
        % (
            Fs,
            block_size,
            N_blocks,
            N_taps,
            T_settle,
            pass_zero,
            [round(x, 1) for x in cutoff],
        )
    )
    plt.ylabel("ampl. attn. (dB)")
    plt.xlabel("frequency (Hz)")
    if cutoff[0] < 50:
        plt.xlim([48, 52])
    else:
        plt.xlim([217, 221])
    plt.ylim([-130, 1])
    plt.grid()
    plt.legend()

    plt.sca(ax2)
    plt.ylabel("ampl. attn. (dB)")
    plt.xlabel("frequency (Hz)")
    if cutoff[0] < 50:
        plt.xlim([50.5, 54])
        plt.ylim([-0.4, 0.1])
    else:
        plt.xlim([216, 218])
        plt.ylim([-0.2, 0.1])
    plt.grid()
    # plt.legend()

    plt.savefig(
        "FIR_try_cutoff_%s_Fs_%i_N_%i.png"
        % ([round(x, 1) for x in cutoff], Fs, N_taps),
        dpi=300,
    )
    plt.show()
