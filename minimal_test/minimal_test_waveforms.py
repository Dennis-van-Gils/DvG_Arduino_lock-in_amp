# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 21:46:35 2019

@author: vangi
"""

import numpy as np
import matplotlib.pyplot as plt

BLOCK_SIZE = 250
SAMPLING_PERIOD = 200e-6  # [sec]
SAMPLING_RATE_Hz = 1 / SAMPLING_PERIOD  # [Hz]

ref_freq = 200  # [Hz]

if __name__ == "__main__":
    N_LUT = int(np.round(SAMPLING_RATE_Hz / ref_freq))
    ref_freq = SAMPLING_RATE_Hz / N_LUT

    lut_cos = np.full(N_LUT, np.nan)
    lut_sqr = np.full(N_LUT, np.nan)
    lut_saw = np.full(N_LUT, np.nan)
    lut_tri = np.full(N_LUT, np.nan)

    # Generate waveforms
    for i in range(N_LUT):
        # -- Sine
        # N_LUT integer multiple of 4: extrema [-1, 1], symmetric
        # N_LUT others               : extrema <-1, 1>, symmetric
        lut_cos[i] = np.sin(2 * np.pi * i / N_LUT)

        # -- Square
        # N_LUT even                 : extrema [-1, 1], symmetric
        # N_LUT odd                  : extrema [-1, 1], asymmetric !!!
        lut_sqr[i] = 1 if i / N_LUT < 0.5 else -1

        # -- Sawtooth
        #
        lut_saw[i] = 2 * i / (N_LUT - 1) - 1

        # -- Triangle
        # N_LUT integer multiple of 4: extrema [-1, 1], symmetric
        # N_LUT others               : extrema <-1, 1>, symmetric
        lut_tri[i] = np.arcsin(np.sin(2 * np.pi * i / N_LUT)) / np.pi * 2

    x_cos = np.resize(lut_cos, BLOCK_SIZE)
    x_sqr = np.resize(lut_sqr, BLOCK_SIZE)
    x_saw = np.resize(lut_saw, BLOCK_SIZE)
    x_tri = np.resize(lut_tri, BLOCK_SIZE)

    print("N_LUT: %i" % N_LUT)
    print("Sine    : [%7.4f, %7.4f]" % (min(x_cos), max(x_cos)))
    print("Square  : [%7.4f, %7.4f]" % (min(x_sqr), max(x_sqr)))
    print("Sawtooth: [%7.4f, %7.4f]" % (min(x_saw), max(x_saw)))
    print("Triangle: [%7.4f, %7.4f]" % (min(x_tri), max(x_tri)))

    t = np.arange(BLOCK_SIZE) * SAMPLING_PERIOD
    plt.plot(t, np.zeros(BLOCK_SIZE), "-k")
    plt.plot(t, np.ones(BLOCK_SIZE), "-k")
    plt.plot(t, np.ones(BLOCK_SIZE) / 2, "-k")
    plt.plot(t, x_cos, ".-")
    plt.plot(t, x_sqr, ".-")
    plt.plot(t, x_saw, ".-")
    plt.plot(t, x_tri, ".-")

    plt.xlim([0, 3 / ref_freq])
    plt.show()
