# -*- coding: utf-8 -*-
"""
Testing faster fftconvolve

Dennis van Gils
29-05-2021
"""
# pylint: disable=invalid-name, wrong-import-position, import-error

import os
import sys
import timeit
import platform

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from dvg_fftw_convolver import FFTW_Convolver_Valid1D

TEST_SCIPY = True
TEST_CUDA = False

FFTW_THREADS = 3


def test_fftw():
    global wave_out2
    wave_out2 = fftw_convolver.convolve(wave, taps)


if TEST_SCIPY:

    def test_scipy():
        global wave_out1
        wave_out1 = signal.fftconvolve(wave, taps, mode="valid")


if TEST_CUDA:
    from numba import cuda
    import cupy
    import sigpy

    def test_cuda():
        global wave_out3
        # Transfer to GPU memory
        # We also transform the 1-D array to a column vector, preferred by CUDA
        wave_cp = cupy.array(wave[:, None])
        # Perform fft convolution on the GPU
        z_cp = sigpy.convolve(wave_cp, taps_cupy, mode="valid")
        # Transfer result back to CPU memory
        # And reduce dimension again
        wave_out3 = cupy.asnumpy(z_cp)[:, 0]


if __name__ == "__main__":
    # Generate signal
    block_size = 2000  # small: 500, large: 2000
    N_blocks = 21  # small: 21, large: 41
    rb_capacity = block_size * N_blocks
    N_taps = block_size * (N_blocks - 1) + 1

    Fs = 5000  # [Hz]
    t = np.arange(rb_capacity) * 1 / Fs  # [s]
    f1 = 100  # [Hz]
    f2 = 200  # [Hz]
    f_fluct = 1 + np.sin(2 * np.pi * 1 * t) / 600
    wave = np.sin(2 * np.pi * (f1 * f_fluct) * t)
    wave += np.sin(2 * np.pi * f2 * t)

    # Generate filter
    taps = signal.firwin(
        numtaps=N_taps,
        cutoff=[190, 210],
        window=("chebwin", 50),
        pass_zero=True,
        fs=Fs,
    )

    if TEST_CUDA:
        taps_cupy = cupy.array(taps[:, None])

    # Init
    fftw_convolver = FFTW_Convolver_Valid1D(
        len(wave), len(taps), fftw_threads=FFTW_THREADS
    )
    wave_out1 = np.empty(rb_capacity - N_taps + 1)
    wave_out2 = np.empty(rb_capacity - N_taps + 1)
    wave_out3 = np.empty(rb_capacity - N_taps + 1)

    # Logging
    try:
        f = open("timeit_FIR_filter_fftconvolve.log", "w")
    except:  # pylint: disable=bare-except
        f = None

    def report(str_txt):
        if f is not None:
            f.write("%s\n" % str_txt)
        print(str_txt)

    # -----------------------------------
    #   Timeit
    # -----------------------------------
    N = 1000
    REPS = 5
    p = {"number": N, "repeat": REPS}  # , 'setup': 'gc.enable()'}

    report("Timeit: fftconvolve\n")

    uname = platform.uname()
    report("Conda environment:")
    report("  %s\n" % os.environ["CONDA_PREFIX"])
    report("Running on...")
    report("  node   : %s" % uname.node)
    report("  system : %s" % uname.system)
    report("  release: %s" % uname.release)
    report("  version: %s" % uname.version)
    report("  machine: %s" % uname.machine)
    report("  proc   : %s" % uname.processor)
    report("  Python : %s" % platform.python_version())

    report("\nN = %i, REPS = %i" % (N, REPS))

    report("\nblock size = %i" % block_size)
    report("N_blocks   = %i" % N_blocks)
    report("--> rb_capacity = %i" % rb_capacity)
    report("--> N_taps      = %i" % N_taps)

    if TEST_SCIPY:
        result1 = np.array(timeit.repeat(test_scipy, **p)) / N * 1000
        report("\n#1  scipy.signal.fftconvolve:")
        for r in result1:
            report("%20.3f ms" % r)

    result2 = np.array(timeit.repeat(test_fftw, **p)) / N * 1000
    report("\n#2  dvg_fftw_convolver.convolve:")
    report("    FFTW_THREADS = %d" % FFTW_THREADS)
    for r in result2:
        report("%20.3f ms" % r)

    if TEST_CUDA:
        result3 = np.array(timeit.repeat(test_cuda, **p)) / N * 1000
        report("\n#3  CUDA sigpy.convolve:")
        for r in result3:
            report("%20.3f ms" % r)

    if TEST_SCIPY:
        report("\nTimes faster #1/#2: %.2f" % (min(result1) / min(result2)))

    if TEST_CUDA:
        report("Times faster #1/#3: %.2f" % (min(result1) / min(result3)))

    # -----------------------------------
    #   Plot for comparison
    # -----------------------------------

    if False:
        plt.plot(wave_out2, ".-r")

        if TEST_SCIPY:
            plt.plot(wave_out1, ".-")

        if TEST_CUDA:
            plt.plot(wave_out3, ".-k")
        plt.show()
