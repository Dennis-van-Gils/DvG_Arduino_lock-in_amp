# -*- coding: utf-8 -*-
"""
Testing faster Welch power spectrum calculation

Dennis van Gils
24-05-2021
"""

import os
import sys
import inspect
import timeit
import platform

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))
)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from dvg_fftw_welchpowerspectrum import FFTW_WelchPowerSpectrum


def test1():
    global freq1, Pxx1
    freq1, Pxx1 = signal.welch(
        wave,
        fs=Fs,
        window="hanning",
        nperseg=Fs,
        detrend=False,
        scaling="spectrum",
    )


def test2():
    global Pxx2
    Pxx2 = fftw_welch.process(wave)


if __name__ == "__main__":
    # Generate harmonic wave
    Fs = 5000  # [Hz]
    t = np.arange(40500) * 1 / Fs  # [s]
    f1 = 200  # [Hz]
    f2 = 2000  # [Hz]
    f_fluct = 1 + np.sin(2 * np.pi * 1 * t) / 600
    wave = np.sin(2 * np.pi * (f1 * f_fluct) * t)
    wave += np.sin(2 * np.pi * f2 * t)

    # Init
    fftw_welch = FFTW_WelchPowerSpectrum(len(wave), fs=Fs, nperseg=Fs)

    freq1 = np.empty(Fs // 2 + 1)
    freq2 = fftw_welch.freqs
    Pxx1 = np.empty(Fs // 2 + 1)
    Pxx2 = np.empty(Fs // 2 + 1)

    # Logging
    try:
        f = open("timeit_Welch_power_spectrum.log", "w")
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

    report("Timeit: Welch power spectrum\n")

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

    # """
    result1 = np.array(timeit.repeat(test1, **p)) / N * 1000
    report("\nscipy.signal.welch:")
    for r in result1:
        report("%20.3f ms" % r)
    # """

    result2 = np.array(timeit.repeat(test2, **p)) / N * 1000
    report("\ndvg_fftw_welchpowerspectrum:")
    for r in result2:
        report("%20.3f ms" % r)

    report("\nTimes faster: %.2f" % (min(result1) / min(result2)))

    # -----------------------------------
    #   Plot for comparison
    # -----------------------------------

    """
    plt.plot(freq1, 10 * np.log10(Pxx1), '.-')
    plt.plot(freq2, 10 * np.log10(Pxx2), '.-r')
    plt.show()
    """
