#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Performs lightning-fast power-spectrum calculations on 1D time series data
acquired at a fixed sampling frequency using Welch's method.

The fast-Fourier transform (FFT) is performed by the excellent `fftw`
(http://www.fftw.org/) library. It will plan the transformations ahead of time
to optimize the calculations. Also, multiple threads can be specified for the
FFT and, when set to > 1, the Python GIL will not be invoked. This results in
true multithreading across multiple cores, which can result in a huge
performance gain. It can outperform the `numpy` and `scipy` libraries by a
factor of > 8 in calculation speed.

Futher improvement to the calculation speed in this module comes from the use of
the `numba.njit()` decorator around arithmetic functions, releasing the Python
GIL as well.
"""
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/python-dvg-signal-processing"
__date__ = "29-05-2021"
__version__ = "1.0.0"
# pylint: disable=invalid-name, missing-function-docstring, too-many-instance-attributes

import sys
import numpy as np
from scipy import signal
import pyfftw
from numba import njit

p_njit = {"nogil": True, "cache": True, "fastmath": False}


@njit("float64[:,:](float64[:], float64[:,:])", **p_njit)
def fast_multiply_window(window: np.ndarray, data: np.ndarray) -> np.ndarray:
    return np.multiply(window, data)


@njit("float64[:,:](complex128[:,:], float64)", **p_njit)
def fast_conjugate_rescale(data: np.ndarray, scale: float) -> np.ndarray:
    data = np.multiply(np.conjugate(data), data)
    return np.multiply(np.real(data), scale)


@njit("float64[:,:](float64[:,:])", **p_njit)
def fast_transpose(data: np.ndarray) -> np.ndarray:
    return np.transpose(data)


@njit("float64[:](float64[:])", **p_njit)
def fast_10log10(data: np.ndarray) -> np.ndarray:
    return np.multiply(np.log10(data), 10)


# ------------------------------------------------------------------------------
#   FFTW_WelchPowerSpectrum
# ------------------------------------------------------------------------------


class FFTW_WelchPowerSpectrum:
    """Manages a power-spectrum calculation on 1D time series data `data` as
    passed to methods `compute_spectrum()` or `compute_spectrum_dBV()`.

    The input data array must always be of the same length as specified by
    argument `len_data`. When the length of the passed input array is not equal
    to the `len_data`, an array full of `numpy.nan`s is returned.

    The Welch algorithm is based on: `scipy.signal.welch()` with hard-coded
    defaults:
        window   = 'hanning'
        noverlap = 50 %
        detrend  = False
        scaling  = 'spectrum'
        mode     = 'psd'
        boundary = None
        padded   = False
        sides    = 'onesided'

    Args:
        len_data (int):
            Full length of the upcoming input array `data` passed to methods
            `compute_spectrum()` or `compute_spectrum_dBV().

        fs (float):
            Sampling frequency of the time series data [Hz].

        nperseg (float):
            Length of each segment in Welch's method to average over.

        fftw_threads (int, optional):
            Number of threads to use for the FFT transformations. When set to
            > 1, the Python GIL will not be invoked.

            Default: 5

    Attributes:
        freqs (np.ndarray):
            The frequency table in [Hz] corresponding to the power spectrum
            output of `compute_spectrum()` and `compute_spectrum_dBV()`.
    """

    def __init__(self, len_data: int, fs: float, nperseg: int, fftw_threads=5):
        nperseg = int(nperseg)
        if nperseg > len_data:
            print(
                "nperseg = {0:d} is greater than input length "
                " = {1:d}, using nperseg = {1:d}".format(nperseg, len_data)
            )
            nperseg = len_data

        self.len_data = len_data
        self.fs = fs
        self.nperseg = nperseg

        # Calculate the Hanning window in advance
        self.win = signal.hann(nperseg, False)
        self.scale = 1.0 / self.win.sum() ** 2  # For normalization

        # Calculate the frequency table in advance
        self.freqs = np.fft.rfftfreq(nperseg, 1 / fs)

        # Prepare the FFTW plan
        # fmt: off
        self.noverlap  = nperseg // 2
        self.step      = nperseg - self.noverlap
        self.shape_in  = ((len_data - self.noverlap) // self.step, nperseg)
        self.shape_out = (
            (len_data - self.noverlap) // self.step,
            nperseg // 2 + 1,
        )
        # fmt: on

        self._rfft_in = pyfftw.empty_aligned(self.shape_in, dtype="float64")
        self._rfft_out = pyfftw.empty_aligned(
            self.shape_out, dtype="complex128"
        )

        print("Creating FFTW plan for Welch power spectrum...", end="")
        sys.stdout.flush()

        self._fftw_welch = pyfftw.FFTW(
            self._rfft_in,
            self._rfft_out,
            flags=("FFTW_MEASURE", "FFTW_DESTROY_INPUT"),
            threads=fftw_threads,
        )
        print(" done.")

    # --------------------------------------------------------------------------
    #   compute_spectrum
    # --------------------------------------------------------------------------

    def compute_spectrum(self, data: np.ndarray) -> np.ndarray:
        """Returns the power spectrum array of the passed 1D time series array
        `data`. When `data` is in units [V], the output units are [V^2]. Use
        `compute_spectrum_dBV()` to get the equivalent power ratio in units of
        [dBV].

        Returns:
            The power spectrum array as a 1D numpy array in units of [V^2].
        """
        x = np.asarray(data)

        if self.len_data != len(x):
            return (
                np.full(self.len_data, np.nan),
                np.full(self.len_data, np.nan),
            )

        strides = (self.step * x.strides[-1], x.strides[-1])
        Pxx_in = np.lib.stride_tricks.as_strided(
            x, shape=self.shape_in, strides=strides
        )

        # Apply window
        Pxx_in = fast_multiply_window(self.win, Pxx_in)

        # Perform the fft
        self._rfft_in[:] = Pxx_in  # float64
        Pxx = self._fftw_welch()  # returns complex128

        # Equivalent of:
        # Pxx = np.conjugate(Pxx) * Pxx
        # Pxx = Pxx.real * self.scale
        Pxx = fast_conjugate_rescale(Pxx, self.scale)

        if self.nperseg % 2:
            Pxx[..., 1:] *= 2
        else:
            # Last point is unpaired Nyquist freq point, don't double
            Pxx[..., 1:-1] *= 2

        Pxx = fast_transpose(Pxx)

        # Average over windows
        if len(Pxx.shape) >= 2 and Pxx.size > 0:
            if Pxx.shape[-1] > 1:
                Pxx = Pxx.mean(axis=-1)
            else:
                Pxx = np.reshape(Pxx, Pxx.shape[:-1])

        return Pxx

    # --------------------------------------------------------------------------
    #   compute_spectrum_dBV
    # --------------------------------------------------------------------------

    def compute_spectrum_dBV(self, data: np.ndarray) -> np.ndarray:
        """Like `compute_spectrum()`, but now output as the power ratio in [dBV]
        assuming `data` is in units of [V].

        Returns:
            The power spectrum array as a 1D numpy array in units of [dBV].
        """
        return fast_10log10(self.compute_spectrum(data))
