#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Performs lightning-fast convolutions on 1D input arrays. The convolution is
based on the fast-Fourier transform (FFT) as computed by the excellent `fftw`
(http://www.fftw.org/) library. It will plan the transformations ahead of time
to optimize the calculations. Also, multiple threads can be specified for the
FFT and, when set to > 1, the Python GIL will not be invoked. This results in
true multithreading across multiple cores, which can result in a huge
performance gain.
"""
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/python-dvg-signal-processing"
__date__ = "28-05-2021"
__version__ = "1.0.0"
# pylint: disable=invalid-name, missing-function-docstring

import sys
import numpy as np
import pyfftw
from numba import njit


# pylint: disable=pointless-string-statement
""" DEV NOTE:
One might be tempted to use a `numba.njit(nogil=True)` decorator on the
`numpy.concatenate()` functions appearing in this module, but timeit tests
reveal that it actually hurts the performance. Numpy has already optimised its
`concatenate()` method to maximum performance.

    # Don't use this:
    @njit("float64[:](float64[:], float64[:])", **p_njit)
    def fast_concatenate(in1: np.ndarray, in2: np.ndarray) -> np.ndarray:
        return np.concatenate((in1, in2))
"""


@njit("complex128[:](complex128[:], complex128[:])", nogil=True, cache=True)
def fast_multiply(in1: np.ndarray, in2: np.ndarray) -> np.ndarray:
    return np.multiply(in1, in2)


# ------------------------------------------------------------------------------
#   FFTW_Convolver_Valid1D
# ------------------------------------------------------------------------------


class FFTW_Convolver_Valid1D:
    """Manages a fast-Fourier transform (FFT) convolution on 1D input arrays
    `in1` and `in2` as passed to method `convolve()`, which will return the
    result as a contiguous C-style `numpy.ndarray` containing only the 'valid'
    convolution elements.

    When the lengths of the passed input arrays are not equal to the lengths
    `len1` and `len2` as specified during the object creation, an array full of
    `numpy.nan`s is returned.

    Args:
        len1 (int):
            Full length of the upcoming input array `in1` passed to method
            `convolve()`.

        len2 (int):
            Full length of the upcoming input array `in2` passed to method
            `convolve()`.

        fftw_threads (int, optional):
            Number of threads to use for the FFT transformations. When set to
            > 1, the Python GIL will not be invoked.

            Default: 5
    """

    def __init__(self, len1: int, len2: int, fftw_threads: int = 5):
        # Check that input sizes are compatible with 'valid' mode
        self.switch_inputs = len2 > len1
        if self.switch_inputs:
            len1, len2 = len2, len1

        self.len1 = len1
        self.len2 = len2

        # Speed up FFT by zero-padding to optimal size for FFTW
        self.fast_len = pyfftw.next_fast_len(len1 + len2 - 1)
        self.padding_in1 = np.zeros(self.fast_len - self.len1)
        self.padding_in2 = np.zeros(self.fast_len - self.len2)

        # Compute the slice containing the valid convolution results
        self.valid_len = len1 - len2 + 1
        idx_start = (2 * len2 - 2) // 2
        self.valid_slice = slice(idx_start, idx_start + self.valid_len)

        # Create the FFTW plans
        # fmt: off
        fast_len2 = self.fast_len // 2 + 1
        self._rfft_in1  = pyfftw.empty_aligned(self.fast_len, dtype="float64")
        self._rfft_in2  = pyfftw.empty_aligned(self.fast_len, dtype="float64")
        self._rfft_out1 = pyfftw.empty_aligned(fast_len2    , dtype="complex128")
        self._rfft_out2 = pyfftw.empty_aligned(fast_len2    , dtype="complex128")
        self._irfft_in  = pyfftw.empty_aligned(fast_len2    , dtype="complex128")
        self._irfft_out = pyfftw.empty_aligned(self.fast_len, dtype="float64")
        # fmt: on

        print("Creating FFTW plans for convolution...", end="")
        sys.stdout.flush()

        p = {
            "flags": ("FFTW_MEASURE", "FFTW_DESTROY_INPUT"),
            "threads": fftw_threads,
        }
        self._fftw_rfft1 = pyfftw.FFTW(self._rfft_in1, self._rfft_out1, **p)
        self._fftw_rfft2 = pyfftw.FFTW(self._rfft_in2, self._rfft_out2, **p)
        self._fftw_irfft = pyfftw.FFTW(
            self._irfft_in,
            self._irfft_out,
            direction="FFTW_BACKWARD",
            **p,
        )

        print(" done.")

    # --------------------------------------------------------------------------
    #   convolve
    # --------------------------------------------------------------------------

    def convolve(self, in1: np.ndarray, in2: np.ndarray) -> np.ndarray:
        """Performs the FFT convolution on input arrays `in1` and `in2` and
        returns the result as a contiguous C-style `numpy.ndarray` containing
        only the 'valid' convolution elements.

        When the lengths of the passed input arrays are not equal to the lengths
        `len1` and `len2` as specified during the object creation, an
        array full of `np.nan`s is returned.

        Returns:
            The valid convolution results as a 1D numpy array.
        """
        # Force contiguous C-style numpy arrays, super fast when already so
        in1 = np.asarray(in1)
        in2 = np.asarray(in2)

        # Return np.nans when the input arrays are not fully populated yet
        if len(in1) != self.len1 or len(in2) != self.len2:
            return np.full(self.valid_len, np.nan)

        # Check that input sizes are compatible with 'valid' mode
        if self.switch_inputs:
            in1, in2 = in2, in1

        # Perform FFT convolution
        # -----------------------
        # Zero padding and forwards Fourier transformation
        self._rfft_in1[:] = np.concatenate((in1, self.padding_in1))
        self._rfft_in2[:] = np.concatenate((in2, self.padding_in2))
        self._fftw_rfft1()
        self._fftw_rfft2()

        # Convolution and backwards Fourier transformation
        self._irfft_in[:] = fast_multiply(self._rfft_out1, self._rfft_out2)
        result = self._fftw_irfft()

        # Return only the 'valid' elements
        return result[self.valid_slice]
