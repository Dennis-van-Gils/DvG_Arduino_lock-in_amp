#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

FFT: Fast-Fourier Transform
"""
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/DvG_Arduino_lock-in_amp"
__date__ = "21-05-2021"
__version__ = "1.0.0"
# pylint: disable=pointless-string-statement, invalid-name

import sys
import numpy as np
import pyfftw

"""
from numba import njit


@njit("complex128[:](complex128[:], complex128[:])", nogil=True, cache=True)
def fast_multiply(in1: np.ndarray, in2: np.ndarray) -> np.ndarray:
    return np.multiply(in1, in2)
"""


class FFTW_Convolver_Valid1D:
    """
    """

    def __init__(self, len1: int, len2: int):
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

        p = {"flags": ("FFTW_MEASURE", "FFTW_DESTROY_INPUT")}
        self._fftw_rfft1 = pyfftw.FFTW(self._rfft_in1, self._rfft_out1, **p)
        self._fftw_rfft2 = pyfftw.FFTW(self._rfft_in2, self._rfft_out2, **p)
        self._fftw_irfft = pyfftw.FFTW(
            self._irfft_in, self._irfft_out, direction="FFTW_BACKWARD", **p,
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
            The convolution result.
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
        # Note: Using `numba.njit()` on the following `np.multiply()` operation
        # has no apparent speed improvement. Tested 21-05-2021. We'll forgo
        # the use of numba.njit().
        # self._irfft_in[:] = fast_multiply(self._rfft_out1, self._rfft_out2)
        self._irfft_in[:] = np.multiply(self._rfft_out1, self._rfft_out2)
        ret = self._fftw_irfft()

        # Return only the 'valid' elements
        return ret[self.valid_slice]
