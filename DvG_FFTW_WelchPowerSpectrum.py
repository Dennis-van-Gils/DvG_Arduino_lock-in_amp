# -*- coding: utf-8 -*-
"""Class FFTW_WelchPowerSpectrum is a fast implementation for calculating
power spectra using Welch's method on 1-D timeseries data. This class relies
upon the highly optimized FFTW library, which will outperform the numpy or
scipy libraries by a factor of ~8!

The windowing function is fixed to hanning with 50% overlap and no detrending
will take place on the input data. The input data must always be of the same
length.

Class:
    FFTW_WelchPowerSpectrum(len_data, fs, nperseg):
        Args:
            len_data:
                Length of the 1-D data array that will be fed into the power
                spectrum calculation each time when calling 'process()'.
            fs:
                Sampling frequency of the timeseries data [Hz].
            nperseg:
                Length of each segment in Welch's method.
                
        Methods:
            process(data):
                Returns the power spectrum array of the passed 1-D array
                'data'. The output units are V^2. You still have to apply
                10*log10() to get the power ratio in dB.
            process_dB(data):
                Like process(), but now output as the power ratio in dB.
                
        Important member:
            freqs:
                The frequency table in [Hz].
                

Based on: scipy.signal.welch()
  with hard-coded defaults: window   = 'hanning'
                            noverlap = 50 %
                            detrend  = False
                            scaling  = 'spectrum'
                            mode     = 'psd'
                            boundary = None
                            padded   = False
                            sides    = 'onesided'
"""
__author__      = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__         = "https://github.com/Dennis-van-Gils/DvG_Arduino_lock-in_amp"
__date__        = "03-08-2019"
__version__     = "1.0.0"

import numpy as np
from scipy import signal
import pyfftw
from numba import njit


@njit('void(float64[:], float64[:,:])', nogil=True, cache=True)
def fast_multiply_window(window_in, data):
    #Pxx_in = self.win * Pxx_in  # float64, leave as A = B * A
    data = np.multiply(window_in, data)

@njit('float64[:,:](complex128[:,:], float64)', nogil=True, cache=True)
def fast_conjugate_rescale(data_in, scale):
    #Pxx = np.conjugate(Pxx) * Pxx
    #Pxx = Pxx.real
    #Pxx *= self.scale
    data_out = np.conjugate(data_in)
    data_out = np.multiply(data_out, data_in)
    data_out = np.real(data_out)
    data_out = np.multiply(data_out, scale)
    return data_out

@njit('float64[:,:](float64[:,:])', nogil=True, cache=True)
def fast_transpose(data_in):
    data_out = np.transpose(data_in)
    return data_out

@njit('float64[:](float64[:])', nogil=True, cache=True)
def fast_10log10(data_in):
    data_out = np.log10(data_in)
    data_out = np.multiply(data_out, 10)
    return data_out


class FFTW_WelchPowerSpectrum:
    def __init__(self, len_data, fs, nperseg):        
        nperseg = int(nperseg)
        if nperseg > len_data:
            print('nperseg = {0:d} is greater than input length '
                  ' = {1:d}, using nperseg = {1:d}'
                  .format(nperseg, len_data))
            nperseg = len_data
        
        self.len_data = len_data
        self.fs       = fs
        self.nperseg  = nperseg
        
        # Calculate the Hanning window in advance
        self.win = signal.hann(nperseg, False)
        self.scale = 1.0 / self.win.sum()**2    # For normalization
        
        # Calculate the frequency table in advance
        self.freqs = np.fft.rfftfreq(nperseg, 1/fs)

        # Prepare the FFTW plan
        self.noverlap  = nperseg // 2
        self.step      = nperseg - self.noverlap
        self.shape_in  = ((len_data - self.noverlap)//self.step, nperseg)
        self.shape_out = ((len_data - self.noverlap)//self.step, nperseg//2 + 1)
        
        self._rfft_in  = pyfftw.empty_aligned(self.shape_in,  dtype='float64')
        self._rfft_out = pyfftw.empty_aligned(self.shape_out, dtype='complex128')
        
        flags = ('FFTW_PATIENT', 'FFTW_DESTROY_INPUT')
        print("Creating FFTW plan for Welch power spectrum...", end="")
        self._fftw_welch = pyfftw.FFTW(self._rfft_in, self._rfft_out,
                                       flags=flags)
        print(" done.")
        
    def process(self, data):
        x = np.asarray(data)
    
        if self.len_data != len(x):
            return (np.full(self.len_data, np.nan),
                    np.full(self.len_data, np.nan))
        
        strides = (self.step * x.strides[-1], x.strides[-1])
        Pxx_in = np.lib.stride_tricks.as_strided(x, shape=self.shape_in,
                                                 strides=strides)
    
        # Apply window by multiplication
        fast_multiply_window(self.win, Pxx_in)
        #Pxx_in = self.win * Pxx_in  # float64, leave as A = B * A
    
        # Perform the fft
        self._rfft_in[:] = Pxx_in   # float64
        Pxx = self._fftw_welch()    # returns complex128
        
        Pxx = fast_conjugate_rescale(Pxx, self.scale)
        #Pxx = np.conjugate(Pxx) * Pxx
        #Pxx = Pxx.real
        #Pxx *= self.scale
        
        if self.nperseg % 2:
            Pxx[..., 1:] *= 2
        else:
            # Last point is unpaired Nyquist freq point, don't double
            Pxx[..., 1:-1] *= 2

        Pxx = fast_transpose(Pxx)    
        #Pxx = Pxx.transpose()
    
        # Average over windows.
        if len(Pxx.shape) >= 2 and Pxx.size > 0:
            if Pxx.shape[-1] > 1:
                Pxx = Pxx.mean(axis=-1)
            else:
                Pxx = np.reshape(Pxx, Pxx.shape[:-1])
        
        return Pxx
    
    def process_dB(self, data):
        return fast_10log10(self.process(data))