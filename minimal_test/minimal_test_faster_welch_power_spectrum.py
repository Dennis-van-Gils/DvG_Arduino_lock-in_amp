# -*- coding: utf-8 -*-
"""
Minimal test case for faster Welch power spectrum calculation

Dennis van Gils
31-07-2019
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import pyfftw
import time as Time

#from numba import cuda
#import cupy
#import sigpy
#import scipy

def custom_welch_using_fftw(x, fs=1.0, nperseg=None):
    # Hard-coded defaults:
    # window   = 'hanning'
    # noverlap = 50 %
    # detrend  = False
    # scaling  = 'spectrum'
    # mode     = 'psd'
    # boundary = None
    # padded   = False
    # sides    = 'onesided'
    
    x = np.asarray(x)
    outdtype = np.result_type(x, np.complex64)

    if x.size == 0:
        return np.empty(x.shape), np.empty(x.shape), np.empty(x.shape)

    if nperseg is not None:  # if specified by user
        nperseg = int(nperseg)
        if nperseg < 1:
            raise ValueError('nperseg must be a positive integer')

    input_length = x.shape[-1]
    if nperseg is None:
        nperseg = 256  # then change to default
    if nperseg > input_length:
        print('nperseg = {0:d} is greater than input length '
              ' = {1:d}, using nperseg = {1:d}'
              .format(nperseg, input_length))
        nperseg = input_length
    
    #win = signal.hann(nperseg, False)  # Hanning window

    noverlap = nperseg//2
    scale    = 1.0 / win.sum()**2
    freqs    = np.fft.rfftfreq(nperseg, 1/fs)
    
    # -------------------------------------
    #   Perform the windowed FFTs
    # -------------------------------------
    
    # Created strided array of data segments
    if nperseg == 1 and noverlap == 0:
        Pxx = x[..., np.newaxis]
    else:
        # http://stackoverflow.com/a/5568169
        step = nperseg - noverlap
        shape = x.shape[:-1]+((x.shape[-1]-noverlap)//step, nperseg)
        strides = x.strides[:-1]+(step*x.strides[-1], x.strides[-1])
        Pxx = np.lib.stride_tricks.as_strided(x, shape=shape,
                                                 strides=strides)

    # Apply window by multiplication
    Pxx = win * Pxx

    # Perform the fft. Acts on last axis by default. Zero-pads automatically
    #Pxx = np.fft.rfft(Pxx.real, n=nperseg)
    Pxx_fftw[:] = Pxx.real
    Pxx = welch_fftw_rfft() # returns the output
    
    Pxx = np.conjugate(Pxx) * Pxx
    Pxx *= scale
    
    if nperseg % 2:
        Pxx[..., 1:] *= 2
    else:
        # Last point is unpaired Nyquist freq point, don't double
        Pxx[..., 1:-1] *= 2

    Pxx = Pxx.astype(outdtype)
    Pxx = Pxx.real
    Pxx = Pxx.transpose()

    # Average over windows.
    if len(Pxx.shape) >= 2 and Pxx.size > 0:
        if Pxx.shape[-1] > 1:
            Pxx = Pxx.mean(axis=-1)
        else:
            Pxx = np.reshape(Pxx, Pxx.shape[:-1])
    
    return freqs, Pxx.real

#"""
    
if __name__ == "__main__":
    # Generate signal
    Fs = 5000                    # [Hz]
    t = np.arange(40500) * 1/Fs  # [s]
    f1 = 200                     # [Hz]
    f2 = 2000                    # [Hz]
    f_fluct = 1 + np.sin(2*np.pi*1*t)/600
    deque_in  = np.sin(2*np.pi*(f1 * f_fluct)*t)
    deque_in += np.sin(2*np.pi*f2*t)
    
    # --------------------------------------
    #   Default numpy Welch
    #   CPU    
    # -----------------------------------
    
    tick = Time.process_time()
    
    for i in range(1000):
        [f, Pxx] = signal.welch(deque_in,
                                fs=Fs,
                                window='hanning',
                                nperseg=Fs,
                                detrend=False,
                                scaling='spectrum')
        Pxx = 10 * np.log10(Pxx)
    
    print("%.3f ms" % (Time.process_time() - tick))
    
    # -----------------------------------
    #   Custom pyFFTW Welch
    #   CPU
    # -----------------------------------
    
    """
    # Configure PyFFTW to use all cores (the default is single-threaded)
    #pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
    # Monkey patch fftpack
    np.fft = pyfftw.interfaces.numpy_fft
    # Turn on the cache for optimum performance
    pyfftw.interfaces.cache.enable()
    """
    
    # Turn on the cache for optimum performance
    pyfftw.interfaces.cache.enable()
    signal.fftconvolve
    nperseg  = Fs
    noverlap = Fs//2
    step     = Fs - noverlap
    shape = (len(deque_in) - noverlap)//step, nperseg
    Pxx_fftw = pyfftw.empty_aligned(shape, dtype='float64')    
    welch_fftw_rfft = pyfftw.builders.rfft(Pxx_fftw,
                                           n=nperseg,
                                           threads=1)
    
    win = signal.hann(nperseg, False)  # Hanning window
    
    tick = Time.process_time()
    
    for i in range(1000):
        [f2, Pxx2] = custom_welch_using_fftw(deque_in,
                                             fs=Fs,
                                             nperseg=Fs)                     
        Pxx2 = 10 * np.log10(Pxx2)
    
    print("%.3f ms" % (Time.process_time() - tick))
    
    plt.plot(f, Pxx, '.-')
    plt.plot(f2, Pxx2, '.-r')
    plt.show
   
