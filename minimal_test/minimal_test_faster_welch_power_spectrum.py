# -*- coding: utf-8 -*-
"""
Minimal test case for faster Welch power spectrum calculation

Dennis van Gils
01-08-2019
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import time as Time
import pyfftw

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from DvG_FFTW_WelchPowerSpectrum import FFTW_WelchPowerSpectrum
    
if __name__ == "__main__":
    # Generate signal
    Fs = 5000                    # [Hz]
    t = np.arange(40500) * 1/Fs  # [s]
    f1 = 200                     # [Hz]
    f2 = 2000                    # [Hz]
    f_fluct = 1 + np.sin(2*np.pi*1*t)/600
    wave  = np.sin(2*np.pi*(f1 * f_fluct)*t)
    wave += np.sin(2*np.pi*f2*t)
    
    # -----------------------------------
    #   Default Welch using scipy
    #   CPU    
    # -----------------------------------
    
    tick = Time.process_time()
    for i in range(1000):
        [f1, Pxx1] = signal.welch(wave,
                                  fs=Fs,
                                  window='hanning',
                                  nperseg=Fs,
                                  detrend=False,
                                  scaling='spectrum')
    print("%.3f ms" % (Time.process_time() - tick))
    
    # -----------------------------------
    #   Custom Welch using fftw
    #   CPU
    # -----------------------------------
    
    pyfftw.forget_wisdom()
    
    fftw_welch = FFTW_WelchPowerSpectrum(len(wave),
                                         fs=Fs,
                                         nperseg=Fs)
    
    tick = Time.process_time()
    for i in range(1000):
        Pxx2 = fftw_welch.process(wave)
    print("%.3f ms" % (Time.process_time() - tick))
    
    f2 = fftw_welch.freqs

    # -----------------------------------
    #   Plot
    # -----------------------------------

    plt.plot(f1, np.multiply(np.log10(Pxx1), 10), '.-')
    plt.plot(f2, np.multiply(np.log10(Pxx2), 10), '.-r')
    plt.show
