# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 21:46:35 2019

@author: vangi
"""

import numpy as np
import matplotlib.pyplot as plt

BLOCK_SIZE = 250
SAMPLING_PERIOD = 200e-6                # [sec]
SAMPLING_RATE_Hz = 1/ SAMPLING_PERIOD   # [Hz]

ref_freq = 200   # [Hz]

if __name__ == "__main__":
    N_LUT = np.int(np.round(SAMPLING_RATE_Hz / ref_freq))
    ref_freq = SAMPLING_RATE_Hz / N_LUT

    lut_cos = np.full(N_LUT, np.nan)
    lut_sqr = np.full(N_LUT, np.nan)
    lut_saw = np.full(N_LUT, np.nan)
    lut_tri = np.full(N_LUT, np.nan)
    
    for i in range(N_LUT):
        # N_LUT even: [ 0, 1]
        # N_LUT odd : [>0, 1]
        lut_cos[i] = 0.5 * (1 + np.cos(2*np.pi * i / N_LUT))
        
        # Guaranteed  [ 0, 1]
        # Might need fmod in C for shifting the phase by N_LUT/4 to allign
        # the center of the high-state with the top with the cosine.
        #lut_sqr[i] = np.round((N_LUT - i) / N_LUT)
        lut_sqr[i] = np.round(((N_LUT - i - N_LUT/4) % N_LUT) / N_LUT)
        
        # Guaranteed  [ 0, 1]
        lut_saw[i] = (N_LUT - 1 - i) / (N_LUT - 1)
        
        # N_LUT even: [ 0, 1]
        # N_LUT odd : [>0, 1]
        lut_tri[i] = 2 * np.abs(i / N_LUT - 0.5) 
    
    x_cos = np.resize(lut_cos, BLOCK_SIZE)
    x_sqr = np.resize(lut_sqr, BLOCK_SIZE)
    x_saw = np.resize(lut_saw, BLOCK_SIZE)        
    x_tri = np.resize(lut_tri, BLOCK_SIZE)
    
    print("N_LUT: %i" % N_LUT)
    print("Cosine  : [%6.4f, %6.4f]" % (min(x_cos), max(x_cos)))
    print("Square  : [%6.4f, %6.4f]" % (min(x_sqr), max(x_sqr)))
    print("Sawtooth: [%6.4f, %6.4f]" % (min(x_saw), max(x_saw)))
    print("Triangle: [%6.4f, %6.4f]" % (min(x_tri), max(x_tri)))
    
    t = np.arange(BLOCK_SIZE) * SAMPLING_PERIOD
    plt.plot(t, np.zeros(BLOCK_SIZE), '-k')
    plt.plot(t, np.ones(BLOCK_SIZE), '-k')
    plt.plot(t, np.ones(BLOCK_SIZE)/2, '-k')
    plt.plot(t, x_cos, '.-')
    plt.plot(t, x_sqr, '.-')
    plt.plot(t, x_saw, '.-')
    plt.plot(t, x_tri, '.-')
    
    plt.xlim([0, 3/ref_freq])