# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 21:22:33 2018

@author: vangi
"""

import timeit
import numpy as np
from scipy.signal import firwin

# Original data series
Fs = 1;                         # [Hz] sampling frequency
time = np.arange(400)           # [s] time
sig = (np.sin(2*np.pi*.02*time) + 
       np.sin(2*np.pi*.01*time) + 
       np.sin(2*np.pi*.1*time))
np.random.seed(0)
#sig = sig + np.random.randn(len(sig))

# -----------------------------------------
#   FIR filters
# -----------------------------------------

BLOCK_SIZE = 200
NTAPS = 101;

num_valid   = BLOCK_SIZE - NTAPS + 1
offset_valid = int((NTAPS - 1)/2)

b_LP = firwin(NTAPS, [0.0001, 0.05], width=0.05, fs=Fs, pass_zero=False)
b_HP = firwin(NTAPS, 0.05          , width=0.05, fs=Fs, pass_zero=False)
b_BP = firwin(NTAPS, [0.019, 0.021], width=0.01, fs=Fs, pass_zero=False)
b_BG = firwin(NTAPS, ([0.0001, 0.015, 0.025, .5-1e-4]), width=0.005, fs=Fs, pass_zero=False)

def apply_FIR_filters():
    iter_no = 0;
    
    idx_start = iter_no * num_valid
    idx_end   = idx_start + BLOCK_SIZE
    
    sig_LP = np.convolve(sig[idx_start:idx_end], b_LP, mode='valid')
    sig_HP = np.convolve(sig[idx_start:idx_end], b_HP, mode='valid')
    sig_BP = np.convolve(sig[idx_start:idx_end], b_BP, mode='valid')
    sig_BG = np.convolve(sig[idx_start:idx_end], b_BG, mode='valid')
    
    time_sel_valid = time[idx_start + offset_valid:
                          idx_start + offset_valid + num_valid]
                    
#timeit.timeit('apply_FIR_filters()', setup='from test_filter_speed import apply_FIR_filters', number=1000)