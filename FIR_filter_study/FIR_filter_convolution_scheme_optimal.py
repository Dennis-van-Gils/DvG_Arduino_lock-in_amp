# -*- coding: utf-8 -*-
"""
Dennis van Gils
23-04-2019
"""

from collections import deque
import numpy as np
import time as Time

import matplotlib as mpl
#mpl.use("qt5agg")
import matplotlib.pyplot as plt
import pylab
mpl.rcParams['lines.linewidth'] = 3.0
marker = '-'

from DvG_Buffered_FIR_Filter import Buffered_FIR_Filter

f_SHOW_PLOT = True

# Original data series
Fs = 5000;           # [Hz] sampling frequency
total_time = 8;      # [s]
time = np.arange(0, total_time, 1/Fs) # [s]

sig1_ampl    = 1
sig1_freq_Hz = 48.8

sig2_ampl    = 1
sig2_freq_Hz = 50

sig3_ampl    = 1
sig3_freq_Hz = 52

sig1 = sig1_ampl * np.sin(2*np.pi*time * sig1_freq_Hz)
sig2 = sig2_ampl * np.sin(2*np.pi*time * sig2_freq_Hz)
sig3 = sig3_ampl * np.sin(2*np.pi*time * sig3_freq_Hz)
sig = sig1 + sig2 + sig3

#np.random.seed(0)
#sig = sig + np.random.randn(len(sig))

if f_SHOW_PLOT:
    plt.figure(1)
    ax1 = plt.subplot(3, 1, 1)
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax3 = plt.subplot(3, 1, 3)
    plt.sca(ax1)

    #plt.plot(time, sig1, marker, color='k')
    #plt.plot(time, sig2, marker, color='k')
    #plt.plot(time, sig2, marker, color='k')
    #plt.plot(time, sig, marker, color=('0.8'))
    ax1.plot(time, sig1+sig3, marker, color=('0.8'))

# -----------------------------------------
#   FIR filter
# -----------------------------------------

# Optimal scenario:
#   Perform convolve every incoming buffer
#   Maximize N_taps given N_buffers_in_deque
BUFFER_SIZE = 500  # [samples]
N_buffers_in_deque = 41

N_deque = BUFFER_SIZE * N_buffers_in_deque
deque_time  = deque(maxlen=N_deque)
deque_sig_I = deque(maxlen=N_deque)

firwin_cutoff = [49, 51]
firwin_window = "blackmanharris"
firf = Buffered_FIR_Filter(BUFFER_SIZE,
                           N_buffers_in_deque,
                           Fs,
                           firwin_cutoff,
                           firwin_window,
                           pass_zero=True,
                           use_CUDA=False)
firf.report()

tick = Time.time()
residuals = np.full(len(time), np.nan)
for i_buffer in range(int(len(time)/BUFFER_SIZE)):
    # Simulate incoming buffers on the fly
    buffer_time  = time[BUFFER_SIZE * i_buffer:
                        BUFFER_SIZE * (i_buffer + 1)]
    buffer_sig_I = sig [BUFFER_SIZE * i_buffer:
                        BUFFER_SIZE * (i_buffer + 1)]
           
    # Simulate dropped buffer
    if 0:
        if i_buffer == 43 or i_buffer == 46:
            continue
        
    # Extend deque with incoming buffer data
    deque_time.extend(buffer_time)
    deque_sig_I.extend(buffer_sig_I)

    filt_I = firf.process(deque_sig_I)
    
    # Retrieve the block of original data from the past that aligns with
    # the current filter output
    old_time  = (np.array(deque_time)
                 [firf.win_idx_valid_start:firf.win_idx_valid_end])
    old_sig_I = (np.array(deque_sig_I)
                 [firf.win_idx_valid_start:firf.win_idx_valid_end])
    
    if firf.deque_has_settled:
        idx_orig_array_start = (BUFFER_SIZE * (i_buffer - N_buffers_in_deque + 1) +
                                firf.win_idx_valid_start)
        idx_orig_array_end   = (BUFFER_SIZE * (i_buffer - N_buffers_in_deque + 1) +
                                firf.win_idx_valid_end)
        idx_orig_array = np.arange(idx_orig_array_start,
                                   idx_orig_array_end)
        residuals[idx_orig_array] = (sig1[idx_orig_array] +
                                     sig3[idx_orig_array] -
                                     filt_I)
        
        if f_SHOW_PLOT:
            color = 'r' if (i_buffer % 2 == 0) else 'g'
            ax1.plot(old_time, filt_I, marker + color)

print('\nsingle process step: %.2f ms' %
      ((Time.time() - tick) / (i_buffer + 1) * 1000))

# Quality check of band-gap filter
dev = np.nanstd(residuals)

if f_SHOW_PLOT:
    # Plot vertical lines indicating each incoming buffer
    y_bars = np.array(plt.ylim()) * 1.1
    for i in range(int(len(time)/BUFFER_SIZE)):
        ax1.plot([i * time[BUFFER_SIZE], i * time[BUFFER_SIZE]], y_bars, 'k')

    plt.title("N_taps = %i, dev = %.3f" % (firf.N_taps, dev))
    plt.xlabel('time (s)')
    plt.ylabel('signal')
    plt.xlim([firf.T_settle_filter - 0.1, firf.T_settle_filter + .8])
    plt.ylim([-2.5, 2.5])
    plt.grid()
    
    # Plot residuals
    plt.sca(ax2)
    plt.plot(time, residuals, '.-b')
    plt.xlabel('time (s)')
    plt.ylabel('residuals')
    plt.grid()
    
    # Filter frequency response
    plt.sca(ax3)
    plt.plot(firf.full_resp_freq_Hz, firf.full_resp_ampl_dB, '.-b')
    plt.xlabel('frequency (Hz)')
    plt.ylabel('ampl. attn. (dB)')
    plt.xlim([48, 52])
    plt.grid()
    
    plt.sca(ax1)
    thismanager = pylab.get_current_fig_manager()
    if hasattr(thismanager, 'window'):
        thismanager.window.setGeometry(500, 120, 1200, 700)
    plt.show()
