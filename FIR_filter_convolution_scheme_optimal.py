# -*- coding: utf-8 -*-
"""
Dennis van Gils
13-02-2019
"""

import numpy as np
import time as Time

import matplotlib as mpl
#mpl.use("qt5agg")
import matplotlib.pyplot as plt
import pylab
mpl.rcParams['lines.linewidth'] = 3.0
marker = '-'

from DvG_Buffered_FIR_Filter import DvG_Buffered_FIR_Filter

f_SHOW_PLOT = True

# Original data series
Fs = 10000;          # [Hz] sampling frequency
total_time = 8;      # [s]
time = np.arange(0, total_time, 1/Fs) # [s]

sig1_ampl    = 1
sig1_freq_Hz = 49.2

sig2_ampl    = 1
sig2_freq_Hz = 50

sig3_ampl    = 1
sig3_freq_Hz = 52

sig1 = sig1_ampl * np.sin(2*np.pi*time * sig1_freq_Hz)
sig2 = sig2_ampl * np.sin(2*np.pi*time * sig2_freq_Hz)
#sig2 = (sig2_ampl * np.sin(2*np.pi*time * sig2_freq_Hz)).round()
sig3 = sig3_ampl * np.sin(2*np.pi*time * sig3_freq_Hz)
sig = sig1 + sig2 + sig3

#np.random.seed(0)
#sig = sig + np.random.randn(len(sig))

if f_SHOW_PLOT:
    plt.figure(1)
    ax1 = plt.subplot(3, 1, 1)
    ax2 = plt.subplot(3, 1, 2)
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
N_buffers_in_deque = 81

firwin_cutoff = [0.0001, 49.5, 50.5, Fs/2-0.0001]
firwin_window = ("chebwin", 50)

firf = DvG_Buffered_FIR_Filter(BUFFER_SIZE,
                               N_buffers_in_deque,
                               Fs,
                               firwin_cutoff,
                               firwin_window)
firf.report()

tick = Time.time()
for i_buffer in range(int(len(time)/BUFFER_SIZE)):
    # Simulate incoming buffers on the fly
    buffer_time = time[BUFFER_SIZE * i_buffer:
                       BUFFER_SIZE * (i_buffer + 1)]
    buffer_sig  = sig [BUFFER_SIZE * i_buffer:
                       BUFFER_SIZE * (i_buffer + 1)]
        
    # Simulate dropped buffer
    if 0:
        if i_buffer == 43 or i_buffer == 46:
            continue

    if not(firf.process(buffer_time, buffer_sig)):
        continue
        
    if f_SHOW_PLOT:
        color = 'r' if (i_buffer % 2 == 0) else 'g'
        ax1.plot(firf.valid_time_out, firf.valid_sig_out, marker + color)

print('\nsingle process step: %.2f ms' %
      ((Time.time() - tick) / (i_buffer + 1) * 1000))

# Quality check of band-gap filter, last window
idx_time_valid_last_start = np.where(time == firf.valid_time_out[0])[0][0]
idx_time_valid_last_end   = idx_time_valid_last_start + BUFFER_SIZE
idx_time_valid_last_range = np.arange(idx_time_valid_last_start,
                                      idx_time_valid_last_end)
residuals = (sig1[idx_time_valid_last_range] +
             sig3[idx_time_valid_last_range] -
             firf.valid_sig_out)
dev = np.std(residuals)

if f_SHOW_PLOT:
    # Plot vertical lines indicating each incoming buffer
    y_bars = np.array(plt.ylim()) * 1.1
    for i in range(int(len(time)/BUFFER_SIZE)):
        ax1.plot([i * time[BUFFER_SIZE], i * time[BUFFER_SIZE]], y_bars, 'k')

    plt.title("N_taps = %i, dev = %.3f" % (firf.N_taps, dev))
    plt.xlabel('time (s)')
    plt.ylabel('signal')
    plt.xlim([firf.T_settling, firf.T_settling + .4])
    plt.ylim([-2.5, 2.5])
    
    # Filter frequency response
    plt.sca(ax2)
    plt.plot(firf.resp_freq_Hz, firf.resp_ampl_dB, '.-b')
    plt.ylabel('amplitude (dB)')
    plt.xlabel('frequency (Hz)')
    plt.xlim([48, 52])
    
    plt.sca(ax3)
    plt.plot(firf.resp_freq_Hz, firf.resp_phase_rad/np.pi, '.-b')
    plt.ylabel('phase ($\pi$ rad)')
    plt.xlabel('frequency (Hz)')
    plt.xlim([0, Fs/2])
    
    plt.sca(ax1)
    thismanager = pylab.get_current_fig_manager()
    if hasattr(thismanager, 'window'):
        thismanager.window.setGeometry(500, 120, 1200, 700)
    plt.show()