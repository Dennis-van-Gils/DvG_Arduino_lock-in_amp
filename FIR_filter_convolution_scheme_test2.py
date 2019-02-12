# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 21:22:33 2018

@author: vangi
"""

"""
TO DO: 13-02-2019

Create class 'DvG_FIR_Filter'
attributes:
    T_delay_valid_out
methods:
    create(FIR filter settings, Fs)
    valid_data_out = process(buffer_in)
    (freq_table, attenuation) = examine_response()
"""


import numpy as np
from scipy.signal import firwin
from collections import deque

import matplotlib as mpl
#mpl.use("qt5agg")
import matplotlib.pyplot as plt
import pylab
mpl.rcParams['lines.linewidth'] = 3.0
marker = '-'

# Original data series
Fs = 10000;          # [Hz] sampling frequency
total_time = 4;      # [s]
time = np.arange(0, total_time, 1/Fs) # [s]

sig1_ampl    = 1
sig1_freq_Hz = 48.5

sig2_ampl    = 1
sig2_freq_Hz = 50

sig3_ampl    = 0
sig3_freq_Hz = 137

sig1 = sig1_ampl * np.sin(2*np.pi*time * sig1_freq_Hz)
sig2 = sig2_ampl * np.sin(2*np.pi*time * sig2_freq_Hz)
sig3 = sig3_ampl * np.sin(2*np.pi*time * sig3_freq_Hz)
sig = sig1 + sig2 + sig3

#np.random.seed(0)
#sig = sig + np.random.randn(len(sig))

#plt.plot(time, sig1, marker, color='k')
#plt.plot(time, sig2, marker, color='k')
#plt.plot(time, sig2, marker, color='k')
#plt.plot(time, sig, marker, color=('0.8'))
plt.plot(time, sig1+sig3, marker, color=('0.8'))

# -----------------------------------------
#   FIR filters
# -----------------------------------------
# Optimal scenario:
#   Perform convolve every incoming buffer
#   Maximize N_taps given N_buffers_in_deque
BUFFER_SIZE = 500  # [samples]
N_buffers_in_deque = 41

# Create deque buffers
N_deque    = BUFFER_SIZE * N_buffers_in_deque
deque_time = deque(maxlen=N_deque)
deque_sig  = deque(maxlen=N_deque)

# Calculate max number of possible taps [samples]. Must be an odd number!
N_taps = BUFFER_SIZE * (N_buffers_in_deque - 1) + 1

# Indices within window corresponding to valid filter output
win_idx_valid_start = int((N_taps - 1)/2)
win_idx_valid_end   = N_deque - win_idx_valid_start
T_delay_valid_start = win_idx_valid_start/Fs

T_span_buffer = BUFFER_SIZE / Fs
T_span_taps   = N_taps / Fs
print('Fs                 = %.0f Hz'    % Fs)
print('BUFFER_SIZE        = %i samples' % BUFFER_SIZE)
print('N_buffers_in_deque = %i'         % N_buffers_in_deque)
print('--------------------------------')
print('T_span_buffer = %.3f s'     % T_span_buffer)
print('N_taps        = %i samples' % N_taps)
print('T_span_taps   = %.3f s'     % T_span_taps)
#print('tap resolution = %.3f Hz' % (Fs / N_taps))
print('win_idx_valid_start = %i samples' % win_idx_valid_start)
print('T_delay_valid_start = %.3f s'     % T_delay_valid_start)

# Create tap filters for upcoming convolution
window = "hamming"
b_LP = firwin(N_taps, 105       , window=window, fs=Fs)
b_HP = firwin(N_taps, 750       , window=window, fs=Fs, pass_zero=False)
b_BP = firwin(N_taps, [496, 504], window=window, fs=Fs, pass_zero=False)
b_BG = firwin(N_taps, ([0.1, 49, 51, Fs/2-0.1]), window=window, fs=Fs, pass_zero=False)

for i_window in range(int(len(time)/BUFFER_SIZE)):
    # Simulate incoming buffers on the fly
    buffer_time = time[BUFFER_SIZE * i_window:
                       BUFFER_SIZE * (i_window + 1)]
    buffer_sig  = sig [BUFFER_SIZE * i_window:
                       BUFFER_SIZE * (i_window + 1)]
    deque_time.extend(buffer_time)
    deque_sig.extend(buffer_sig)
    
    if i_window < N_buffers_in_deque - 1:
        # Start-up
        continue
    
    # Select window out of the signal deque to feed into the convolution.
    # By optimal design, this happens to be the full deque.
    win_sig = np.array(deque_sig)
    
    # Filtered signal output of current window
    win_sig_LP = np.convolve(win_sig, b_LP, mode='valid')
    win_sig_BP = np.convolve(win_sig, b_BP, mode='valid')
    win_sig_BG = np.convolve(win_sig, b_BG, mode='valid')
    
    # Fetch the time stamps correspondig to the valid filtered signal
    win_time_valid = np.array(deque_time)[win_idx_valid_start:win_idx_valid_end]
    
    # Plot
    color = 'r' if (i_window % 2 == 0) else 'g'
    #plt.plot(win_valid_time, win_sig_LP, marker + color)
    #plt.plot(win_valid_time, win_sig_BP, marker + color)
    plt.plot(win_time_valid, win_sig_BG, marker + color)

# Plot vertical lines indicating each incoming buffer
y_bars = np.array(plt.ylim()) * 1.1
for i in range(int(len(time)/BUFFER_SIZE)):
    plt.plot([i * time[BUFFER_SIZE], i * time[BUFFER_SIZE]], y_bars, 'k')

# Quality check of band-gap filter, last window
idx_time_valid_last_start = np.where(time == win_time_valid[0])[0][0]
idx_time_valid_last_end   = idx_time_valid_last_start + BUFFER_SIZE
idx_time_valid_last_range = np.arange(idx_time_valid_last_start,
                                      idx_time_valid_last_end)
residuals = (sig1[idx_time_valid_last_range] +
             sig3[idx_time_valid_last_range] -
             win_sig_BG)
dev = np.std(residuals)

plt.title("N_taps = %i, dev = %.3f" % (N_taps, dev))
#plt.xlim([T_delay_valid_start, T_delay_valid_start + .2])
plt.xlim([1, 1.2])
plt.ylim([-2.5, 2.5])

thismanager = pylab.get_current_fig_manager()
if hasattr(thismanager, 'window'):
    thismanager.window.setGeometry(500, 120, 1200, 700)
plt.show()