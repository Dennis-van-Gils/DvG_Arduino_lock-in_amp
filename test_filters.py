# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 21:22:33 2018

@author: vangi
"""

import numpy as np
from scipy.signal import firwin, iirfilter, filtfilt
from collections import deque

import matplotlib as mpl
mpl.use("qt5agg")
import matplotlib.pyplot as plt
import pylab
mpl.rcParams['lines.linewidth'] = 3.0
marker = '-'

# Original data series
Fs = 10000;          # [Hz] sampling frequency
total_time = .4;     # [s]
time = np.arange(0, total_time, 1/Fs) # [s]

sig1_ampl    = 1
sig1_freq_Hz = 100

sig2_ampl    = 1
sig2_freq_Hz = 500

sig3_ampl    = 0
sig3_freq_Hz = 100

sig1 = sig1_ampl * np.sin(2*np.pi*time * sig1_freq_Hz)
sig2 = sig2_ampl * np.sin(2*np.pi*time * sig2_freq_Hz)
sig3 = sig3_ampl * np.sin(2*np.pi*time * sig3_freq_Hz)
sig = sig1 + sig2 + sig3

#np.random.seed(0)
#sig = sig + np.random.randn(len(sig))

plt.plot(time, sig1, marker, color='k')
plt.plot(time, sig2, marker, color='k')
plt.plot(time, sig, marker, color=('0.8'))

# -----------------------------------------
#   FIR filters
# -----------------------------------------

BUFFER_SIZE = 500  # [samples]
N_taps = 491       # [samples] Use an odd number!


BUFFER_TIME = BUFFER_SIZE / Fs
print('buffer_time = %.3f s' % BUFFER_TIME)
print('min freq.   = %.3f Hz' % (1/BUFFER_TIME)) # TO DO: figure out correctly

N_shift_buffers = 2
maxlen = N_shift_buffers * BUFFER_SIZE
hist_time = deque(maxlen=maxlen)
hist_sig  = deque(maxlen=maxlen)

N_valid = BUFFER_SIZE - N_taps + 1 # [samples]
offset_valid = int((N_taps - 1)/2)

f_LP = 110
width_LP = 10
b_LP = firwin(N_taps, f_LP, width=width_LP, fs=Fs)
b_HP = firwin(N_taps, 0.05          , width=0.05, fs=Fs, pass_zero=False)
b_BP = firwin(N_taps, [0.019, 0.021], width=0.01, fs=Fs, pass_zero=False)
b_BG = firwin(N_taps, ([0.0001, 0.015, 0.025, .5-1e-4]), width=0.005, fs=Fs, pass_zero=False)

for i_window in range(int(len(time)/BUFFER_SIZE)):
    # Simulate incoming buffers on the fly
    buffer_time = time[BUFFER_SIZE * i_window:
                       BUFFER_SIZE * (i_window + 1)]
    buffer_sig  = sig[BUFFER_SIZE * i_window:
                      BUFFER_SIZE * (i_window + 1)]
        
    hist_time.extend(buffer_time)
    hist_sig.extend(buffer_sig)
    
    if i_window < N_shift_buffers - 1:
        # Start-up
        continue
    
    conv_in_selection_sig = np.array(hist_sig)[N_valid:]
    
    sig_LP = np.convolve(conv_in_selection_sig, b_LP, mode='valid')
    
    idx_valid_start = BUFFER_SIZE - offset_valid
    idx_valid_end   = maxlen - offset_valid
    
    sel_valid_time = np.array(hist_time)[idx_valid_start:idx_valid_end]
    
    color = 'r' if (i_window % 2 == 0) else 'g'
    #nudge = 0 if (i_window % 2 == 0) else 0.05
    nudge = 0
    plt.plot(sel_valid_time, sig_LP + nudge, marker + color)
    
"""
# -----------------------------------------
#   IIR filters
# -----------------------------------------

a_LP, b_LP = iirfilter(6, Wn=0.05, btype='lowpass' , analog=False, ftype='butter')
a_HP, b_HP = iirfilter(6, Wn=0.05, btype='highpass', analog=False, ftype='butter')

for i in range(2):
    if i == 0:
        time_sel = time[0:200]
        sig_sel  = sig[0:200]
    elif i == 1:
        time_sel = time[200:400]
        sig_sel  = sig[200:400]
    
    sig_LP = filtfilt(a_LP, b_LP, sig_sel, method='pad')
    sig_HP = filtfilt(a_HP, b_HP, sig_sel, method='pad')
    
    plt.plot(time_sel, sig_LP, 'b')
    #plt.plot(time_sel, sig_HP, 'y')    
    #plt.plot(time_sel, sig_LP+sig_HP, 'g')

# -----------------------------------------
# -----------------------------------------

# -----------------------------------------
#   Continuous discrete filter
#   http://www.schwietering.com/jayduino/filtuino/
# -----------------------------------------

class FilterBuLp2():
    def __init__(self):
        self.v = np.array([0.0, 0.0, 0.0])
    
    def step(self, x):
        self.v[0] = self.v[1];
        self.v[1] = self.v[2];
        self.v[2] = (2.008336556421122521e-2 * x +
                     -0.64135153805756306422 * self.v[0] +
                     1.56101807580071816339 * self.v[1])
        
        return (self.v[0] + self.v[2] + 2 * self.v[1])
    
class FilterBuHp2():
    def __init__(self):
        self.v = np.array([0.0, 0.0, 0.0])
    
    def step(self, x):
        self.v[0] = self.v[1];
        self.v[1] = self.v[2];
        self.v[2] = (8.005924034645703902e-1 * x +
                     -0.64135153805756306422 * self.v[0] +
                     1.56101807580071816339 * self.v[1])
        
        return (self.v[0] + self.v[2] - 2 * self.v[1])
    
def array_map(fun, x):
    return np.array(list(map(fun, x)))

def array_for_N(fun, N, x):
    out = np.zeros(len(x));
    for i in range(len(x)):
        n = 1
        out[i] = fun(x[i])
        while n < N:
            out[i] = fun(out[i])
            n += 1
        
    return out

#https://stackoverflow.com/questions/40483518/how-to-real-time-filter-with-scipy-and-lfilter
N_taps=3
b_LP = firwin(N_taps, [0.0001, 0.05], width=0.05, fs=Fs, pass_zero=False)
filterFIR = FilterFIR(b_LP)
sig_LP4 = array_map(filterFIR.step, sig)

filterBuLp2 = FilterBuLp2()
filterBuHp2 = FilterBuHp2()
        
sig_LP = array_map(filterBuLp2.step, sig)
sig_LP = array_map(filterBuLp2.step, sig_LP)
sig_LP = array_map(filterBuLp2.step, sig_LP)

sig_HP = array_map(filterBuHp2.step, sig)
sig_HP = array_map(filterBuHp2.step, sig_HP)
sig_HP = array_map(filterBuHp2.step, sig_HP)

sig_LP2 = array_for_N(filterBuLp2.step, 1, sig)
sig_LP3 = array_map(filterBuLp2.step, sig)

correct_phase_lag = 13.5
plt.plot(time - correct_phase_lag, sig_LP, 'r')
#plt.plot(time - correct_phase_lag, sig_HP, 'y')
#plt.plot(time - correct_phase_lag, sig_LP + sig_HP, 'g')
    
plt.plot(time - correct_phase_lag, sig_LP2, 'g')
plt.plot(time - correct_phase_lag, sig_LP3, 'b')
plt.plot(time - correct_phase_lag, sig_LP4, 'y')
"""

y_bars = np.array(plt.ylim()) * 1.1
for i in range(int(len(time)/BUFFER_SIZE)):
    plt.plot([i * time[BUFFER_SIZE], i * time[BUFFER_SIZE]], y_bars, 'k')

plt.title("f_LP = %i, width = %.3f" % (f_LP, width_LP))
plt.xlim([0.125, 0.14])
plt.ylim([-1.02, 1.02])

thismanager = pylab.get_current_fig_manager()
thismanager.window.setGeometry(500, 120, 1200, 700)
plt.show()