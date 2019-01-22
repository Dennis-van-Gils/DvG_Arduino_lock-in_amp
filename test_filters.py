# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 21:22:33 2018

@author: vangi
"""

import numpy as np
from scipy.signal import firwin, iirfilter, filtfilt
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['lines.linewidth'] = 3.0

# Original data series
Fs = 10000;          # [Hz] sampling frequency
total_time = .5;     # [s]
time = np.arange(0, total_time - 1/Fs, 1/Fs) # [s]

sig1_ampl    = 1
sig1_freq_Hz = 10

sig2_ampl    = 1
sig2_freq_Hz = 50

sig3_ampl    = 0
sig3_freq_Hz = 100

sig = (sig1_ampl * np.sin(2*np.pi*time * sig1_freq_Hz) + 
       sig2_ampl * np.sin(2*np.pi*time * sig2_freq_Hz) + 
       sig3_ampl * np.sin(2*np.pi*time * sig3_freq_Hz))
np.random.seed(0)
#sig = sig + np.random.randn(len(sig))

plt.plot(time, sig, ('0.8'))

# -----------------------------------------
#   FIR filters
# -----------------------------------------

BLOCK_SIZE = 500
NTAPS = 201;

num_valid = BLOCK_SIZE - NTAPS + 1
offset_valid = int((NTAPS - 1)/2)

b_LP = firwin(NTAPS, [0.0001, 0.05], width=0.05, fs=Fs, pass_zero=False)
b_HP = firwin(NTAPS, 0.05          , width=0.05, fs=Fs, pass_zero=False)
b_BP = firwin(NTAPS, [0.019, 0.021], width=0.01, fs=Fs, pass_zero=False)
b_BG = firwin(NTAPS, ([0.0001, 0.015, 0.025, .5-1e-4]), width=0.005, fs=Fs, pass_zero=False)

iter_no = 0;
while True:
    idx_start = iter_no * num_valid
    idx_end   = idx_start + BLOCK_SIZE
    #idx_start = iter_no * BLOCK_SIZE
    #idx_end   = idx_start + BLOCK_SIZE
    
    if idx_end > len(time):
        break
    color = 'r' if (iter_no % 2 == 0) else 'g'
    iter_no += 1
    
    sig_LP = np.convolve(sig[idx_start:idx_end], b_LP, mode='valid')
    sig_HP = np.convolve(sig[idx_start:idx_end], b_HP, mode='valid')
    sig_BP = np.convolve(sig[idx_start:idx_end], b_BP, mode='valid')
    sig_BG = np.convolve(sig[idx_start:idx_end], b_BG, mode='valid')
    
    time_sel_valid = time[idx_start + offset_valid:
                          idx_start + offset_valid + num_valid]
    
    plt.plot(time_sel_valid, sig_HP, 'y')
    plt.plot(time_sel_valid, sig_LP, color)
    #plt.plot(time_sel_valid, sig_LP + sig_HP, 'k')
    #plt.plot(time_sel_valid, sig_BP, 'r')
    #plt.plot(time_sel_valid, sig_BG, 'm')
    #plt.plot(time_sel_valid, sig_BP + sig_BG, 'k')

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
ntaps=3
b_LP = firwin(ntaps, [0.0001, 0.05], width=0.05, fs=Fs, pass_zero=False)
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
for i in range(np.mod(len(time), BLOCK_SIZE)):
    plt.plot([i * time[BLOCK_SIZE - 1], i * time[BLOCK_SIZE - 1]], y_bars, 'k')
plt.xlim([0, time[5 * BLOCK_SIZE]])
plt.show()