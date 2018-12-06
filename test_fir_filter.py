# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 21:22:33 2018

@author: vangi
"""

import numpy as np
from scipy.signal import firwin, iirfilter, filtfilt
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['lines.linewidth'] = 2.0

# Original data series
Fs = 1;                         # [Hz] sampling frequency
time = np.arange(400)           # [s] time
sig = np.sin(2*np.pi*.01*time) + np.sin(2*np.pi*.1*time)

plt.plot(time, sig, 'k')
  
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

class FilterFIR():
    def __init__(self, b):
        self.b = b
        self.v = np.zeros(len(b))
    
    def step(self, x):
        self.v[0] = self.v[1];
        self.v[1] = self.v[2];
        self.v[2] = (self.b[0] * x +
                     self.b[1] * self.v[0] +
                     self.b[2] * self.v[1])
		
        return (self.v[0] + self.v[2] + 2* self.v[1])

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
#plt.plot(time - correct_phase_lag, sig_HP, 'm')
#plt.plot(time - correct_phase_lag, sig_LP + sig_HP, 'g')
    
plt.plot(time - correct_phase_lag, sig_LP2, 'g')
plt.plot(time - correct_phase_lag, sig_LP3, 'b')
plt.plot(time - correct_phase_lag, sig_LP4, 'y')

# -----------------------------------------
#   FIR filters
# -----------------------------------------

ntaps = 99;
b_LP = firwin(ntaps, [0.0001, 0.05], width=0.05, fs=Fs, pass_zero=False)
b_HP = firwin(ntaps, 0.05, width=0.05, fs=Fs, pass_zero=False)

for i in range(2):
    if i == 0:
        time_sel = time[0:200]
        sig_sel  = sig[0:200]
    elif i == 1:
        time_sel = time[200:400]
        sig_sel  = sig[200:400]
    
    sig_LP = np.convolve(sig_sel, b_LP, mode='same')
    sig_HP = np.convolve(sig_sel, b_HP, mode='same')
    
    #plt.plot(time_sel, sig_LP, 'g')
    #plt.plot(time_sel, sig_HP, 'm')    
    #plt.plot(time_sel, sig_LP+sig_HP, 'g')

# -----------------------------------------
#   IIR filters
# -----------------------------------------

ntaps = 99;
a_LP, b_LP = iirfilter(6, Wn=0.05, btype='lowpass', analog=False, ftype='butter')
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
    
    #plt.plot(time_sel, sig_LP, 'b')
    #plt.plot(time_sel, sig_HP, 'm')    
    #plt.plot(time_sel, sig_LP+sig_HP, 'g')

# -----------------------------------------
# -----------------------------------------

plt.xlim([120, 280])
plt.show()