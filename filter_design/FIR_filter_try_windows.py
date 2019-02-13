# -*- coding: utf-8 -*-
"""
Dennis van Gils
13-02-2019
"""

import numpy as np
from scipy.signal import firwin, freqz

import matplotlib as mpl
#mpl.use("qt5agg")
import matplotlib.pyplot as plt
import pylab
mpl.rcParams['lines.linewidth'] = 2.0
marker = '.-'

# Original data series
Fs = 10000;          # [Hz] sampling frequency
N_taps = 40001

# Indices within window corresponding to valid filter output
win_idx_valid_start = int((N_taps - 1)/2)
T_delay_valid_start = win_idx_valid_start/Fs

# Create tap filters for upcoming convolution
windows = ("hamming", "hanning", "blackman", "blackmanharris", ("chebwin", 50))
firwin_cutoff = ([0.0001, 49.5, 50.5, Fs/2-0.0001])

plt.figure(1)
ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)
for i in range(len(windows)):
    window = windows[i]
    b_BG = firwin(N_taps, firwin_cutoff, window=window, fs=Fs, pass_zero=False)

    # Filter frequency response
    w, h = freqz(b_BG, worN = 2**18)
    freq_table_Hz = w / np.pi * Fs / 2

    ax1.plot(freq_table_Hz, 20 * np.log10(abs(h)), marker, label=window)
    ax2.plot(freq_table_Hz, 20 * np.log10(abs(h)), marker, label=window)

plt.sca(ax1)
plt.title('Fs = %.0f Hz, N_taps = %i, T_delay = %.1f s\ncutoff = %s' %
          (Fs,
           N_taps,
           T_delay_valid_start,
           [round(x, 1) for x in firwin_cutoff]))
plt.ylabel('amplitude (dB)')
plt.xlabel('frequency (Hz)')
plt.xlim([48, 52])
plt.ylim([-110, 1])
plt.grid()
plt.legend()

plt.sca(ax2)
plt.ylabel('amplitude (dB)')
plt.xlabel('frequency (Hz)')
plt.xlim([50.5, 54])
plt.ylim([-.4, .1])
plt.grid()
#plt.legend()

thismanager = pylab.get_current_fig_manager()
if hasattr(thismanager, 'window'):
    thismanager.window.setGeometry(500, 120, 1200, 900)

plt.savefig('FIR_try_Fs_%i_N_%i_cutoff_%s.png' %
            (Fs,
             N_taps,
             [round(x, 1) for x in firwin_cutoff]),
            dpi=300)
plt.show()