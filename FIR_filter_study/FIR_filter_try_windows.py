# -*- coding: utf-8 -*-
"""
Dennis van Gils
25-04-2019
"""

import numpy as np
from scipy.signal import firwin, freqz

import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 2.0
#mpl.use("qt5agg")
import matplotlib.pyplot as plt
import pylab

# Create tap filters for upcoming convolution
windows = ("hamming", "hanning", "blackman", "blackmanharris", ("chebwin", 50))

#       Fs [Hz], N_taps, pass_zero, cutoff [Hz]
tests = [[ 5000, 20001 , True     , [218]],
         [ 5000, 40001 , True     , [218]],
         [10000, 20001 , True     , [218]],
         [10000, 40001 , True     , [218]],
         [10000, 80001 , True     , [218]],
         [ 5000, 20001 , True     , [49  , 51]],
         [ 5000, 20001 , True     , [49.5, 50.5]],
         [ 5000, 40001 , True     , [49  , 51]],
         [ 5000, 40001 , True     , [49.5, 50.5]],
         [10000, 20001 , True     , [49  , 51]],
         [10000, 40001 , True     , [49  , 51]],
         [10000, 40001 , True     , [49.5, 50.5]],
         [10000, 80001 , True     , [49.5, 50.5]]]

# Prepare plot
plt.figure(1)
ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)
marker = '.-'

thismanager = pylab.get_current_fig_manager()
if hasattr(thismanager, 'window'):
    thismanager.window.setGeometry(500, 120, 1200, 900)

for test in tests:
    Fs        = test[0]
    N_taps    = test[1]
    pass_zero = test[2]
    cutoff    = test[3]

    # Indices within window corresponding to valid filter output
    win_idx_valid_start = int((N_taps - 1)/2)
    T_delay_valid_start = win_idx_valid_start/Fs
    
    plt.figure(1)
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)

    for window in windows:
        b_BG = firwin(N_taps, cutoff, window=window, fs=Fs, pass_zero=pass_zero)
    
        # Filter frequency response
        w, h = freqz(b_BG, worN = 2**18)
        freq_table_Hz = w / np.pi * Fs / 2
    
        ax1.plot(freq_table_Hz, 20 * np.log10(abs(h)), marker, label=window)
        ax2.plot(freq_table_Hz, 20 * np.log10(abs(h)), marker, label=window)
    
    plt.sca(ax1)
    plt.title('Fs = %.0f Hz, N_taps = %i, T_settle = %.1f s\n'
              'pass_zero = %s, cutoffs = %s' %
              (Fs, N_taps, T_delay_valid_start,
               pass_zero, [round(x, 1) for x in cutoff]))
    plt.ylabel('ampl. attn. (dB)')
    plt.xlabel('frequency (Hz)')
    if cutoff[0] < 50:
        plt.xlim([48, 52])
    else:
        plt.xlim([217, 221])
    plt.ylim([-130, 1])
    plt.grid()
    plt.legend()
    
    plt.sca(ax2)
    plt.ylabel('ampl. attn. (dB)')
    plt.xlabel('frequency (Hz)')
    if cutoff[0] < 50:
        plt.xlim([50.5, 54])
        plt.ylim([-.4, .1])
    else:
        plt.xlim([216, 218])
        plt.ylim([-.2, .1])
    plt.grid()
    #plt.legend()
    
    plt.savefig('FIR_try_cutoff_%s_Fs_%i_N_%i.png' %
                ([round(x, 1) for x in cutoff], Fs, N_taps), dpi=300)
    plt.show()