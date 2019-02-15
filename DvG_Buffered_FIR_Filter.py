#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
__author__      = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__         = "https://github.com/Dennis-van-Gils"
__date__        = "13-02-2019"
__version__     = "1.0.0"

import numpy as np
from scipy import convolve  # TO DO: test np.convolve against scipy.convolve
from scipy.signal import firwin, freqz
from collections import deque

class DvG_Buffered_FIR_Filter():
    def __init__(self, buffer_size, N_buffers_in_deque, Fs, firwin_cutoff,
                 firwin_window):
        self.buffer_size = buffer_size                # [samples]
        self.N_buffers_in_deque = N_buffers_in_deque  # [int]
        self.Fs = Fs                                  # [Hz]
        self.firwin_cutoff = firwin_cutoff            # list of [Hz]
        self.firwin_window = firwin_window            # type of window, see scipy.signal.get_window        
        
        self.N_deque = self.buffer_size * self.N_buffers_in_deque # [samples]
        
        # Track the number of received buffers
        self.i_buffer = -1
        
        # Calculate max number of possible taps [samples]. Must be an odd number!
        self.N_taps = self.buffer_size * (self.N_buffers_in_deque - 1) + 1
        self.T_span_taps   = self.N_taps / self.Fs        # [s]
        self.T_span_buffer = self.buffer_size / self.Fs   # [s]
        
        # Indices within window corresponding to valid filter output
        self.win_idx_valid_start = int((self.N_taps - 1)/2)
        self.win_idx_valid_end   = self.N_deque - self.win_idx_valid_start
                
        # Settling time of the filter
        # Was named 'self.T_delay_valid_start'
        self.T_settling = self.win_idx_valid_start / self.Fs # [s]
    
        # Create filter
        self.b = firwin(self.N_taps,
                        self.firwin_cutoff,
                        window=self.firwin_window,
                        fs=self.Fs,
                        pass_zero=False)
        self.calc_freqz_response()
        
        # Init filtered valid output
        self.valid_time_out = np.array([np.nan] * self.buffer_size)
        self.valid_sig_out  = np.array([np.nan] * self.buffer_size)
        
    def process(self, deque_sig_in):
        """
        """
        self.i_buffer += 1
        
        if self.i_buffer < self.N_buffers_in_deque - 1:
            # Start-up. Filter still needs time to settle.
            #print("Filter is settling")
            return np.array([np.nan] * self.buffer_size)
        
        # Select window out of the signal deque to feed into the convolution.
        # By optimal design, this happens to be the full deque.
        #win_sig = np.array(deque_sig_in)
        
        # Valid filtered signal output of current window
        return np.convolve(deque_sig_in, self.b, mode='valid')
        #return convolve(deque_sig_in, self.b, mode='valid')
        
    def reset(self):
        self.i_buffer = -1
        
    def calc_freqz_response(self, worN=2**18):
        w, h = freqz(self.b, worN=worN)
        self.resp_freq_Hz   = w / np.pi * self.Fs / 2
        self.resp_ampl_dB   = 20 * np.log10(abs(h))
        self.resp_phase_rad = np.unwrap(np.angle(h))/self.Fs
    
    def report(self):
        print('--------------------------------')
        print('Fs                 = %.0f Hz'    % self.Fs)
        print('buffer_size        = %i samples' % self.buffer_size)
        print('N_buffers_in_deque = %i'         % self.N_buffers_in_deque)
        print('--------------------------------')
        print('N_taps        = %i samples' % self.N_taps)
        print('T_span_taps   = %.3f s'     % self.T_span_taps)
        print('T_span_buffer = %.3f s'     % self.T_span_buffer)
        print('--------------------------------')
        print('firwin_cutoff = %s' % [round(x, 1) for x in self.firwin_cutoff])
        print('firwin_window = %s' % [x for x in self.firwin_window])
        print('--------------------------------')
        print('win_idx_valid_start = %i samples' % self.win_idx_valid_start)
        print('T_settling          = %.3f s'     % self.T_settling)