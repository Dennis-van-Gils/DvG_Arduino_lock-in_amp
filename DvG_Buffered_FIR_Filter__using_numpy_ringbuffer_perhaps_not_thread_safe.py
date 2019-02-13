#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uses numpy_ringbuffer which is potentially not thread-safe!
"""
__author__      = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__         = "https://github.com/Dennis-van-Gils"
__date__        = "13-02-2019"
__version__     = "1.0.0"

import numpy as np
from scipy.signal import firwin, freqz
#from collections import deque
from numpy_ringbuffer import RingBuffer

class DvG_Buffered_FIR_Filter():
    def __init__(self, buffer_size, N_buffers_in_deque, Fs, firwin_cutoff,
                 firwin_window):
        self.buffer_size = buffer_size                # [samples]
        self.N_buffers_in_deque = N_buffers_in_deque  # [int]
        self.Fs = Fs                                  # [Hz]
        self.firwin_cutoff = firwin_cutoff            # list of [Hz]
        self.firwin_window = firwin_window            # type of window, see scipy.signal.get_window
        
        # Create deque buffers
        self.N_deque = self.buffer_size * self.N_buffers_in_deque # [samples]
        # Explicitly keep track of the time stamps to prevent troubles with
        # possibly dropped buffers
        self.deque_time = RingBuffer(capacity=self.N_deque, dtype=np.float)
        self.deque_sig  = RingBuffer(capacity=self.N_deque, dtype=np.float)
        
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
        
    def process(self, buffer_time_in, buffer_sig_in):
        """
        
        Returns True when the filter has settled and has computed valid filtered
        output data, False otherwise.
        
        Valid filtered output data is stored in:
            [DvG_Buffered_FIR_Filter.valid_time_out] and
            [DvG_Buffered_FIR_Filter.valid_sig_out]
        """
        self.i_buffer += 1
        prev_last_deque_time = (self.deque_time[-1] if self.i_buffer > 0 else
                                np.nan)
        
        # Detect dropped buffers
        dT = buffer_time_in[0] - prev_last_deque_time
        if dT > (1/self.Fs)*1.01:  # Allow 1 percent clock jitter
            print("dropped buffer %i" % self.i_buffer)
            N_dropped_buffers = int(round(dT / self.T_span_buffer))
            print("N_dropped_buffers %i" % N_dropped_buffers)
            
            # Replace dropped buffers with ...
            N_dropped_samples = self.buffer_size * N_dropped_buffers
            self.deque_time.extend(prev_last_deque_time +
                                   np.arange(1, N_dropped_samples + 1) *
                                   1/self.Fs)
            if 1:
                """ Proper: with np.nan samples.
                As a result, the filter output will contain a continuous series
                of np.nan values in the output for up to T_settling seconds long
                after the occurance of the last dropped buffer.
                """
                self.deque_sig.extend(np.array([np.nan] * N_dropped_samples))
            else:
                """ Improper: with linearly interpolated samples.
                As a result, the filter output will contain fake data where
                ever dropped buffers occured. The advantage is that, in contrast
                to using above proper technique, the filter output remains a
                continuous series of values.
                """
                self.deque_sig.extend(self.deque_sig[-1] +
                                      np.arange(1, N_dropped_samples + 1) *
                                      (buffer_sig_in[0] - self.deque_sig[-1]) /
                                      N_dropped_samples)

        self.deque_time.extend(buffer_time_in)
        self.deque_sig.extend(buffer_sig_in)
        
        if self.i_buffer < self.N_buffers_in_deque - 1:
            # Start-up. Filter still needs time to settle.
            self.valid_time_out = np.array([np.nan] * self.buffer_size)
            self.valid_sig_out  = np.array([np.nan] * self.buffer_size)
            return False
        
        # Select window out of the signal deque to feed into the convolution.
        # By optimal design, this happens to be the full deque.
        win_sig = np.array(self.deque_sig)
        
        # Valid filtered signal output of current window
        self.valid_sig_out  = np.convolve(win_sig, self.b, mode='valid')
        self.valid_time_out = (np.array(self.deque_time)
                               [self.win_idx_valid_start:
                                self.win_idx_valid_end])
        return True
    
    def reset(self):
        self.i_buffer = -1
        self.valid_time_out = np.array([np.nan] * self.buffer_size)
        self.valid_sig_out  = np.array([np.nan] * self.buffer_size)
        
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