#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
__author__      = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__         = "https://github.com/Dennis-van-Gils"
__date__        = "26-03-2019"
__version__     = "1.0.0"

from collections import deque
from scipy.signal import firwin, freqz, fftconvolve
import numpy as np
import time as Time

class Buffered_FIR_Filter():
    def __init__(self, buffer_size, N_buffers_in_deque, Fs, firwin_cutoff,
                 firwin_window, display_name=""):
        self.buffer_size = buffer_size                # [samples]
        self.N_buffers_in_deque = N_buffers_in_deque  # [int]
        self.Fs = Fs                                  # [Hz]
        self.firwin_window = firwin_window            # type of window, see scipy.signal.get_window        
        self.display_name = display_name              # [str]
        
        # Check firwin_cutoff for illegal values and cap when necessary
        self._constrain_firwin_cutoff(firwin_cutoff)  # list of [Hz]
       
        # Deque size
        self.N_deque = self.buffer_size * self.N_buffers_in_deque # [samples]
        
        # Calculate max number of possible taps [samples]. Must be an odd number
        # in order to create a zero-phase distortion filter, aka linear filter!
        self.N_taps = self.buffer_size * (self.N_buffers_in_deque - 1) + 1
        self.T_span_taps   = self.N_taps / self.Fs        # [s]
        self.T_span_buffer = self.buffer_size / self.Fs   # [s]
        
        # Indices within window corresponding to valid filter output
        self.win_idx_valid_start = int((self.N_taps - 1)/2)
        self.win_idx_valid_end   = self.N_deque - self.win_idx_valid_start
                
        # Keep track of filter settling
        # Note: 'T_settling' was named 'self.T_delay_valid_start'
        self.T_settling = self.win_idx_valid_start / self.Fs # [s]
        self.was_settled = False
        self.has_settled = False
        
        """ RAMBLINGS: rethink if I should implement this the following.
        Boolean 'self.starting_up' is used to distinguish deque buffers being
        populated from scratch, in contrast to 'was/has_settled' that are used
        to signal a mathematically valid settling time has been reached
        regardless of the number of samples in deque.
        #self.N_buffers_received = 0
        #self.starting_up = True
        """
        
        # Create filter
        self.b = firwin(self.N_taps,
                        self.firwin_cutoff,
                        window=self.firwin_window,
                        fs=self.Fs,
                        pass_zero=False)
        self.calc_freqz_response()

    def _constrain_firwin_cutoff(self, firwin_cutoff):
        """Check firwin_cutoff for illegal values and cap when necessary.
        I.e.:
        Frequencies <= 0 Hz will be set to the tiny value 'cutoff_grain'.
        Frequencies >= Nyquist freq. will be set to Nyquist - 'cutoff_grain'.
        cutoff_grain = 1e-6 Hz
        """
        cutoff_grain = 1e-6
        firwin_cutoff = np.array(firwin_cutoff, dtype=np.float64)
        firwin_cutoff[firwin_cutoff <= cutoff_grain] = cutoff_grain
        firwin_cutoff[firwin_cutoff >= self.Fs/2] = (self.Fs/2 - cutoff_grain)
        self.firwin_cutoff = firwin_cutoff

    def update_firwin_cutoff(self, firwin_cutoff):
        self._constrain_firwin_cutoff(firwin_cutoff)
        self.b = firwin(self.N_taps,
                        self.firwin_cutoff,
                        window=self.firwin_window,
                        fs=self.Fs,
                        pass_zero=False)
        self.calc_freqz_response()
        
    def process(self, deque_sig_in: deque):
        """
        """
        
        #print("%s: %i" % (self.display_name, len(deque_sig_in)))
        if len(deque_sig_in) < self.N_deque:
            # Start-up. Filter still needs time to settle.
            self.has_settled = False
            valid_out = np.array([np.nan] * self.buffer_size)
        else:
            self.has_settled = True
            """Select window out of the signal deque to feed into the
            convolution. By optimal design, this happens to be the full deque.
            Returns valid filtered signal output of current window.
            """
            #tick = Time.perf_counter()
            valid_out = fftconvolve(deque_sig_in, self.b, mode='valid')
            #print("%.1f" % ((Time.perf_counter() - tick)*1000))
        
        if self.has_settled and not(self.was_settled):
            #print("%s: Filter has settled" % self.display_name)
            self.was_settled = True
        elif not(self.has_settled) and self.was_settled:
            #print("%s: Filter has reset" % self.display_name)
            self.was_settled = False
        
        return valid_out
        
    def calc_freqz_response(self, worN=2**18):
        # Calculate full frequency response.
        # Note that these arrays will become of length 'worN', which could
        # overwhelm a user-interface when plotting such large number of points.
        w, h = freqz(self.b, worN=worN)
        self.full_resp_freq_Hz   = w / np.pi * self.Fs / 2
        self.full_resp_ampl_dB   = 20 * np.log10(abs(h))
        self.full_resp_phase_rad = np.unwrap(np.angle(h))/self.Fs
        
        # -------------------------------------------------
        #  Select region of interest for plotting later on
        # -------------------------------------------------
        # First flat-line all power below the dB floor
        dB_floor = -80
        idx_dB_floor = np.asarray(self.full_resp_ampl_dB<dB_floor).nonzero()[0]
        __ampl_dB = self.full_resp_ampl_dB
        __ampl_dB[idx_dB_floor] = dB_floor
        
        # Keep points on the curve with large absolute acceleration.
        # Will in effect simplify a 'boring' region to a linear curve.
        dAdF_2_threshold = 1e-4
        dAdF_2 = np.abs(np.diff(__ampl_dB, 2))
        idx_keep = np.asarray(dAdF_2 > dAdF_2_threshold).nonzero()[0]

        # Store region of interest
        if len(idx_keep) <= 1:
            self.resp_freq_Hz__ROI_start = 0
            self.resp_freq_Hz__ROI_end   = self.Fs / 2
        else:        
            self.resp_freq_Hz__ROI_start = self.full_resp_freq_Hz[idx_keep[0]]
            self.resp_freq_Hz__ROI_end   = self.full_resp_freq_Hz[idx_keep[-1]]
        
        # -------------------------------------------
        #  Lossy compress curves for faster plotting
        # -------------------------------------------        
        # Keep points on the curve with large absolute acceleration.
        # Will in effect simplify a 'boring' region to a linear curve.
        dAdF_2_threshold = 1e-6
        dAdF_2 = np.abs(np.diff(__ampl_dB, 2))
        idx_keep = np.asarray(dAdF_2 > dAdF_2_threshold).nonzero()[0]
        
        if len(idx_keep) <= 1:
            idx_keep = np.array([0, len(self.full_resp_ampl_dB) - 1])
            
        # Add back zero (first 2 points) and Nyquist frequency (last point)
        if not(idx_keep[0] == 0): idx_keep = np.insert(idx_keep, 0, 0)
        if not(idx_keep[1] == 1): idx_keep = np.insert(idx_keep, 1, 1)
        if not(idx_keep[-1] == (len(self.full_resp_ampl_dB) - 1)):
            idx_keep = np.append(idx_keep, len(self.full_resp_ampl_dB) - 1)
        
        # Store compressed curves
        self.resp_freq_Hz   = self.full_resp_freq_Hz[idx_keep]
        self.resp_ampl_dB   = self.full_resp_ampl_dB[idx_keep]
        self.resp_phase_rad = self.full_resp_phase_rad[idx_keep]
        
    def report(self):
        if isinstance(self.firwin_window, str):
            __window = '%s' % self.firwin_window
        else:
            __window = '%s' % [x for x in self.firwin_window]
        
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
        print('firwin_window = %s' % __window)
        print('--------------------------------')
        print('win_idx_valid_start = %i samples' % self.win_idx_valid_start)
        print('T_settling          = %.3f s'     % self.T_settling)
        print('--------------------------------')