#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
__author__      = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__         = "https://github.com/Dennis-van-Gils"
__date__        = "21-04-2019"
__version__     = "1.0.0"

from collections import deque
from scipy.signal import firwin, freqz, fftconvolve
import numpy as np
import time as Time

# When True enables 1D FFT-convolution running on a NVidia GPU using CUDA.
# Only beneficial when processing large amounts of data (approx. > 1e5 samples)
# as the overhead of copying memory from CPU to GPU is substantial.
# Not really worth it when buffer_size <= 500 and N_buffers_in_deque <= 41.
USE_CUDA_ACCELERATION = False
if USE_CUDA_ACCELERATION:
    import cupy
    import sigpy as sp

class Buffered_FIR_Filter():
    def __init__(self, buffer_size, N_buffers_in_deque, Fs, cutoff, window,
                 pass_zero=True, display_name=""):
        self.buffer_size = buffer_size                # [samples]
        self.N_buffers_in_deque = N_buffers_in_deque  # [int]
        self.Fs = Fs                                  # [Hz]
        self.cutoff = cutoff                          # list of [Hz], see scipy.signal.firwin
        self.window = window                          # type of window, see scipy.signal.get_window        
        self.pass_zero = pass_zero                    # [bool], see scipy.signal.firwin
        self.display_name = display_name              # [str]
                
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
                
        self.T_settle_filter = self.win_idx_valid_start / self.Fs # [s]
        self.T_settle_deque  = self.T_settle_filter * 2           # [s]
        self.deque_was_settled = False
        self.deque_has_settled = False
        
        # Friendly window description for direct printing as string
        if isinstance(self.window, str):
            self.window_description = '%s' % self.window
        else:
            self.window_description = '%s' % [x for x in self.window]
        
        # Compute the filter
        self.compute_firwin()
        
    def compute_firwin(self, cutoff=None, window=None, pass_zero=None):
        """Check cutoff frequencies for illegal values and cap when necessary,
        compute the FIR filter and the frequency response.
        """
        if cutoff is not None:
            self.cutoff = cutoff
        if window is not None:
            self.window = window
            if isinstance(self.window, str):
                self.window_description = '%s' % self.window
            else:
                self.window_description = '%s' % [x for x in self.window]
        if pass_zero is not None:
            self.pass_zero = pass_zero
            
        self._constrain_cutoff()
        self.b = firwin(numtaps=self.N_taps,
                        cutoff=self.cutoff,
                        window=self.window,
                        pass_zero=self.pass_zero,
                        fs=self.Fs)
        self.compute_freqz()
        #self.report()
        
        if USE_CUDA_ACCELERATION:
            # Copy to cupy array (resides at the GPU!)
            self.b_cp = cupy.array(self.b)
            
    def compute_freqz(self, worN=2**18):
        # Compute the full frequency response.
        # Note: these arrays will become of length 'worN', which could overwhelm
        # a user-interface when plotting such large number of points.
        # Note: Amplitude ratio in dB: 20 log_10(A1/A2)
        #       Power     ratio in dB: 10 log_10(P1/P2)
        w, h = freqz(self.b, worN=worN)
        self.full_resp_freq_Hz   = w / np.pi * self.Fs / 2
        self.full_resp_ampl_dB   = 20 * np.log10(abs(h))
        self.full_resp_phase_rad = np.unwrap(np.angle(h))/self.Fs
        
        # -------------------------------------------------
        #  Select region of interest for plotting later on
        # -------------------------------------------------
        # First flat-line all power below the dB floor
        dB_floor = -120
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

    def _constrain_cutoff(self):
        """Check cutoff frequencies for illegal values and cap when necessary,
        sort by increasing value and remove duplicates.
        I.e.:
        Frequencies <= 0 Hz will be removed.
        Frequencies >= Nyquist freq. will be set to Nyquist - 'cutoff_grain'.
        cutoff_grain = 1e-6 Hz
        """
        cutoff_grain = 1e-6
        cutoff = np.array(self.cutoff, dtype=np.float64, ndmin=1)
        cutoff = cutoff[cutoff > 0]
        cutoff[cutoff >= self.Fs/2] = (self.Fs/2 - cutoff_grain)
        self.cutoff = np.unique(np.sort(cutoff))
        
    def process(self, deque_sig_in: deque):
        """
        """
        
        #print("%s: %i" % (self.display_name, len(deque_sig_in)))
        if (len(deque_sig_in) < self.N_deque) or np.isnan(deque_sig_in).any():
            # Start-up. Deque still needs time to settle.
            self.deque_has_settled = False
            valid_out = np.array([np.nan] * self.buffer_size)
        else:
            self.deque_has_settled = True
            """Select window out of the signal deque to feed into the
            convolution. By optimal design, this happens to be the full deque.
            Returns valid filtered signal output of current window.
            """
            #tick = Time.perf_counter()
            if USE_CUDA_ACCELERATION:
                cp_valid_out = sp.convolve(cupy.array(list(deque_sig_in)),
                                           self.b_cp,
                                           mode='valid')
                valid_out = cupy.asnumpy(cp_valid_out)
            else:
                valid_out = fftconvolve(deque_sig_in, self.b, mode='valid')
            #print("%.1f" % ((Time.perf_counter() - tick)*1000))
        
        if self.deque_has_settled and not(self.deque_was_settled):
            #print("%s: Deque has settled" % self.display_name)
            self.deque_was_settled = True
        elif not(self.deque_has_settled) and self.deque_was_settled:
            #print("%s: Deque has reset" % self.display_name)
            self.deque_was_settled = False
        
        return valid_out
        
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
        print('window = %s' % self.window_description)
        print('cutoff = %s' % [round(x, 1) for x in self.cutoff])
        print('pass_zero = %s' % self.pass_zero)
        print('--------------------------------')
        print('win_idx_valid_start = %i samples' % self.win_idx_valid_start)
        print('T_settle_filter     = %.3f s'     % self.T_settle_filter)
        print('T_settle_deque      = %.3f s'     % self.T_settle_deque)
        print('--------------------------------')