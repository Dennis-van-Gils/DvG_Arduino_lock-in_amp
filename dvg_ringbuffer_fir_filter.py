#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

FIR: finite impulse response
Zero-phase distortion filter, aka linear filter
"""
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils"
__date__ = "22-05-2021"
__version__ = "1.0.0"
# pylint: disable=invalid-name

from typing import Optional, Union, List, Tuple
from importlib import import_module

import numpy as np
from scipy.signal import firwin, freqz

from dvg_ringbuffer import RingBuffer
from dvg_fftw_convolver import FFTW_Convolver_Valid1D


class Ringbuffer_FIR_Filter:
    """Ringbuffer_FIR_Filter description...

    Acts upon a `DvG_RingBuffer` class instance, which we call incorrectly a
    'deque` in this module.

    Args:
        buffer_size: int
            The fixed number of samples of one incoming data buffer.

        N_buffers_in_deque: int
            Number of incoming data buffers to make up a full ringbuffer, aka
            'deque'.

        Fs: float
            The sampling frequency of the signal in Hz. Each frequency in
            `cutoff` must be between 0 and ``Fs/2``.

            See :meth:`scipy.signal.firwin` for more details.

        cutoff: float or 1-D array_like
            Cutoff frequency of the filter in Hz OR an array of cutoff
            frequencies (that is, band edges). In the latter case, the
            frequencies in `cutoff` should be positive and monotonically
            increasing between 0 and `Fs/2`. The values 0 and `Fs/2` must
            not be included in `cutoff`.

            See :meth:`scipy.signal.firwin` for more details.

        window: string or tuple of string and parameter values, optional
            Desired window to use.

            See :meth:`scipy.signal.get_window` for a list of windows and
            required parameters.

            Default: "hamming"

        pass_zero: {True, False, 'bandpass', 'lowpass', 'highpass', 'bandstop'}, optional
            If True, the gain at the frequency 0 (i.e., the "DC gain") is 1.
            If False, the DC gain is 0. Can also be a string argument for the
            desired filter type (equivalent to ``btype`` in IIR design
            functions).

            See :meth:`scipy.signal.firwin` for more details.

            Default: True

        display_name: str, optional

            Default: ""

        use_CUDA: bool, optional
            Use NVidia's CUDA acceleration for the 1D FFT convolution? You'll
            need `cupy` and `sigpy` properly installed in your system for CUDA
            to work. Only beneficial when processing large amounts of data
            (approx. > 1e5 samples) as the overhead of copying memory from CPU
            to GPU is substantial. Not really worth it when buffer_size <= 500
            and N_buffers_in_deque <= 41.

            Default: False

    Attributes:
        window_description:
        T_settle_filter:
        T_settle_deque:
        was_deque_settled:
        has_deque_settled:
        freqz: FreqResponse()
    """

    class FreqResponse:
        """Container for the calculated frequency response of the filter based
        on the output of :meth:`scipy.signal.freqz`.

        ROI: Region of interest
        """

        # pylint: disable=too-many-instance-attributes, too-few-public-methods
        def __init__(self):
            self.full_freq_Hz = np.nan  # was `full_resp_freq_Hz`
            self.full_ampl_dB = np.nan  # was `full_resp_ampl_dB`
            self.full_phase_rad = np.nan  # was `full_resp_phase_rad`
            self.freq_Hz = np.nan  # was `resp_freq_Hz`
            self.ampl_dB = np.nan  # was `resp_ampl_dB`
            self.phase_rad = np.nan  # was `resp_phase_rad`
            self.freq_Hz__ROI_start = np.nan  # was `resp_freq_Hz__ROI_start`
            self.freq_Hz__ROI_end = np.nan  # was `resp_freq_Hz__ROI_end`

    class Config:
        """In progress
        """

        def __init__(self):
            self.buffer_size = np.nan
            self.N_buffers_in_deque = np.nan
            self.Fs = np.nan
            self.cutoff = np.nan
            self.window = "hamming"
            self.pass_zero = True
            self.use_CUDA = False

            # Calculated based on above
            self.window_description = ""
            self.N_deque = np.nan
            self.N_taps = np.nan
            self.T_span_taps = np.nan
            self.T_span_buffer = np.nan
            self.valid_slice = slice(0)
            self.T_settle_filter = np.nan
            self.T_settle_deque = np.nan

    def __init__(
        self,
        buffer_size: int,
        N_buffers_in_deque: int,
        Fs: float,
        cutoff: Union[float, List[float]],
        window: Optional[Union[str, Tuple]] = "hamming",
        pass_zero: Optional[Union[bool, str]] = True,
        display_name: Optional[str] = "",
        use_CUDA: Optional[bool] = False,
    ):
        self.buffer_size = buffer_size
        self.N_buffers_in_deque = N_buffers_in_deque
        self.Fs = Fs
        self.cutoff = np.atleast_1d(cutoff)
        self.window = window
        self.pass_zero = pass_zero
        self.display_name = (
            display_name  # TODO: Keep as member, do not put in Config()
        )
        self.use_CUDA = use_CUDA

        # Friendly window description
        if isinstance(self.window, str):
            self.window_description = "%s" % self.window
        else:
            self.window_description = "%s" % [x for x in self.window]

        # Deque size
        self.N_deque = self.buffer_size * self.N_buffers_in_deque  # [samples]

        # Calculate max number of possible taps [samples]. Must be an odd number
        # in order to create a zero-phase distortion filter, aka linear filter!
        self.N_taps = self.buffer_size * (self.N_buffers_in_deque - 1) + 1
        self.T_span_taps = self.N_taps / self.Fs  # [s]
        self.T_span_buffer = self.buffer_size / self.Fs  # [s]

        # Indices within window corresponding to valid filter output
        # TODO: instead of storing `win_idx_valid_start` and `win_idx_valid_end`
        # store the valid slice instead, like:
        # valid_slice = slice(win_idx_valid_start, win_idx_valid_end)
        self.win_idx_valid_start = int((self.N_taps - 1) / 2)
        self.win_idx_valid_end = self.N_deque - self.win_idx_valid_start

        self.T_settle_filter = self.win_idx_valid_start / self.Fs  # [s]
        self.T_settle_deque = self.T_settle_filter * 2  # [s]

        # TODO: Keep as members, do not put in Config()
        self.was_deque_settled = False
        self.has_deque_settled = False

        # Calculated FIR filter tap array
        self.b = None  # Used when `use_CUDA = False`
        self.b_cp = None  # Used when `use_CUDA = True`

        # Container for the calculated frequency response of the filter based
        # on the output of :meth:`scipy.signal.freqz`.
        self.freqz = self.FreqResponse()

        if self.use_CUDA:
            self.cupy = import_module("cupy")
            self.sigpy = import_module("sigpy")
        else:
            # Create FFTW plans for fft convolution
            self.fftw_convolver = FFTW_Convolver_Valid1D(
                self.N_deque, self.N_taps
            )

        # Compute the FIR filter tap array
        # TODO: perhaps remove the method call and have the user make the call
        # instead. Benefits: user can alter `freq_worN` and `freqz_dB_floor`.
        self.compute_firwin()

    # --------------------------------------------------------------------------
    #   compute_firwin
    # --------------------------------------------------------------------------

    def compute_firwin(
        self,
        cutoff: Optional[Union[float, List[float]]] = None,
        window: Optional[Union[str, Tuple]] = None,
        pass_zero: Optional[Union[bool, str]] = None,
        freqz_worN: Optional[int] = 2 ** 18,
        freqz_dB_floor: Optional[float] = -120,
    ):
        """Compute the FIR filter tap array and the frequency response `freqz`.

        Args:
            cutoff: float or 1-D array_like, optional
                See :class:`Ringbuffered_FIR_Filter`. When set to None or not
                supplied, the last set `cutoff` value will be used.

            window: string or tuple of string and parameter values, optional
                See :class:`Ringbuffered_FIR_Filter`. When set to None or not
                supplied, the last set `window` value will be used.

            pass_zero: {True, False, 'bandpass', 'lowpass', 'highpass', 'bandstop'}, optional
                See :class:`Ringbuffered_FIR_Filter`. When set to None or not
                supplied, the last set `pass_zero` value will be used.

            freqz_worN: int
                Bla... # TODO

            freqz_dB_floor: float
                Decibel (noise-)floor bla... # TODO
        """
        if cutoff is not None:
            self.cutoff = np.atleast_1d(cutoff)

        # Check cutoff frequencies for illegal values and cap when necessary,
        # sort by increasing value and remove duplicates.
        #   Frequencies <= 0 Hz will be removed
        #   Frequencies >= Nyquist will be set to Nyquist - 'cutoff_grain'
        cutoff_grain = 1e-6
        cutoff = self.cutoff
        cutoff = cutoff[cutoff > 0]
        cutoff[cutoff >= self.Fs / 2] = self.Fs / 2 - cutoff_grain
        self.cutoff = np.unique(np.sort(cutoff))

        if window is not None:
            self.window = window

            # Friendly window description
            if isinstance(self.window, str):
                self.window_description = "%s" % self.window
            else:
                self.window_description = "%s" % [x for x in self.window]

        if pass_zero is not None:
            self.pass_zero = pass_zero

        # Calculate the FIR filter tap array
        self.b = firwin(
            numtaps=self.N_taps,
            cutoff=self.cutoff,
            window=self.window,
            pass_zero=self.pass_zero,
            fs=self.Fs,
        )

        if self.use_CUDA:
            # Copy FIR filter tap array from CPU to GPU memory.
            # Turning 1-D array into column vector by [:, None] for CUDA.
            self.b_cp = self.cupy.array(self.b[:, None])

        # Calculate the frequency response
        self._compute_freqz(worN=freqz_worN, dB_floor=freqz_dB_floor)

        # self.report()

    # --------------------------------------------------------------------------
    #   _compute_freqz
    # --------------------------------------------------------------------------

    def _compute_freqz(self, worN: int, dB_floor: float):
        """Compute the full frequency response and store it in class member
        `freqz`.

        Note: The full result arrays will become of length `worN`, which could
        overwhelm a user-interface when plotting so many points. Hence, we will
        also calculate a 'lossy compressed' dataset: `freq_Hz`, `ampl_dB`, and
        `phase_rad`, useful for faster and less-memory hungry plotting.

        Note: Amplitude ratio in dB: 20 log_10(A1/A2)
              Power     ratio in dB: 10 log_10(P1/P2)
        """
        w, h = freqz(self.b, worN=worN)
        full_freq = w / np.pi * self.Fs / 2  # [Hz]
        full_ampl = 20 * np.log10(abs(h))  # [dB]
        full_phase = np.unwrap(np.angle(h)) / self.Fs  # [rad]

        # ---------------------------------
        #  Select region of interest (ROI)
        # ---------------------------------

        # First flat-line all power below the dB floor to simplify the ROI
        ampl = full_ampl
        ampl[np.asarray(full_ampl < dB_floor).nonzero()[0]] = dB_floor

        # Keep points on the curve with large absolute acceleration. Will in
        # effect simplify a 'boring' region to a linear curve.
        dAdF_2_threshold = 1e-4
        dAdF_2 = np.abs(np.diff(ampl, 2))
        idx_keep = np.asarray(dAdF_2 > dAdF_2_threshold).nonzero()[0] + 1

        if len(idx_keep) <= 1:
            freq_Hz__ROI_start = 0
            freq_Hz__ROI_end = self.Fs / 2
        else:
            freq_Hz__ROI_start = full_freq[idx_keep[0]]
            freq_Hz__ROI_end = full_freq[idx_keep[-1]]

        # -------------------------------------------
        #  Lossy compress curves for faster plotting
        # -------------------------------------------

        # Keep points on the curve with large absolute acceleration. Will in
        # effect simplify a 'boring' region to a linear curve.
        dAdF_2_threshold = 1e-6
        dAdF_2 = np.abs(np.diff(ampl, 2))
        idx_keep = np.asarray(dAdF_2 > dAdF_2_threshold).nonzero()[0] + 1

        if len(idx_keep) <= 1:
            idx_keep = np.array([0, len(full_ampl) - 1])

        # Add back DC (first 2 points) and Nyquist frequency (last point)
        if not (idx_keep[0] == 0):
            idx_keep = np.insert(idx_keep, 0, 0)

        if not (idx_keep[1] == 1):
            idx_keep = np.insert(idx_keep, 1, 1)

        if not (idx_keep[-1] == (len(full_ampl) - 1)):
            idx_keep = np.append(idx_keep, len(full_ampl) - 1)

        # -------
        #  Store
        # -------

        self.freqz.full_freq_Hz = full_freq
        self.freqz.full_ampl_dB = full_ampl
        self.freqz.full_phase_rad = full_phase

        self.freqz.freq_Hz__ROI_start = freq_Hz__ROI_start
        self.freqz.freq_Hz__ROI_end = freq_Hz__ROI_end

        self.freqz.freq_Hz = full_freq[idx_keep]
        self.freqz.ampl_dB = ampl[idx_keep]
        self.freqz.phase_rad = full_phase[idx_keep]

    # --------------------------------------------------------------------------
    #   process
    # --------------------------------------------------------------------------

    def process(self, deque_sig_in: RingBuffer) -> np.ndarray:
        """Perform a convolution between the FIR filter tap array and the
        deque_sig_in array and return the valid convolution output. Will track
        if the filter has settled. Any NaNs in deque_sig_in will desettle the
        filter.
        """

        # print("%s: %i" % (self.display_name, len(deque_sig_in)))
        if (not deque_sig_in.is_full) or np.isnan(deque_sig_in).any():
            # Start-up. Deque still needs time to settle.
            self.has_deque_settled = False
            valid_out = np.full(self.buffer_size, np.nan)
        else:
            self.has_deque_settled = True
            # Select window out of the signal deque to feed into the
            # convolution. By optimal design, this happens to be the full deque.
            # Returns valid filtered signal output of current window.

            # tick = Time.perf_counter()
            if self.use_CUDA:
                # Perform convolution on the GPU

                # Turning 1-D array into column vector by [:, None] for CUDA
                cp_valid_out = self.sigpy.convolve(
                    self.cupy.array(np.asarray(deque_sig_in)[:, None]),
                    self.b_cp,
                    mode="valid",
                )

                # Transfer result from GPU to CPU memory
                valid_out = self.cupy.asnumpy(cp_valid_out)

                # Reduce the dimension again
                valid_out = valid_out[:, 0]
            else:
                # Perform convolution on the CPU
                valid_out = self.fftw_convolver.convolve(deque_sig_in, self.b)
            # print("%.1f" % ((Time.perf_counter() - tick)*1000))

        if self.has_deque_settled and not (self.was_deque_settled):
            # print("%s: Deque has settled" % self.display_name)
            self.was_deque_settled = True
        elif not (self.has_deque_settled) and self.was_deque_settled:
            # print("%s: Deque has reset" % self.display_name)
            self.was_deque_settled = False

        return valid_out

    # --------------------------------------------------------------------------
    #   report
    # --------------------------------------------------------------------------

    def report(self):
        print("Buffered_FIR_Filter: %s" % self.display_name)
        print("--------------------------------")
        print("Fs                 = %.0f Hz" % self.Fs)
        print("buffer_size        = %i samples" % self.buffer_size)
        print("N_buffers_in_deque = %i" % self.N_buffers_in_deque)
        print("--------------------------------")
        print("N_taps        = %i samples" % self.N_taps)
        print("T_span_taps   = %.3f s" % self.T_span_taps)
        print("T_span_buffer = %.3f s" % self.T_span_buffer)
        print("--------------------------------")
        print("window = %s" % self.window_description)
        print("cutoff = %s" % [round(x, 1) for x in self.cutoff])
        print("pass_zero = %s" % self.pass_zero)
        print("--------------------------------")
        print("win_idx_valid_start = %i samples" % self.win_idx_valid_start)
        print("T_settle_filter     = %.3f s" % self.T_settle_filter)
        print("T_settle_deque      = %.3f s" % self.T_settle_deque)
        print("--------------------------------")

