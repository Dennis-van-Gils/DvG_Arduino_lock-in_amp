#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TODO: rename buffer_size into block_size, and buffer into block

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

    Use-case:
    Real-time data acquisition and processing, hence the use of a ringbuffer
    where timeseries data are continuously being appended to in chunks of size
    `buffer_size` (TODO: should be renamed block_size).

    The FIR filter output is (by definition) delayed with respect to the
    incoming timeseries data, namely by `T_settle_filter` seconds. Attribute
    `valid_slice` will contain the slice to be taken from the incoming
    ringbuffer corresponding to the matching time... bla... history

    The class name implies that the FIR filter takes in a `DvG_RingBuffer` class
    instance containing timeseries data. It does not mean that the FIR filter
    output is a ringbuffer. The ringbuffer is referenced as a 'deque` throughout
    this module.

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
        config: Config()
        display_name: str
        freqz: FreqResponse()
        was_deque_settled: bool
        has_deque_settled: bool
    """

    class Config:
        """In progress
        """

        # pylint: disable=too-many-instance-attributes, too-few-public-methods
        def __init__(self):
            self.Fs = None
            self.buffer_size = None
            self.N_buffers_in_deque = None
            self.cutoff = None
            self.window = None
            self.pass_zero = None
            self.use_CUDA = None

            # Derived parameters
            self.window_description = None
            self.N_deque = None
            self.N_taps = None
            self.T_span_taps = None
            self.T_span_buffer = None
            self.valid_deque_slice = None
            self.T_settle_filter = None
            self.T_settle_deque = None

    class FreqResponse:
        """Container for the calculated frequency response of the filter based
        on the output of :meth:`scipy.signal.freqz`.

        ROI: Region of interest
        """

        # pylint: disable=too-many-instance-attributes, too-few-public-methods
        def __init__(self):
            self.full_freq_Hz = None  # was `full_resp_freq_Hz`
            self.full_ampl_dB = None  # was `full_resp_ampl_dB`
            self.full_phase_rad = None  # was `full_resp_phase_rad`
            self.freq_Hz = None  # was `resp_freq_Hz`
            self.ampl_dB = None  # was `resp_ampl_dB`
            self.phase_rad = None  # was `resp_phase_rad`
            self.freq_Hz__ROI_start = None  # was `resp_freq_Hz__ROI_start`
            self.freq_Hz__ROI_end = None  # was `resp_freq_Hz__ROI_end`

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        Fs: float,
        buffer_size: int,
        N_buffers_in_deque: int,
        cutoff: Union[float, List[float]],
        window: Optional[Union[str, Tuple]] = "hamming",
        pass_zero: Optional[Union[bool, str]] = True,
        use_CUDA: Optional[bool] = False,
        display_name: Optional[str] = "",
    ):
        #  Passed config parameters
        # --------------------------
        c = self.Config()
        c.Fs = Fs
        c.buffer_size = buffer_size
        c.N_buffers_in_deque = N_buffers_in_deque
        c.cutoff = np.atleast_1d(cutoff)
        c.window = window
        c.pass_zero = pass_zero
        c.use_CUDA = use_CUDA

        #  Derived config parameters
        # ---------------------------
        c.window_description = (
            "%s" % window
            if isinstance(window, str)
            else "%s" % [x for x in window]
        )
        c.N_deque = c.buffer_size * c.N_buffers_in_deque  # [samples]

        # Calculate max number of possible taps. We'll force an odd number in
        # order to create a zero-phase distortion filter, aka linear filter.
        c.N_taps = c.buffer_size * (c.N_buffers_in_deque - 1) + 1  # [samples]
        c.T_span_taps = c.N_taps / c.Fs  # [s]
        c.T_span_buffer = c.buffer_size / c.Fs  # [s]

        # Indices within window corresponding to valid filter output
        idx_valid_start = int((c.N_taps - 1) / 2)
        idx_valid_end = c.N_deque - idx_valid_start
        c.valid_deque_slice = slice(idx_valid_start, idx_valid_end)

        # Settling times
        c.T_settle_filter = idx_valid_start / c.Fs  # [s]
        c.T_settle_deque = c.T_settle_filter * 2  # [s]

        #  Root level members
        # --------------------
        self.config = c
        self.display_name = display_name
        self.was_deque_settled = False
        self.has_deque_settled = False

        # Container for the calculated FIR filter tap array
        self._taps = None  # Used when `use_CUDA = False`
        self._taps_cupy = None  # Used when `use_CUDA = True`

        # Container for the calculated frequency response of the filter based
        # on the output of :meth:`scipy.signal.freqz`.
        self.freqz = self.FreqResponse()

        if not c.use_CUDA:
            # Create FFTW plans for convolution
            self._fftw_convolver = FFTW_Convolver_Valid1D(c.N_deque, c.N_taps)
        else:
            self.cupy = import_module("cupy")
            self.sigpy = import_module("sigpy")

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
        c = self.config  # Shorthand

        if cutoff is not None:
            c.cutoff = np.atleast_1d(cutoff)

        # Check cutoff frequencies for illegal values and cap when necessary,
        # sort by increasing value and remove duplicates.
        #   Frequencies <= 0 Hz will be removed
        #   Frequencies >= Nyquist will be set to Nyquist - 'cutoff_grain'
        cutoff_grain = 1e-6
        cutoff = c.cutoff
        cutoff = cutoff[cutoff > 0]
        cutoff[cutoff >= c.Fs / 2] = c.Fs / 2 - cutoff_grain
        c.cutoff = np.unique(np.sort(cutoff))

        if window is not None:
            c.window = window
            c.window_description = (
                "%s" % window
                if isinstance(window, str)
                else "%s" % [x for x in window]
            )

        if pass_zero is not None:
            c.pass_zero = pass_zero

        # Calculate the FIR filter tap array
        self._taps = firwin(
            numtaps=c.N_taps,
            cutoff=c.cutoff,
            window=c.window,
            pass_zero=c.pass_zero,
            fs=c.Fs,
        )

        if c.use_CUDA:
            # Copy FIR filter tap array from CPU to GPU memory.
            # Turning 1-D array into column vector by [:, None] for CUDA.
            self._taps_cupy = self.cupy.array(self._taps[:, None])

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
        c = self.config  # Shorthand
        w, h = freqz(self._taps, worN=worN)
        full_freq = w / np.pi * c.Fs / 2  # [Hz]
        full_ampl = 20 * np.log10(abs(h))  # [dB]
        full_phase = np.unwrap(np.angle(h)) / c.Fs  # [rad]

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
            freq_Hz__ROI_end = c.Fs / 2
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

        # ---------------
        #  Store results
        # ---------------

        self.freqz.full_freq_Hz = full_freq
        self.freqz.full_ampl_dB = full_ampl
        self.freqz.full_phase_rad = full_phase
        self.freqz.freq_Hz__ROI_start = freq_Hz__ROI_start
        self.freqz.freq_Hz__ROI_end = freq_Hz__ROI_end
        self.freqz.freq_Hz = full_freq[idx_keep]
        self.freqz.ampl_dB = ampl[idx_keep]
        self.freqz.phase_rad = full_phase[idx_keep]

    # --------------------------------------------------------------------------
    #   process, TODO: rename as `perform_filter`?
    # --------------------------------------------------------------------------

    def process(self, deque_sig_in: RingBuffer) -> np.ndarray:
        """Perform a convolution between the FIR filter tap array and the
        deque_sig_in array and return the valid convolution output. Will track
        if the filter has settled. Any NaNs in deque_sig_in will desettle the
        filter.

        Returns:
          The output as numpy.ndarray
        """
        c = self.config  # Shorthand

        # print("%s: %i" % (self.display_name, len(deque_sig_in)))
        if not deque_sig_in.is_full or np.isnan(deque_sig_in).any():
            # Start-up. Deque still needs time to settle.
            self.has_deque_settled = False
            valid_out = np.full(c.buffer_size, np.nan)
        else:
            self.has_deque_settled = True
            # Select window out of the signal deque to feed into the
            # convolution. By optimal design, this happens to be the full deque.
            # Returns valid filtered signal output of current window.

            # tick = Time.perf_counter()

            if not c.use_CUDA:
                # Perform convolution on the CPU
                valid_out = self._fftw_convolver.convolve(
                    deque_sig_in, self._taps
                )
            else:
                # Perform convolution on the GPU
                cp_valid_out = self.sigpy.convolve(
                    self.cupy.array(np.asarray(deque_sig_in)[:, None]),
                    # Turning 1-D array into column vector by using [:, None] as
                    # is needed for CUDA
                    self._taps_cupy,
                    mode="valid",
                )

                # Transfer result from GPU to CPU memory
                valid_out = self.cupy.asnumpy(cp_valid_out)

                # Reduce the dimension again
                valid_out = valid_out[:, 0]

            # print("%.1f" % ((Time.perf_counter() - tick)*1000))

        if self.has_deque_settled and not self.was_deque_settled:
            # print("%s: Deque has settled" % self.display_name)
            self.was_deque_settled = True
        elif not self.has_deque_settled and self.was_deque_settled:
            # print("%s: Deque has reset" % self.display_name)
            self.was_deque_settled = False

        return valid_out

    # --------------------------------------------------------------------------
    #   report
    # --------------------------------------------------------------------------

    def report(self):
        c = self.config  # Shorthand

        def f(name, value, value_format, unit=""):
            format_str = "{:>19s}  %s  {:<s}" % value_format
            print(format_str.format(name, value, unit))

        print("\nRingbuffer_FIR_Filter `%s`" % self.display_name)
        print("═" * 50)
        f("Fs", c.Fs, "{:>9,.2f}", "Hz")
        f("buffer_size", c.buffer_size, "{:>9d}", "samples")
        f("N_buffers_in_deque", c.N_buffers_in_deque, "{:>9d}")
        print("─" * 50)
        f("window", c.window_description, "{:<s}")
        f("cutoff", "%s" % [round(x, 1) for x in c.cutoff], "{:<s}", "Hz")
        f("pass_zero", str(c.pass_zero), "{:<s}")
        print("─" * 50)
        f("N_deque", c.N_deque, "{:>9d}", "samples")
        f("N_taps", c.N_taps, "{:>9d}", "samples")
        f("T_span_taps", c.T_span_taps, "{:>9.3f}", "s")
        f("T_span_buffer", c.T_span_buffer, "{:>9.3f}", "s")
        print("─" * 50)
        f("valid_deque_slice", str(c.valid_deque_slice), "{:<s}")
        f("T_settle_filter", c.T_settle_filter, "{:>9.3f}", "s")
        f("T_settle_deque", c.T_settle_deque, "{:>9.3f}", "s")
        print("═" * 50)

