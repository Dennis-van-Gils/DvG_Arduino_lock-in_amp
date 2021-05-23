#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

FIR: finite impulse response
rb: RingBuffer
Zero-phase distortion filter, aka linear filter
"""
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils"
__date__ = "23-05-2021"
__version__ = "1.0.0"
# pylint: disable=invalid-name, too-many-instance-attributes, too-few-public-methods, too-many-arguments

from typing import Optional, Union, List, Tuple
from collections import deque
from importlib import import_module

import numpy as np
from scipy.signal import firwin, freqz

from dvg_ringbuffer import RingBuffer
from dvg_fftw_convolver import FFTW_Convolver_Valid1D


class RingBuffer_FIR_Filter_Config:
    """In progress...

    You should not alter the class/object attributes from outside of this
    module!

    Args:
        Fs: float
            The sampling frequency of the input signal in Hz. Each frequency in
            `firwin_cutoff` must be between 0 and ``Fs/2``.

            See :meth:`scipy.signal.firwin` for more details.

        block_size: int
            The fixed number of samples of one incoming block of signal data.

        N_blocks: int
            Number of blocks that make up a full ring buffer.

        firwin_cutoff: float or 1-D array_like
            Cutoff frequency of the filter in Hz OR an array of cutoff
            frequencies (that is, band edges). In the latter case, the
            frequencies in `firwin_cutoff` should be positive and monotonically
            increasing between 0 and `Fs/2`. The values 0 and `Fs/2` must
            not be included in `firwin_cutoff`.

            See :meth:`scipy.signal.firwin` for more details.

        firwin_window: string or tuple of string and parameter values, optional
            Desired window to use.

            See :meth:`scipy.signal.get_window` for a list of windows and
            required parameters.

            Default: "hamming"

        firwin_pass_zero: {True, False, 'bandpass', 'lowpass', 'highpass', 'bandstop'}, optional
            If True, the gain at the frequency 0 (i.e., the "DC gain") is 1.
            If False, the DC gain is 0. Can also be a string argument for the
            desired filter type (equivalent to ``btype`` in IIR design
            functions).

            See :meth:`scipy.signal.firwin` for more details.

            Default: True

    Parameters for :meth:`scipy.signal.firwin`
    Parameters for :meth:`scipy.signal.freqz`
    """

    def __init__(
        self,
        Fs: float,
        block_size: int,
        N_blocks: int,
        firwin_cutoff: Union[float, List[float]],
        firwin_window: Union[str, Tuple] = "hamming",
        firwin_pass_zero: Union[bool, str] = True,
        freqz_worN: int = 2 ** 18,
        freqz_dB_floor: float = -120.0,
    ):
        self.Fs = Fs
        self.block_size = block_size
        self.N_blocks = N_blocks
        self.firwin_cutoff = np.atleast_1d(firwin_cutoff)
        self.firwin_window = firwin_window
        self.firwin_pass_zero = firwin_pass_zero
        self.freqz_worN = freqz_worN
        self.freqz_dB_floor = freqz_dB_floor

        #  Derived config parameters
        # ---------------------------
        # Full capacity of ring buffer
        self.rb_capacity = block_size * N_blocks

        # Informational: String description of window settings
        self.firwin_window_descr = (
            "%s" % firwin_window
            if isinstance(firwin_window, str)
            else "%s" % [x for x in firwin_window]
        )

        # Calculate max number of possible taps. We'll force an odd number
        # in order to create a zero-phase distortion filter, aka linear
        # filter.
        self.firwin_numtaps = block_size * (N_blocks - 1) + 1

        # Indices within the input-signal ring buffer corresponding to the
        # time stamps of the computed valid filter output
        idx_rb_valid_start = int((self.firwin_numtaps - 1) / 2)
        self.rb_valid_slice = slice(
            idx_rb_valid_start, self.rb_capacity - idx_rb_valid_start
        )

        # Informational: Time periods
        # fmt: off
        self.T_span_taps     = self.firwin_numtaps / Fs           # [s]
        self.T_span_block    = block_size / Fs                    # [s]
        self.T_settle_filter = (N_blocks - 1) * block_size / Fs   # [s]
        # fmt: on


class RingBuffer_FIR_Filter:
    """In progress...

    Use-case:
    Real-time data acquisition and processing, hence the use of a ring buffer
    where timeseries data are continuously being appended to in chunks of size
    `block_size`

    The FIR filter output is (by definition) delayed with respect to the
    incoming timeseries data, namely by `T_settle_filter` seconds. Attribute
    `valid_slice` will contain the slice to be taken from the incoming
    ring buffer corresponding to the matching time... bla... history

    The class name implies that the FIR filter takes in a `DvG_RingBuffer` class
    instance containing timeseries data. It does not mean that the FIR filter
    output is a ring buffer.

    Args:
        config: RingBuffer_FIR_Filter_Config(...)
            See :class:`RingBuffer_FIR_Filter_Config`

        name: str, optional

            Default: ""

        use_CUDA: bool, optional
            Use NVidia's CUDA acceleration for the 1D FFT convolution? You'll
            need `cupy` and `sigpy` properly installed in your system for CUDA
            to work. Only beneficial when processing large amounts of data
            (approx. > 1e5 samples) as the overhead of copying memory from CPU
            to GPU is substantial. Not really worth it when block_size <= 500
            and N_blocks <= 41.

            Default: False

    Attributes:
        config: RingBuffer_FIR_Filter_Config()
        name: str
        freqz: FreqResponse()

        filter_has_settled: bool
            True when the filter starts outputting, not to be confused with the
            filter theoretical response time to an impulse.
    """

    class FreqResponse:
        """Container for the computed theoretical frequency response of the
        filter based on the output of :meth:`scipy.signal.freqz`.

        ROI: Region of interest
        """

        def __init__(self):
            self.full_freq_Hz = None
            self.full_ampl_dB = None
            self.full_phase_rad = None
            self.freq_Hz__ROI_start = None
            self.freq_Hz__ROI_end = None
            self.freq_Hz = None
            self.ampl_dB = None
            self.phase_rad = None

    def __init__(
        self,
        config: RingBuffer_FIR_Filter_Config,
        name: str = "",
        use_CUDA: bool = False,
    ):
        self.config = config
        self.name = name
        self.use_CUDA = use_CUDA
        self.filter_has_settled = False
        self._filter_was_settled = False

        # Container for the computed FIR filter tap array
        self._taps = None
        self._taps_cupy = None  # Only used when `use_CUDA = True`

        # Container for the computed frequency response of the filter based
        # on the output of :meth:`scipy.signal.freqz`.
        self.freqz = self.FreqResponse()

        if not self.use_CUDA:
            # Create FFTW plan for FFT convolution
            self._fftw_convolver = FFTW_Convolver_Valid1D(
                self.config.rb_capacity, self.config.firwin_numtaps
            )
        else:
            self.cupy = import_module("cupy")
            self.sigpy = import_module("sigpy")

        self.compute_firwin_and_freqz()

    # --------------------------------------------------------------------------
    #   compute_firwin_and_freqz
    # --------------------------------------------------------------------------

    def compute_firwin_and_freqz(
        self,
        firwin_cutoff: Optional[Union[float, List[float]]] = None,
        firwin_window: Optional[Union[str, Tuple]] = None,
        firwin_pass_zero: Optional[Union[bool, str]] = None,
        freqz_worN: Optional[int] = None,
        freqz_dB_floor: Optional[float] = None,
    ):
        """Compute the FIR filter tap array and the frequency response `freqz`.

        For each of the input arguments:
            When set to None or not supplied, the value stored in `config` will
            be used. When supplied, it will use the new value and update
            `config`.

        Args:
            See :class:`Ringbuffer_FIR_Filter.Config`.
        """
        c = self.config  # Shorthand

        #  Check input arguments
        # -----------------------
        if firwin_cutoff is not None:
            c.firwin_cutoff = np.atleast_1d(firwin_cutoff)

        # Check cutoff frequencies for illegal values and cap when necessary,
        # sort by increasing value and remove duplicates.
        #   Frequencies <= 0 Hz will be removed
        #   Frequencies >= Nyquist will be set to Nyquist - 'cutoff_grain'
        cutoff_grain = 1e-6
        cutoff = c.firwin_cutoff
        cutoff = cutoff[cutoff > 0]
        cutoff[cutoff >= c.Fs / 2] = c.Fs / 2 - cutoff_grain
        c.firwin_cutoff = np.unique(np.sort(cutoff))

        if firwin_window is not None:
            c.firwin_window = firwin_window
            c.firwin_window_descr = (
                "%s" % firwin_window
                if isinstance(firwin_window, str)
                else "%s" % [x for x in firwin_window]
            )

        if firwin_pass_zero is not None:
            c.firwin_pass_zero = firwin_pass_zero

        if freqz_worN is not None:
            c.freqz_worN = freqz_worN

        if freqz_dB_floor is not None:
            c.freqz_dB_floor = freqz_dB_floor

        #  Compute the FIR filter tap array
        # ----------------------------------
        self._taps = firwin(
            numtaps=c.firwin_numtaps,
            cutoff=c.firwin_cutoff,
            window=c.firwin_window,
            pass_zero=c.firwin_pass_zero,
            fs=c.Fs,
        )

        if self.use_CUDA:
            # Copy FIR filter tap array from CPU to GPU memory
            self._taps_cupy = self.cupy.array(self._taps[:, None])
            # Turning 1-D array `_taps` into column vector by [:, None]

        #  Compute the frequency response
        # --------------------------------
        self._compute_freqz(worN=c.freqz_worN, dB_floor=c.freqz_dB_floor)
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
        w, h = freqz(self._taps, worN=worN)
        Fs = self.config.Fs
        full_freq = w / np.pi * Fs / 2  # [Hz]
        full_ampl = 20 * np.log10(abs(h))  # [dB]
        full_phase = np.unwrap(np.angle(h)) / Fs  # [rad]

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
            freq_Hz__ROI_end = Fs / 2
        else:
            freq_Hz__ROI_start = full_freq[idx_keep[0]]
            freq_Hz__ROI_end = full_freq[idx_keep[-1]]

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
        if not idx_keep[0] == 0:
            idx_keep = np.insert(idx_keep, 0, 0)

        if not idx_keep[1] == 1:
            idx_keep = np.insert(idx_keep, 1, 1)

        if not idx_keep[-1] == (len(full_ampl) - 1):
            idx_keep = np.append(idx_keep, len(full_ampl) - 1)

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
    #   apply_filter
    # --------------------------------------------------------------------------

    def apply_filter(
        self, ringbuffer_in: Union[RingBuffer, deque]
    ) -> np.ndarray:
        """Apply the currently set FIR filter to the incoming `ringbuffer_in`
        data and return the filter output. I.e., perform a convolution between
        the FIR filter tap array and the `ringbuffer_in` array and keep only the
        valid convolution output.

        Using a `dvg_ringbuffer::RingBuffer` instead of a `collections.deque`
        is way faster.

        Will track if the filter has settled. Any NaNs in `ringbuffer_in` will
        desettle the filter. If `ringbuffer_in` is not yet fully populated with
        data, the filter will return an array filled with NaNs.

        Returns:
          The filter output as numpy.ndarray
        """
        c = self.config  # Shorthand
        # print("%s: %i" % (self.name, len(ringbuffer_in)))

        is_full = (
            len(ringbuffer_in) == ringbuffer_in.maxlen
            if isinstance(ringbuffer_in, deque)
            else ringbuffer_in.is_full
        )
        self.filter_has_settled = is_full and not np.isnan(ringbuffer_in).any()
        # Note: Keep the second more cpu-intensive check `isnan` at last

        if self.filter_has_settled:
            # tick = Time.perf_counter()
            if not self.use_CUDA:
                # Perform convolution on the CPU
                valid_out = self._fftw_convolver.convolve(
                    ringbuffer_in, self._taps
                )
            else:
                # Perform convolution on the GPU
                valid_out_cupy = self.sigpy.convolve(
                    self.cupy.array(np.asarray(ringbuffer_in)[:, None]),
                    # Turning 1-D array into column vector by [:, None]
                    self._taps_cupy,
                    mode="valid",
                )

                # Transfer result from GPU to CPU memory
                valid_out = self.cupy.asnumpy(valid_out_cupy)

                # Reduce the dimension again
                valid_out = valid_out[:, 0]

            # print("%.1f" % ((Time.perf_counter() - tick)*1000))
        else:
            valid_out = np.full(c.block_size, np.nan)

        if self.filter_has_settled and not self._filter_was_settled:
            # print("%s: Filter has settled" % self.name)
            self._filter_was_settled = True
        elif not self.filter_has_settled and self._filter_was_settled:
            # print("%s: Filter has reset" % self.name)
            self._filter_was_settled = False

        return valid_out

    # --------------------------------------------------------------------------
    #   report
    # --------------------------------------------------------------------------

    def report(self):
        c = self.config  # Shorthand

        def f(name, value, value_format, unit=""):
            format_str = "{:>19s}  %s  {:<s}" % value_format
            print(format_str.format(name, value, unit))

        print("\nRingbuffer_FIR_Filter `%s`" % self.name)
        print("═" * 50)
        f("Fs", c.Fs, "{:>9,.2f}", "Hz")
        f("block_size", c.block_size, "{:>9d}", "samples")
        f("N_blocks", c.N_blocks, "{:>9d}")
        print("─" * 50)
        f("firwin_window", c.firwin_window_descr, "{:<s}")
        f(
            "firwin_cutoff",
            "%s" % [round(x, 1) for x in c.firwin_cutoff],
            "{:<s}",
            "Hz",
        )
        f("firwin_pass_zero", str(c.firwin_pass_zero), "{:<s}")
        print("─" * 50)
        f("rb_capacity", c.rb_capacity, "{:>9d}", "samples")
        f("firwin_numtaps", c.firwin_numtaps, "{:>9d}", "samples")
        f("T_span_taps", c.T_span_taps, "{:>9.3f}", "s")
        f("T_span_block", c.T_span_block, "{:>9.3f}", "s")
        print("─" * 50)
        f("rb_valid_slice", str(c.rb_valid_slice), "{:<s}")
        f("T_settle_filter", c.T_settle_filter, "{:>9.3f}", "s")
        print("═" * 50)

