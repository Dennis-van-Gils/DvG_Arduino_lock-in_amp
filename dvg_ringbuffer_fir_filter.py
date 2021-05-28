#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides class `RingBuffer_FIR_Filter` that configures and performs a
lightning-fast finite-impulse response (FIR) filter on, typically, 1D time
series data acquired at a fixed sampling frequency.

The time series data to be fed in should originate from a ring buffer, either a
`collections.deque()` or a `dvg_ringbuffer.DvG_RingBuffer()` instance. The use
of a `deque` is supported, but not recommended as it is ~60 times slower than a
`DvG_RingBuffer`. The latter is purposefully made to get the maximum performance
gain in conjunction with the `fftw` library.

The typical use-case for this `RingBuffer_FIR_Filter` class is a real-time data
acquisition program where blocks of time series data of length `block_size` are
appended to the ring buffer for each data-acquisition interval. The FIR filter
will maximize the number of taps that will fit inside the passed ring buffer
while resulting in an 1D filtered output array of also length `block_size`. This
filtered output could then, in turn, be easily taken up in another ring buffer
(outside of this module) and get processed during the current data-acquisition
interval that already is operating on chunks of data with length `block_size`.

The FIR filter algorithm uses convolution based on the fast-Fourier transform
(FFT). The FFT can be configured to get performed on either the CPU or the GPU.

When on the CPU (default), it will use the excellent `fftw`
(http://www.fftw.org) library. It will plan the transformations ahead of time to
optimize the calculations. Also, multiple threads can be specified for the FFT
and, when set to > 1, the Python GIL will not be invoked. This results in true
multithreading across multiple cores, which can result in a huge performance
gain.

When on the GPU, it will rely on NVidia's `CUDA` acceleration provided by the
`sigpy` and `cupy` packages. This can be a hassle to set up correctly, but can
really pay off big time when large amounts of data are involved. For small
amounts of data (typically, ring buffers smaller than 500.000 samples), the
overhead of having to transfer data from system memory to GPU memory and back is
negating any performance gain and simply not worth it.

The FIR filter output is (by mathematical definition) delayed with respect to
the incoming time series data, namely by `T_settle_filter / 2` seconds.
Attribute `valid_slice` will contain the slice to be taken from the incoming
ring buffer corresponding to the matching time stamps of the filter output.

The FIR filter is programmed to be a zero-phase distortion filter, also known
as a linear filter.
"""
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/python-dvg-signal-processing"
__date__ = "28-05-2021"
__version__ = "1.0.0"
# pylint: disable=invalid-name, too-many-instance-attributes, too-few-public-methods, too-many-arguments

from typing import Optional, Union, List, Tuple
from collections import deque  # Use of `deque` is supported but not recommended
from importlib import import_module

import numpy as np
from scipy.signal import firwin, freqz

from dvg_ringbuffer import RingBuffer
from dvg_fftw_convolver import FFTW_Convolver_Valid1D

# ------------------------------------------------------------------------------
#   FreqResponse
# ------------------------------------------------------------------------------


class FreqResponse:
    """Container for the computed theoretical frequency response of the filter
    based on the output of :meth:`scipy.signal.freqz`.

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


# ------------------------------------------------------------------------------
#   RingBuffer_FIR_Filter_Config
# ------------------------------------------------------------------------------


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

        freqz_worN: int
            See :meth:`scipy.signal.freqz` for more details.

        freqz_dB_floor: float

        use_CUDA: bool, optional
            Use NVidia's CUDA acceleration for the 1D FFT convolution? You'll
            need `cupy` and `sigpy` properly installed in your system for CUDA
            to work. Only beneficial when processing large amounts of data
            (approx. > 1e5 samples) as the overhead of copying memory from CPU
            to GPU is substantial. Not really worth it when `block_size <= 500`
            and `N_blocks <= 41`.

            Default: False
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
        fftw_threads: int = 5,
        use_CUDA: bool = False,
    ):
        self.Fs = Fs
        self.block_size = block_size
        self.N_blocks = N_blocks
        self.firwin_cutoff = np.atleast_1d(firwin_cutoff)
        self.firwin_window = firwin_window
        self.firwin_pass_zero = firwin_pass_zero
        self.freqz_worN = freqz_worN
        self.freqz_dB_floor = freqz_dB_floor
        self.fftw_threads = fftw_threads
        self.use_CUDA = use_CUDA

        #  Derived config parameters
        # ---------------------------

        # Full capacity of the ring buffer
        self.rb_capacity = block_size * N_blocks

        # Neat string description of window settings
        self.firwin_window_descr = (
            "%s" % firwin_window
            if isinstance(firwin_window, str)
            else "%s" % [x for x in firwin_window]
        )

        # Calculate max number of possible taps that will fit inside the ring
        # buffer given `block_size` and `N_blocks`. We'll force an odd number in
        # order to create a zero-phase distortion filter, aka linear filter.
        self.firwin_numtaps = block_size * (N_blocks - 1) + 1


# ------------------------------------------------------------------------------
#   RingBuffer_FIR_Filter
# ------------------------------------------------------------------------------


class RingBuffer_FIR_Filter:
    """In progress...

    Args:
        config (RingBuffer_FIR_Filter_Config())
            See :class:`RingBuffer_FIR_Filter_Config`

        name (str, optional):

            Default: ""

    Attributes:
        config (RingBuffer_FIR_Filter_Config())
        name (str)
        freqz (FreqResponse())

        T_settle_filter (float)
            Time period in seconds for the filter to start outputting valid
            data, not to be confused with the filter theoretical response time
            to an impulse. Note that the output of the filter lies in the past
            by `T_settle_filter/2` seconds with respect to the input signal,
            see `rb_valid_slice`.

        filter_has_settled (bool)
            True when the filter starts outputting valid data, not to be
            confused with the filter theoretical response time to an impulse.

        rb_valid_slice (slice)
            Indices within the input-signal ring buffer aligning to the time
            stamps of the computed valid filter output.
    """

    def __init__(
        self,
        config: RingBuffer_FIR_Filter_Config,
        name: str = "",
    ):
        self.config = config
        self.name = name

        # Filter settling
        self.T_settle_filter = (
            (config.N_blocks - 1) * config.block_size / config.Fs
        )  # [s]
        self.filter_has_settled = False
        self._filter_was_settled = False

        # Indices within the input-signal ring buffer aligning to the time
        # stamps of the computed valid filter output
        idx = int((config.firwin_numtaps - 1) / 2)
        self.rb_valid_slice = slice(idx, config.rb_capacity - idx)

        # Container for the computed FIR filter tap array
        self._taps = None
        self._taps_cupy = None  # Only used when `config.use_CUDA = True`

        # Container for the computed frequency response of the filter based
        # on the output of :meth:`scipy.signal.freqz`
        self.freqz = FreqResponse()

        if not config.use_CUDA:
            # Create FFTW plan for FFT convolution
            self._fftw_convolver = FFTW_Convolver_Valid1D(
                config.rb_capacity,
                config.firwin_numtaps,
                fftw_threads=config.fftw_threads,
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

        if c.use_CUDA:
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
            if not c.use_CUDA:
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

        def fancy(name, value, value_format, unit=""):
            format_str = "{:>19s}  %s  {:<s}" % value_format
            print(format_str.format(name, value, unit))

        print("\nRingbuffer_FIR_Filter `%s`" % self.name)
        print("═" * 50)
        fancy("Fs", c.Fs, "{:>9,.2f}", "Hz")
        fancy("block_size", c.block_size, "{:>9d}", "samples")
        fancy("N_blocks", c.N_blocks, "{:>9d}")
        print("─" * 50)
        fancy("rb_capacity", c.rb_capacity, "{:>9d}", "samples")
        fancy("firwin_numtaps", c.firwin_numtaps, "{:>9d}", "samples")
        fancy("rb_valid_slice", str(self.rb_valid_slice), "{:<s}")
        fancy("T_settle_filter", self.T_settle_filter, "{:>9.3f}", "s")
        print("─" * 50)
        fancy("firwin_window", c.firwin_window_descr, "{:<s}")
        fancy(
            "firwin_cutoff",
            "%s" % [round(x, 1) for x in c.firwin_cutoff],
            "{:<s}",
            "Hz",
        )
        fancy("firwin_pass_zero", str(c.firwin_pass_zero), "{:<s}")
        print("═" * 50)
