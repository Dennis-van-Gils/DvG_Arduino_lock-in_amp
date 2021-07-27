#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module to communicate with an Arduino lock-in amplifier device over a serial
connection.
"""
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/DvG_Arduino_lock-in_amp"
__date__ = "27-07-2021"
__version__ = "2.0.0"
# pylint: disable=bare-except, broad-except, pointless-string-statement, invalid-name

import sys
import struct
from enum import Enum
from typing import AnyStr, Optional, Tuple, Union
import time as Time

import serial
import numpy as np
from numba import njit

from dvg_devices import Arduino_protocol_serial
from dvg_debug_functions import dprint, print_fancy_traceback as pft


@njit("float64[:](float64[:])", nogil=True, cache=False)
def round_C_style(array_in: np.ndarray) -> np.ndarray:
    """
    round_C_style([0.1 , 1.45, 1.50, 1.55, -0.1 , -1.45, -1.55])
    Out[]:  array([0.  , 1.  , 1.  , 2.  , -0.  , -1.  , -2.  ])
    """
    _abs = np.abs(array_in)
    _trunc = np.trunc(_abs)
    _frac_rounded = np.zeros_like(_abs)
    _frac_rounded[(_abs % 1) >= 0.5] = 1

    return np.sign(array_in) * (_trunc + _frac_rounded)


class Waveform(Enum):
    # fmt: off
    Unknown  = -1
    Cosine   = 0
    Square   = 1
    Triangle = 2
    # fmt: on


class Alia(Arduino_protocol_serial.Arduino):
    """This class manages the serial protocol for an Arduino lock-in amplifier,
    aka `Alia`.
    """

    class Config:
        """Container for the hardware Arduino lock-in amplifier settings"""

        # fmt: off
        # Microcontroller unit (mcu) info
        mcu_firmware = None  # Firmware version
        mcu_model    = None  # Chipset model
        mcu_fcpu     = None  # Clock frequency
        mcu_uid      = None  # Unique identifier of the chip (serial number)

        # Lock-in amplifier CONSTANTS
        SAMPLING_PERIOD   = 0  # [s]
        BLOCK_SIZE        = 0  # Number of samples send per TX_buffer
        N_BYTES_TX_BUFFER = 0  # [data bytes] Expected number of bytes for each
                               # correctly received TX_buffer from the Arduino
        DAC_OUTPUT_BITS   = 0  # [bits]
        ADC_INPUT_BITS    = 0  # [bits]
        ADC_DIFFERENTIAL  = 0  # [bool]
        ADC_BITS_TO_V     = 0  # Multiplication factor
        A_REF             = 0  # [V] Analog voltage reference of the Arduino
        MIN_N_LUT         = 0  # Minimum allowed number of LUT samples
        MAX_N_LUT         = 0  # Maximum allowed number of LUT samples

        # Derived settings
        Fs                = 0  # [Hz] Sampling rate
        F_Nyquist         = 0  # [Hz] Nyquist frequency
        T_SPAN_TX_BUFFER  = 0  # [s]  Time interval spanned by a single TX_buffer

        # Waveform look-up table (LUT) settings
        N_LUT             = 0  # Number of samples covering a full period

        """ OBSOLETE, kept as reference
        # `LUT_mcu` will contain a copy of the LUT array as used on the
        # microcontroller unit side in units of bit-values as sent out over its
        # DAC. Multiply by `A_REF/(2**ADC_INPUT_BITS - 1)` to get units of [V].
        # `LUT_mcu` is not used in this Python code to reconstruct the `ref_X`
        # and `ref_Y` timeseries, but is kept as a reference for
        # troubleshooting.
        LUT_mcu = np.array([], dtype=np.uint16, order="C")
        """

        # `LUT_X` and `LUT_Y` will contain a single period each of the
        # reference signals, where `Y` is phase-shifted by 90 degrees, i.e. the
        # quadrant. Both will get (re)computed based on the current reference
        # signal parameters.
        LUT_X = np.array([], dtype=float, order="C")  # [V]
        LUT_Y = np.array([], dtype=float, order="C")  # [V]

        # Reference signal parameters
        ref_waveform       = Waveform.Unknown  # Waveform enum
        ref_freq           = 0       # [Hz]
        ref_V_offset       = 0       # [V]
        ref_V_ampl         = 0       # [V]
        ref_V_ampl_RMS     = 0       # [V_RMS]
        ref_RMS_factor     = np.nan  # RMS factor belonging to chosen waveform
        ref_is_clipping_HI = False   # Output is set too high?
        ref_is_clipping_LO = False   # Output is set too low?

        # Serial communication sentinels: Start and end of message
        SOM = b"\x00\x80\x00\x80\x00\x80\x00\x80\x00\x80"
        EOM = b"\xff\x7f\x00\x00\xff\x7f\x00\x00\xff\x7f"
        N_BYTES_SOM = len(SOM)
        N_BYTES_EOM = len(EOM)

        # Binary formats to decode from binary streams
        binfrmt_counter      = ""
        binfrmt_millis       = ""
        binfrmt_micros       = ""
        binfrmt_idx_phase    = ""
        binfrmt_sig_I        = ""
        byte_slice_counter   = slice(0)
        byte_slice_millis    = slice(0)
        byte_slice_micros    = slice(0)
        byte_slice_idx_phase = slice(0)
        byte_slice_sig_I     = slice(0)
        # fmt: on

    def __init__(
        self,
        name="Alia",
        long_name="Arduino lock-in amplifier",
        connect_to_specific_ID="Alia",
        baudrate=1e6,
        read_timeout=1,
        write_timeout=1,
    ):
        super().__init__(
            name=name,
            long_name=long_name,
            connect_to_specific_ID=connect_to_specific_ID,
        )

        self.serial_settings = {
            "baudrate": baudrate,
            "timeout": read_timeout,
            "write_timeout": write_timeout,
        }
        self.read_until_left_over_bytes = bytearray()

        self.config = self.Config()
        self.lockin_paused = True

    # --------------------------------------------------------------------------
    #  begin
    # --------------------------------------------------------------------------

    def begin(
        self,
        waveform: Optional[Waveform] = None,
        freq: Optional[float] = None,
        V_offset: Optional[float] = None,
        V_ampl: Optional[float] = None,
        V_ampl_RMS: Optional[float] = None,
    ) -> bool:
        """Determine the chipset and firmware of the Arduino lock-in amp and
        prepare the lock-in amp for operation. The default startup state is
        off. The optional parameters can be used to set the reference signal and
        when not supplied, the pre-existing values known to the Arduino will be
        used instead, i.e. it will pick up where it left.

        Returns:
            True if successful, False otherwise.
        """
        success, _foo, _bar = self.turn_off()
        if not success:
            return False

        # Shorthand alias
        c = self.config

        print("Microcontroller")
        print("───────────────\n")
        success, ans_str = self.query("mcu?")
        if success:
            try:
                (
                    c.mcu_firmware,
                    c.mcu_model,
                    c.mcu_fcpu,
                    c.mcu_uid,
                ) = ans_str.split("\t")
                c.mcu_fcpu = int(c.mcu_fcpu)
            except Exception as err:
                pft(err)
                return False
        else:
            return False

        print("  firmware  %s" % c.mcu_firmware)
        print("     model  %s" % c.mcu_model)
        print("      fcpu  %.0f MHz" % (c.mcu_fcpu / 1e6))
        print("    serial  %s" % c.mcu_uid)
        print("")

        print("Lock-in constants")
        print("─────────────────\n")
        success, ans_str = self.query("const?")
        if success:
            try:
                # fmt: off
                ans_list = ans_str.split("\t")
                c.SAMPLING_PERIOD   = (round(float(ans_list[0])*1e-6, 9))
                c.BLOCK_SIZE        = int(ans_list[1])
                c.N_BYTES_TX_BUFFER = int(ans_list[2])
                c.DAC_OUTPUT_BITS   = int(ans_list[3])
                c.ADC_INPUT_BITS    = int(ans_list[4])
                c.ADC_DIFFERENTIAL  = bool(int(ans_list[5]))
                c.A_REF             = float(ans_list[6])

                if c.mcu_firmware == "ALIA v0.2.0 VSCODE":
                    # Legacy firmware support
                    pass
                else:
                    c.MIN_N_LUT     = int(ans_list[7])
                    c.MAX_N_LUT     = int(ans_list[8])
                # fmt: on

            except Exception as err:
                pft(err)
                sys.exit(1)
        else:
            return False

        c.Fs = round(1.0 / c.SAMPLING_PERIOD, 6)
        c.F_Nyquist = round(c.Fs / 2, 6)
        c.T_SPAN_TX_BUFFER = c.BLOCK_SIZE * c.SAMPLING_PERIOD
        c.ADC_BITS_TO_V = c.A_REF / ((1 << c.ADC_INPUT_BITS) - 1)
        if c.ADC_DIFFERENTIAL:
            c.ADC_BITS_TO_V *= 2

        def fancy(name, value, value_format, unit=""):
            format_str = "{:>16s}  %s  {:<s}" % value_format
            print(format_str.format(name, value, unit))

        fancy("Fs", c.Fs, "{:>12,.2f}", "Hz")
        fancy("F_Nyquist", c.F_Nyquist, "{:>12,.2f}", "Hz")
        fancy("sampling period", c.SAMPLING_PERIOD * 1e6, "{:>12.3f}", "us")
        fancy("block size", c.BLOCK_SIZE, "{:>12d}", "samples")
        fancy("TX buffer", c.N_BYTES_TX_BUFFER, "{:>12d}", "bytes")
        fancy("TX buffer", c.T_SPAN_TX_BUFFER, "{:>12.3f}", "s")
        fancy(
            "TX baud rate",
            c.N_BYTES_TX_BUFFER * c.Fs / c.BLOCK_SIZE * 10,
            "{:>12,.0f}",
            "Bd",
        )
        fancy("DAC output", c.DAC_OUTPUT_BITS, "{:>12d}", "bit")
        fancy("ADC input", c.ADC_INPUT_BITS, "{:>12d}", "bit")
        fancy(
            "ADC input",
            "differential" if c.ADC_DIFFERENTIAL else "single-ended",
            "{:s}",
        )
        fancy("A_ref", c.A_REF, "{:>12.3f}", "V")
        if c.mcu_firmware == "ALIA v0.2.0 VSCODE":
            # Legacy firmware support
            pass
        else:
            fancy("min N_LUT", c.MIN_N_LUT, "{:>12d}", "samples")
            fancy("max N_LUT", c.MAX_N_LUT, "{:>12d}", "samples")

        self.set_ref(waveform, freq, V_offset, V_ampl, V_ampl_RMS)

        print("┌─────────────────────────┐")
        print("│     All systems GO!     │")
        print("└─────────────────────────┘\n")

        # fmt: off
        # Binary formats to decode from binary streams
        if c.mcu_firmware == "ALIA v0.2.0 VSCODE":
            # Legacy firmware support
            c.binfrmt_counter   = "<I"      # [uint32_t] TX_buffer header
            c.binfrmt_millis    = "<I"      # [uint32_t] TX_buffer header
            c.binfrmt_micros    = "<H"      # [uint16_t] TX_buffer header
            c.binfrmt_idx_phase = "<{:d}H"  # [uint16_t] TX_buffer body
            c.binfrmt_sig_I     = "<{:d}h"  # [int16_t]  TX_buffer body

            c.byte_slice_counter = slice(
                c.N_BYTES_SOM,
                c.N_BYTES_SOM
                + struct.calcsize(c.binfrmt_counter[-1]),
            )
            c.byte_slice_millis = slice(
                c.byte_slice_counter.stop,
                c.byte_slice_counter.stop
                + struct.calcsize(c.binfrmt_millis[-1]),
            )
            c.byte_slice_micros = slice(
                c.byte_slice_millis.stop,
                c.byte_slice_millis.stop
                + struct.calcsize(c.binfrmt_micros[-1]),
            )
            c.byte_slice_idx_phase = slice(
                c.byte_slice_micros.stop,
                c.byte_slice_micros.stop
                + c.BLOCK_SIZE * struct.calcsize(c.binfrmt_idx_phase[-1]),
            )
            c.byte_slice_sig_I = slice(
                c.byte_slice_idx_phase.stop,
                c.byte_slice_idx_phase.stop
                + c.BLOCK_SIZE * struct.calcsize(c.binfrmt_sig_I[-1]),
            )
        else:
            # "ALIA v1.0.0" and above
            c.binfrmt_counter   = "<I"      # [uint32_t] TX_buffer header
            c.binfrmt_millis    = "<I"      # [uint32_t] TX_buffer header
            c.binfrmt_micros    = "<H"      # [uint16_t] TX_buffer header
            c.binfrmt_idx_phase = "<H"      # [uint16_t] TX_buffer header
            c.binfrmt_sig_I     = "<{:d}h"  # [int16_t]  TX_buffer body

            c.byte_slice_counter = slice(
                c.N_BYTES_SOM,
                c.N_BYTES_SOM
                + struct.calcsize(c.binfrmt_counter[-1]),
            )
            c.byte_slice_millis = slice(
                c.byte_slice_counter.stop,
                c.byte_slice_counter.stop
                + struct.calcsize(c.binfrmt_millis[-1]),
            )
            c.byte_slice_micros = slice(
                c.byte_slice_millis.stop,
                c.byte_slice_millis.stop
                + struct.calcsize(c.binfrmt_micros[-1]),
            )
            c.byte_slice_idx_phase = slice(
                c.byte_slice_micros.stop,
                c.byte_slice_micros.stop
                + struct.calcsize(c.binfrmt_idx_phase[-1]),
            )
            c.byte_slice_sig_I = slice(
                c.byte_slice_idx_phase.stop,
                c.byte_slice_idx_phase.stop
                + c.BLOCK_SIZE * struct.calcsize(c.binfrmt_sig_I[-1]),
            )
        # fmt: on

        return True

    # --------------------------------------------------------------------------
    #   safe_query
    # --------------------------------------------------------------------------

    def safe_query(
        self, msg_str: AnyStr, raises_on_timeout: bool = True
    ) -> Tuple[bool, AnyStr]:
        """Wraps `query()` with a check on the running state of the lock-in amp.
        When running it will stop running, perform the query and resume running.

        Returns:
            Tuple(
                success: bool
                ans_str: str | bytes | None
            )
        """
        was_paused = self.lockin_paused

        if not was_paused:
            self.turn_off()

        success, ans_str = self.query(msg_str, raises_on_timeout)

        if success and not was_paused:
            self.turn_on()

        return success, ans_str

    # --------------------------------------------------------------------------
    #   turn_on/off
    # --------------------------------------------------------------------------

    def turn_on(self, reset_timer: bool = False) -> bool:
        """
        Returns:
            True if successful, False otherwise.
        """
        success = self.write("_on" if reset_timer else "on")
        if success:
            self.lockin_paused = False
            self.read_until_left_over_bytes = bytearray()

        return success

    def turn_off(
        self, raises_on_timeout: bool = False
    ) -> Tuple[bool, bool, bytes]:
        """
        Returns:
            Tuple(
                success  : bool,
                was_off  : bool,
                ans_bytes: bytes  # For debugging purposes
            )
        """
        success = False
        was_off = True
        ans_bytes = b""

        # Clear potentially large amount of binary data waiting in the buffer to
        # be read. Essential.
        self.ser.flushInput()

        if self.write("off", raises_on_timeout):
            self.ser.flushOutput()  # Send out 'off' as fast as possible

            # Check for acknowledgement reply
            try:
                ans_bytes = self.ser.read_until("off\n".encode())
                # print(len(ans_bytes))
                # print("found off: ", end ='')
                # print(ans_bytes[-4:])
            except (
                serial.SerialTimeoutException,
                serial.SerialException,
            ) as err:
                # NOTE: The Serial library does not throw an exception when it
                # actually times out! We will check for zero received bytes as
                # indication for timeout, later.
                pft(err, 3)
            except Exception as err:
                pft(err, 3)
                sys.exit(1)
            else:
                if len(ans_bytes) == 0:
                    # Received 0 bytes, probably due to a timeout.
                    pft("Received 0 bytes. Read probably timed out.", 3)
                else:
                    try:
                        was_off = ans_bytes[-12:] == b"already_off\n"
                    except:
                        pass
                    success = True
                    self.lockin_paused = True

        return success, was_off, ans_bytes

    # --------------------------------------------------------------------------
    #   LUT
    # --------------------------------------------------------------------------

    ''' OBSOLETE, kept as reference
    def query_LUT(self) -> bool:
        """Send command "lut?" to the Arduino lock-in amp to retrieve the look-
        up table (LUT) that is used for the output reference signal `ref_X`.

        This method will update members:
            `config.N_LUT`
            `config.is_LUT_dirty`
            `config.LUT_mcu`

        Returns:
            True if successful, False otherwise.
        """
        c = self.config  # Short-hand

        was_paused = self.lockin_paused
        if not was_paused:
            self.turn_off()

        if not self.write("l?"):
            return False

        # First read `N_LUT` and `is_LUT_dirty` from the binary stream
        try:
            ans_bytes = self.ser.read(size=3)
        except:
            pft("'%s' I/O ERROR: Can't read bytes LUT" % self.name)
            self.ser.flushInput()
            return False

        if len(ans_bytes) == 0:
            # Received 0 bytes, probably due to a timeout.
            pft("'%s' I/O ERROR: Timed out reading LUT" % self.name)
            self.ser.flushInput()
            return False

        try:
            N_LUT = struct.unpack("<H", ans_bytes[0:2])
            is_LUT_dirty = struct.unpack("<?", ans_bytes[2:])
        except:
            pft("'%s' I/O ERROR: Can't unpack bytes LUT" % self.name)
            self.ser.flushInput()
            return False

        c.N_LUT = int(N_LUT[0])
        #c.is_LUT_dirty = bool(is_LUT_dirty[0])

        # Now read the remaining LUT array from the binary stream still left in
        # the serial buffer
        try:
            ans_bytes = self.ser.read(size=c.N_LUT * 2)
        except:
            pft("'%s' I/O ERROR: Can't read bytes LUT" % self.name)
            self.ser.flushInput()
            return False

        if len(ans_bytes) == 0:
            # Received 0 bytes, probably due to a timeout.
            pft("'%s' I/O ERROR: Timed out reading LUT" % self.name)
            self.ser.flushInput()
            return False

        try:
            LUT_mcu = np.array(
                struct.unpack("<{:d}H".format(c.N_LUT), ans_bytes),
                dtype=np.uint16,
                order="C",
            )
        except:
            pft("'%s' I/O ERROR: Can't unpack bytes LUT" % self.name)
            self.ser.flushInput()
            return False

        c.LUT_mcu = LUT_mcu

        if not was_paused:
            self.turn_on()

        return True
    '''

    # --------------------------------------------------------------------------
    #   query_ref
    # --------------------------------------------------------------------------

    def query_ref(self) -> bool:
        """Send command "ref?" to the Arduino lock-in amp to retrieve the
        reference signal `ref_X` parameters from it, and to compute the
        LUT waveform internal to the Arduino. Subsequently, `LUT_X` and `LUT_Y`
        will get recomputed on the Python side.

        This method will update members:
            `config.ref_waveform`
            `config.ref_freq`
            `config.ref_V_offset`
            `config.ref_V_ampl`
            `config.ref_V_ampl_RMS`
            `config.ref_RMS_factor`
            `config.ref_is_clipping_HI`
            `config.ref_is_clipping_LO`
            `config.N_LUT`
            `config.LUT_X`
            `config.LUT_Y`

        Returns:
            True if successful, False otherwise.
        """
        c = self.config  # Short-hand

        success, ans_str = self.safe_query("?")
        if success:
            try:
                ans_list = ans_str.split("\t")

                # fmt: off
                c.ref_waveform       = Waveform[ans_list[0]]
                c.ref_freq           = float(ans_list[1])
                c.ref_V_offset       = float(ans_list[2])
                c.ref_V_ampl         = float(ans_list[3])
                c.ref_V_ampl_RMS     = float(ans_list[4])
                c.ref_is_clipping_HI = bool(int(ans_list[5]))
                c.ref_is_clipping_LO = bool(int(ans_list[6]))
                c.N_LUT              = int(ans_list[7])
                # fmt: on
            except Exception as err:
                pft(err)
                return False
        else:
            return False

        if c.mcu_firmware == "ALIA v0.2.0 VSCODE":

            # ---------------------------
            #       Legacy firmware
            # ---------------------------

            c.ref_RMS_factor = np.sqrt(2)  # Fixed to `Cosine`

        else:

            # ---------------------------
            #       Modern firmware
            # ---------------------------

            # Reconstruct `LUT_X` and `LUT_Y` in advance
            if c.ref_waveform == Waveform.Cosine:
                c.ref_RMS_factor = np.sqrt(2)

                phis = np.linspace(0, 2 * np.pi, num=c.N_LUT, endpoint=False)
                LUT_X = 0.5 * (1 + np.cos(phis))
                LUT_Y = 0.5 * (1 + np.sin(phis))

            elif c.ref_waveform == Waveform.Square:
                c.ref_RMS_factor = 1

                idxs = np.arange(0, c.N_LUT)
                LUT_X = np.sign(np.cos(2 * np.pi * idxs / c.N_LUT))
                LUT_X[LUT_X < 0] = 0

                if c.N_LUT % 4 > 0:
                    # Not an integer multiple of 4
                    # Quadrant does not neatly exist, hence interpolate
                    LUT_Y = np.interp(
                        np.arange(c.N_LUT) + c.N_LUT * 3.0 / 4.0,
                        np.arange(c.N_LUT * 2),
                        np.tile(LUT_X, 2),
                    )
                else:
                    # Perfect quadrant exists
                    LUT_Y = np.sign(np.sin(2 * np.pi * idxs / c.N_LUT))
                    LUT_Y[LUT_Y < 0] = 0

            elif c.ref_waveform == Waveform.Triangle:
                c.ref_RMS_factor = np.sqrt(3)

                idxs = np.arange(0, c.N_LUT)
                LUT_X = 2 * np.abs(idxs / c.N_LUT - 0.5)
                LUT_Y = 2 * np.abs(
                    ((idxs - c.N_LUT / 4) % c.N_LUT) / c.N_LUT - 0.5
                )

            elif c.ref_waveform == Waveform.Unknown:
                c.ref_RMS_factor = np.nan

                LUT_X = np.full(np.nan, c.N_LUT)
                LUT_Y = np.full(np.nan, c.N_LUT)

            # Transform [0, 1] to [V]
            LUT_X = (c.ref_V_offset - c.ref_V_ampl) + 2 * c.ref_V_ampl * LUT_X
            LUT_Y = (c.ref_V_offset - c.ref_V_ampl) + 2 * c.ref_V_ampl * LUT_Y
            LUT_X.clip(0, c.A_REF, out=LUT_X)
            LUT_Y.clip(0, c.A_REF, out=LUT_Y)

            c.LUT_X = np.asarray(LUT_X, dtype=float, order="C")
            c.LUT_Y = np.asarray(LUT_Y, dtype=float, order="C")

        return True

    # --------------------------------------------------------------------------
    #   set_ref
    # --------------------------------------------------------------------------

    def set_ref(
        self,
        waveform: Optional[Waveform] = None,
        freq: Optional[float] = None,
        V_offset: Optional[float] = None,
        V_ampl: Optional[float] = None,
        V_ampl_RMS: Optional[float] = None,
    ) -> bool:
        """Send new reference signal `ref_X` parameters to the Arduino and
        retrieve the obtained parameters. The Arduino will compute the new LUT,
        based on the obtained parameters. The actually obtained parameters might
        differ from the requested ones, noticably the frequency. Subsequently,
        `LUT_X` and `LUT_Y` will get recomputed on the Python side.

        This method will update members:
            `config.ref_waveform`
            `config.ref_freq`
            `config.ref_V_offset`
            `config.ref_V_ampl`
            `config.ref_V_ampl_RMS`
            `config.ref_RMS_factor`
            `config.ref_is_clipping_HI`
            `config.ref_is_clipping_LO`
            `config.N_LUT`
            `config.LUT_X`
            `config.LUT_Y`

        Args:
            waveform (Waveform):
                Enumeration decoding a waveform type, like cosine, square or
                triangle wave.

            freq (float):
                The requested frequency in Hz.

            V_offset (float):
                The requested voltage offset in V.

            V_ampl (float):
                The requested voltage amplitude in V.

            V_ampl_RMS (float):
                The requested voltage amplitude in V_RMS.

        Returns:
            True if successful, False otherwise.
        """
        was_paused = self.lockin_paused
        if not was_paused:
            self.turn_off()

        if waveform is not None:
            success, _ans_str = self.query("_wave %i" % waveform.value)
            if not success:
                return False

        if freq is not None:
            success, _ans_str = self.query("_freq %f" % freq)
            if not success:
                return False

        if V_offset is not None:
            success, _ans_str = self.query("_offs %f" % V_offset)
            if not success:
                return False

        if V_ampl is not None:
            success, _ans_str = self.query("_ampl %f" % V_ampl)
            if not success:
                return False

        if V_ampl_RMS is not None:
            success, _ans_str = self.query("_vrms %f" % V_ampl_RMS)
            if not success:
                return False

        if not self.query_ref():
            return False

        if not was_paused:
            self.turn_on()

        def pprint(str_name, val_req, val_obt, str_unit="", str_format="s"):
            line = "  {:>8s}".format(str_name)
            line += (
                "  {:>9s}".format("-")
                if val_req is None
                else "  {:>9{p}}".format(val_req, p=str_format)
            )
            line += "  {:>9{p}}".format(val_obt, p=str_format)
            line += "  " + str_unit
            print(line)

        c = self.config  # Short-hand
        print("\nReference signal `ref_X*`")
        print("─────────────────────────\n")
        print("            REQUESTED   OBTAINED")
        pprint(
            "waveform",
            None if waveform is None else waveform.name,
            c.ref_waveform.name,
        )
        pprint("freq", freq, c.ref_freq, "Hz", ",.3f")
        pprint("offset", V_offset, c.ref_V_offset, "V", ".3f")
        pprint("ampl", V_ampl_RMS, c.ref_V_ampl_RMS, "V_RMS", ".3f")
        pprint("", V_ampl, c.ref_V_ampl, "V", ".3f")
        pprint("N_LUT", None, c.N_LUT, "", "d")
        print()

        if c.ref_is_clipping_HI:
            print("             !! Clipping HI !!")
        if c.ref_is_clipping_LO:
            print("             !! Clipping LO !!")
        if c.ref_is_clipping_HI or c.ref_is_clipping_LO:
            print()

        return True

    # --------------------------------------------------------------------------
    #   read_until_EOM
    # --------------------------------------------------------------------------

    def read_until_EOM(self) -> bytes:
        """Reads from the serial port until the EOM sentinel is found or until
        a timeout occurs. Any left-over bytes after the EOM will be remembered
        and prefixed to the next `read_until_EOM()` call. This method is
        blocking. Read `Behind the scenes` for more information on the use
        of this method in multithreaded scenarios.

        Returns:
            The read contents as type `bytes`.

        Behind the scenes:
            Reading happens in bursts whenever any new bytes are waiting in the
            serial-in buffer of the OS. When no bytes are waiting, this method
            `read_until_EOM()` will sleep 0.01 s, before trying again. All read
            bytes will be collected in a single bytearray and tested for the EOM
            sentinel.

            Even though this method itself is blocking (in its caller thread),
            other threads will be able to get processed by the Python
            Interpreter because of the small sleep period. The sleep period will
            free up the caller thread from the Python GIL.

            See comment by Gabriel Staples
            https://stackoverflow.com/questions/17553543/pyserial-non-blocking-read-loop/38758773
        """

        # pylint: disable=protected-access
        timeout = serial.Timeout(self.ser._timeout)

        c = bytearray(self.read_until_left_over_bytes)
        idx_EOM = -1
        while True:
            try:
                if self.ser.in_waiting > 0:
                    new_bytes = self.ser.read(self.ser.in_waiting)

                    if new_bytes:
                        # print(len(new_bytes))
                        c.extend(new_bytes)
                        idx_EOM = c.find(self.config.EOM)

                    if idx_EOM > -1:
                        # print("_____EOM")
                        N_left_over_bytes_after_EOM = (
                            len(c) - idx_EOM - self.config.N_BYTES_EOM
                        )

                        if N_left_over_bytes_after_EOM:
                            left_over_bytes = c[-N_left_over_bytes_after_EOM:]
                            c = c[:-N_left_over_bytes_after_EOM]
                            # print(
                            #    "LEFT OVER BYTES: %d"
                            #    % N_left_over_bytes_after_EOM
                            # )
                        else:
                            left_over_bytes = bytearray()

                        self.read_until_left_over_bytes = left_over_bytes
                        break

                # Do not hog the CPU
                Time.sleep(0.01)

            except Exception as err:
                pft(err)

            if timeout.expired():
                break

        return bytes(c)

    # --------------------------------------------------------------------------
    #   listen_to_lockin_amp
    # --------------------------------------------------------------------------

    def listen_to_lockin_amp(
        self,
    ) -> Tuple[
        bool, Union[int, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """Reads incoming data packets coming from the lock-in amp. This method
        is blocking until it receives an EOM (end-of-message) sentinel or until
        it times out.

        Returns:
            Tuple (
                success: bool
                counter: int | numpy.nan
                time   : numpy.ndarray, units [us]
                ref_X  : numpy.ndarray, units [V]
                ref_Y  : numpy.ndarray, units [V]
                sig_I  : numpy.ndarray, units [V]
            )
        """
        failed = False, None, [np.nan], [np.nan], [np.nan], [np.nan]
        c = self.config  # Shorthand alias

        ans_bytes = self.read_until_EOM()
        # dprint("EOM found with %i bytes and..." % len(ans_bytes))
        if not ans_bytes[: c.N_BYTES_SOM] == c.SOM:
            dprint("'%s' I/O ERROR: No SOM found" % self.name)
            return failed

        # dprint("SOM okay")
        if not len(ans_bytes) == c.N_BYTES_TX_BUFFER:
            dprint(
                "'%s' I/O ERROR: Expected %i bytes but received %i"
                % (self.name, c.N_BYTES_TX_BUFFER, len(ans_bytes))
            )
            return failed

        if c.mcu_firmware == "ALIA v0.2.0 VSCODE":

            # ---------------------------
            #       Legacy firmware
            # ---------------------------

            try:
                counter = struct.unpack(
                    c.binfrmt_counter, ans_bytes[c.byte_slice_counter]
                )
                millis = struct.unpack(
                    c.binfrmt_millis, ans_bytes[c.byte_slice_millis]
                )
                micros = struct.unpack(
                    c.binfrmt_micros, ans_bytes[c.byte_slice_micros]
                )
                idx_phase = struct.unpack(
                    c.binfrmt_idx_phase.format(c.BLOCK_SIZE),
                    ans_bytes[c.byte_slice_idx_phase],
                )
                sig_I = struct.unpack(
                    c.binfrmt_sig_I.format(c.BLOCK_SIZE),
                    ans_bytes[c.byte_slice_sig_I],
                )
            except:
                dprint("'%s' I/O ERROR: Can't unpack bytes" % self.name)
                return failed

            # fmt: off
            counter   = counter[0]
            millis    = millis[0]
            micros    = micros[0]
            idx_phase = np.array(idx_phase, dtype=int, order="C")
            sig_I     = np.array(sig_I, dtype=float, order="C")
            # fmt: on

            # dprint("%i %i" % (millis, micros))
            t0 = millis * 1000 + micros
            time = t0 + np.arange(0, c.BLOCK_SIZE) * c.SAMPLING_PERIOD * 1e6
            time = np.asarray(time, dtype=float, order="C")

            phi = 2 * np.pi * idx_phase / c.N_LUT

            # DEBUG test: Add artificial phase delay between ref_X/Y and sig_I
            if 0:  # pylint: disable=using-constant-test
                phase_offset_deg = 50
                phi = np.unwrap(phi + phase_offset_deg / 180 * np.pi)

            # Construct `ref_X` and `ref_Y`
            ref_X = np.cos(phi)
            ref_Y = np.sin(phi)

            # OVERRIDE: Only `Cosine` allowed, because `Square` and `Triangle`
            # can not be garantueed deterministic on both Arduino and Python
            # side due to rounding differences and the problem of computing the
            # correct 90 degrees quadrant `ref_Y`.
            """
            if c.ref_waveform == Waveform.Square:
                ref_X.fill(-1)
                ref_X[ref_X_phase < (c.N_LUT / 4.0)] = 1.0
                ref_X[ref_X_phase >= (c.N_LUT / 4.0 * 3.0)] = 1.0
                ref_Y.fill(-1)
                ref_Y[ref_X_phase < (c.N_LUT / 2.0)] = 1.0
                ref_Y[ref_X_phase == (c.N_LUT / 2.0)] = 0.0

            if c.ref_waveform == Waveform.Triangle:
                ref_X = np.arcsin(ref_X) / np.pi * 2
                ref_Y = np.arcsin(ref_Y) / np.pi * 2
            """

            np.multiply(ref_X, c.ref_V_ampl, out=ref_X)
            np.multiply(ref_Y, c.ref_V_ampl, out=ref_Y)

            ref_X = (c.ref_V_offset + ref_X).clip(0, c.A_REF)
            ref_Y = (c.ref_V_offset + ref_Y).clip(0, c.A_REF)

            # Transform `sig_I` from [bits] to [V]
            sig_I = np.multiply(sig_I, c.ADC_BITS_TO_V, out=sig_I)

        else:

            # ---------------------------
            #       Modern firmware
            # ---------------------------

            try:
                counter = struct.unpack(
                    c.binfrmt_counter, ans_bytes[c.byte_slice_counter]
                )
                millis = struct.unpack(
                    c.binfrmt_millis, ans_bytes[c.byte_slice_millis]
                )
                micros = struct.unpack(
                    c.binfrmt_micros, ans_bytes[c.byte_slice_micros]
                )
                idx_phase = struct.unpack(
                    c.binfrmt_idx_phase, ans_bytes[c.byte_slice_idx_phase]
                )
                sig_I = struct.unpack(
                    c.binfrmt_sig_I.format(c.BLOCK_SIZE),
                    ans_bytes[c.byte_slice_sig_I],
                )
            except:
                dprint("'%s' I/O ERROR: Can't unpack bytes" % self.name)
                return failed

            # fmt: off
            counter   = counter[0]
            millis    = millis[0]
            micros    = micros[0]
            idx_phase = idx_phase[0]
            sig_I     = np.array(sig_I, dtype=float, order="C")
            # fmt: on

            # dprint("%i %i" % (millis, micros))
            t0 = millis * 1000 + micros
            time = t0 + np.arange(0, c.BLOCK_SIZE) * c.SAMPLING_PERIOD * 1e6
            time = np.asarray(time, dtype=float, order="C")

            # DEBUG test: Add artificial phase delay between ref_X/Y and sig_I
            if 0:  # pylint: disable=using-constant-test
                phase_offset_deg = 120
                idx_phase += int(np.round(phase_offset_deg / 360 * c.N_LUT))

            # Construct `ref_X` and `ref_Y`
            LUT_X = np.roll(c.LUT_X, -idx_phase)
            LUT_Y = np.roll(c.LUT_Y, -idx_phase)

            ref_X_tiled = np.tile(LUT_X, int(np.ceil(c.BLOCK_SIZE / c.N_LUT)))
            ref_Y_tiled = np.tile(LUT_Y, int(np.ceil(c.BLOCK_SIZE / c.N_LUT)))

            ref_X = np.asarray(
                ref_X_tiled[: c.BLOCK_SIZE],
                dtype=float,
                order="C",
            )

            ref_Y = np.asarray(
                ref_Y_tiled[: c.BLOCK_SIZE],
                dtype=float,
                order="C",
            )

            # Transform `sig_I` from [bits] to [V]
            sig_I = np.multiply(sig_I, c.ADC_BITS_TO_V, out=sig_I)

        return True, counter, time, ref_X, ref_Y, sig_I


# ------------------------------------------------------------------------------
#   Main: Will show a demo when run from the terminal
# ------------------------------------------------------------------------------

if __name__ == "__main__":

    def print_reply(reply: str):
        print("")
        print("success  %s" % reply[0])
        print("counter  %s" % reply[1])
        print("   time  %s" % reply[2])
        print("  ref_X  %s" % reply[3])
        print("  ref_Y  %s" % reply[4])
        print("  sig_I  %s\n" % reply[5])

    alia = Alia()
    alia.auto_connect()

    if not alia.is_alive:
        sys.exit(0)

    alia.begin(
        freq=220,
        waveform=Waveform.Cosine,
    )
    # alia.begin()
    alia.turn_on(reset_timer=True)

    print_reply(alia.listen_to_lockin_amp())
    print_reply(alia.listen_to_lockin_amp())
    alia.set_ref(freq=250, V_offset=1.5, V_ampl=0.5, waveform=Waveform.Triangle)
    print_reply(alia.listen_to_lockin_amp())
    print_reply(alia.listen_to_lockin_amp())

    alia.turn_off()
    alia.close()
