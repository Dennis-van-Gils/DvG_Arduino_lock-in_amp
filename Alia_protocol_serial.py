#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module to communicate with an Arduino lock-in amplifier device over a serial
connection.
"""
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/DvG_Arduino_lock-in_amp"
__date__ = "17-05-2021"
__version__ = "2.0.0"
# pylint: disable=bare-except, broad-except, pointless-string-statement, invalid-name

import sys
import struct
from enum import Enum
from typing import AnyStr, Optional, Tuple

import serial
import numpy as np

from dvg_devices import Arduino_protocol_serial
from dvg_debug_functions import dprint, print_fancy_traceback as pft


def round_C_style(array_in: np.ndarray) -> np.ndarray:
    """
    round_C_style([0.1 , 1.45, 1.55, -0.1 , -1.45, -1.55])
    Out[]:  array([0.  , 1.  , 2.  , -0.  , -1.  , -2.  ])
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
        # fmt: off
        # Serial communication sentinels: start and end of message
        SOM = b"\x00\x80\x00\x80\x00\x80\x00\x80\x00\x80"
        EOM = b"\xff\x7f\x00\x00\xff\x7f\x00\x00\xff\x7f"
        N_BYTES_SOM = len(SOM)
        N_BYTES_EOM = len(EOM)

        # Data types to decode from binary streams
        binary_type_counter   = "I"  # [uint32_t] TX_buffer header
        binary_type_millis    = "I"  # [uint32_t] TX_buffer header
        binary_type_micros    = "H"  # [uint16_t] TX_buffer header
        binary_type_idx_phase = "H"  # [uint16_t] TX_buffer header
        binary_type_sig_I     = "H"  # [uint16_t] TX_buffer body

        # Return types
        return_type_time   = np.float64  # Ensure signed to allow for flexible arithmetic
        return_type_ref_XY = np.float64
        return_type_sig_I  = np.float64

        # Microcontroller unit (mcu) info
        mcu_firmware = ""      # Firmware version
        mcu_model    = ""      # Chipset model
        mcu_uid      = ""      # Unique identifier of the chip (serial number)

        # Lock-in amplifier CONSTANTS
        SAMPLING_PERIOD   = 0  # [s]
        BLOCK_SIZE        = 0  # Number of samples send per TX_buffer
        N_BYTES_TX_BUFFER = 0  # [data bytes] Expected number of bytes for each
                               # correctly received TX_buffer from the Arduino
        DAC_OUTPUT_BITS   = 0  # [bits]
        ADC_INPUT_BITS    = 0  # [bits]
        A_REF             = 0  # [V] Analog voltage reference of the Arduino
        MIN_N_LUT         = 0  # Minimum allowed number of LUT samples
        MAX_N_LUT         = 0  # Maximum allowed number of LUT samples

        # Derived settings
        Fs        = 0          # [Hz] Sampling rate
        F_Nyquist = 0          # [Hz] Nyquist frequency
        T_SPAN_TX_BUFFER = 0   # [s]  Time interval spanned by a single TX_buffer

        # Waveform look-up table (LUT) settings
        N_LUT = 0              # Number of samples covering a full period
        is_LUT_dirty = False   # Is there a pending change on the LUT?
        LUT_wave = np.array([], dtype=np.uint16)
        """LUT_wave will contain a copy of the LUT array of the current
        reference signal waveform as used on the Arduino side. This array will
        be used to reconstruct the ref_X and ref_Y signals, based on the phase
        index that is sent in the header of each TX_buffer. The unit of each
        element in the array is the bit-value that is sent out over the DAC of
        the Arduino. Hence, multiply by A_REF/(2**ADC_INPUT_BITS - 1) to get
        units of [V].
        """

        # Reference signal settings
        ref_freq     = 0                 # [Hz] Frequency
        ref_V_offset = 0                 # [V]  Voltage offset
        ref_V_ampl   = 0                 # [V]  Voltage amplitude
        ref_waveform = Waveform.Unknown  # Waveform enum
        # fmt: on

    def __init__(
        self,
        name="Alia",
        long_name="Arduino lock-in amplifier",
        connect_to_specific_ID="Alia",
        baudrate=1.2e6,
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
        ref_freq: Optional[float] = None,
        ref_V_offset: Optional[float] = None,
        ref_V_ampl: Optional[float] = None,
        ref_waveform: Optional[Waveform] = None,
    ) -> bool:
        """Prepare the lock-in amp for operation. The default startup state is
        off. If the optional parameters `ref_freq`, `ref_V_offset`, `ref_V_ampl`
        or `ref_waveform` are not passed, the pre-existing values known to the
        Arduino will be used instead, i.e. it will pick up where it left.

        Returns:
            True if successful, False otherwise.
        """
        success, _foo, _bar = self.turn_off()
        if not success:
            return False

        # Shorthand alias
        c = self.config

        print("Microcontroller\n")
        success, ans_str = self.query("mcu?")
        if success:
            try:
                c.mcu_firmware, c.mcu_model, c.mcu_uid = ans_str.split("\t")
            except Exception as err:
                pft(err)
                return False
        else:
            return False

        print("  Firmware : %s" % c.mcu_firmware)
        print("  Model    : %s" % c.mcu_model)
        print("  UID      : %s" % c.mcu_uid)

        print("\nLock-in constants\n")
        success, ans_str = self.query("const?")
        if success:
            try:
                ans_list = ans_str.split("\t")
                # Round SAMPLING PERIOD to nanosecond resolution to
                # prevent e.g. 1.0 being stored as 0.9999999999
                # fmt: off
                c.SAMPLING_PERIOD   = (round(float(ans_list[0])*1e-6, 9))
                c.BLOCK_SIZE        = int(ans_list[1])
                c.N_BYTES_TX_BUFFER = int(ans_list[2])
                c.DAC_OUTPUT_BITS   = int(ans_list[3])
                c.ADC_INPUT_BITS    = int(ans_list[4] )
                c.A_REF             = float(ans_list[5])
                c.MIN_N_LUT         = int(ans_list[6])
                c.MAX_N_LUT         = int(ans_list[7])
                # fmt: on
            except Exception as err:
                pft(err)
                return False
        else:
            return False
        # fmt: off
        print("  SAMPLING_PERIOD   : %10.3f  us"    % (c.SAMPLING_PERIOD*1e6))
        print("  BLOCK_SIZE        : %10i  samples" % c.BLOCK_SIZE)
        print("  N_BYTES_TX_BUFFER : %10i  bytes"   % c.N_BYTES_TX_BUFFER)
        print("  DAC_OUTPUT_BITS   : %10i  bit"     % c.DAC_OUTPUT_BITS)
        print("  ADC_INPUT_BITS    : %10i  bit"     % c.ADC_INPUT_BITS)
        print("  A_REF             : %10.3f  V"     % c.A_REF)
        print("  MIN_N_LUT         : %10i  samples" % c.MIN_N_LUT)
        print("  MAX_N_LUT         : %10i  samples" % c.MAX_N_LUT)
        # fmt: on

        c.Fs = round(1.0 / c.SAMPLING_PERIOD, 6)
        c.F_Nyquist = round(c.Fs / 2, 6)
        c.T_SPAN_TX_BUFFER = c.BLOCK_SIZE * c.SAMPLING_PERIOD

        print("  T_SPAN_TX_BUFFER  : %10.6f  s" % c.T_SPAN_TX_BUFFER)
        print("  Fs                : {:>10,.2f}  Hz".format(c.Fs))
        print("  F_Nyquist         : {:>10,.2f}  Hz".format(c.F_Nyquist))

        """The following commands '_freq', '_offs', '_ampl' and '_wave' that
        will be sent to the Arduino will not update the LUT automatically.
        We send the wished-for settings for the reference signal all bunched up
        in a row, and only then will we compute the LUT corresponding to these
        settings. This prevents a lot of overhead by not having to wait for the
        LUT computation every time a single setting gets changed, as would be
        the case when using the methods 'set_ref_{freq/V_offs/V_ampl/waveform}'.
        """
        if any(
            x is not None
            for x in (ref_freq, ref_V_offset, ref_V_ampl, ref_waveform)
        ):
            print("\nRequesting `ref_X`\n")

        if ref_freq is not None:
            print("  freq     : {:>8,.2f}  Hz".format(ref_freq))
            success, _ans_str = self.query("_freq %f" % ref_freq)
            if not success:
                return False

        if ref_V_offset is not None:
            print("  offset   : %8.3f  V" % ref_V_offset)
            success, _ans_str = self.query("_offs %f" % ref_V_offset)
            if not success:
                return False

        if ref_V_ampl is not None:
            print("  ampl     : %8.3f  V" % ref_V_ampl)
            success, _ans_str = self.query("_ampl %f" % ref_V_ampl)
            if not success:
                return False

        if ref_waveform is not None:
            print("  waveform : %8s" % ref_waveform.name)
            success, _ans_str = self.query("_wave %i" % ref_waveform.value)
            if not success:
                return False

        if (
            not self.compute_LUT()
            or not self.query_LUT()
            or not self.query_ref()
        ):
            return False

        print("\n--- All systems GO! ---\n")
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
            Tuple(success: bool, ans_str: AnyStr)
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

    def turn_on(self) -> bool:
        """
        Returns:
            True if successful, False otherwise.
        """
        success = self.write("on")
        if success:
            self.lockin_paused = False
            self.read_until_left_over_bytes = bytearray()

        return success

    def turn_off(
        self, raises_on_timeout: bool = False
    ) -> Tuple[bool, bool, AnyStr]:
        """
        Returns:
            Tuple(
                success: bool,
                was_off: bool,
                ans_bytes: AnyStr # For debugging purposes
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
                # Note though: The Serial library does not throw an
                # exception when it actually times out! We will check for
                # zero received bytes as indication for timeout, later.
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

    def compute_LUT(self) -> bool:
        """Send command "compute_lut" to the Arduino lock-in amp to (re)compute
        the look-up table (LUT) used for the analog output of the reference
        signal `ref_X`. Any pending changes at the Arduino side to `ref_freq`,
        `ref_V_offset`, `ref_V_ampl` and `ref_waveform` will become effective.

        Returns:
            True if successful, False otherwise.
        """
        success, _ans_str = self.safe_query("c")
        return success

    def query_LUT(self) -> bool:
        """Send command "lut?" to the Arduino lock-in amp to retrieve the look-
        up table (LUT) that is used for the output reference signal `ref_X`.

        This method will update members:
            `config.N_LUT`
            `config.is_LUT_dirty`
            `config.LUT_wave`

        Returns:
            True if successful, False otherwise.
        """
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

        self.config.N_LUT = int(N_LUT[0])
        self.config.is_LUT_dirty = bool(is_LUT_dirty[0])

        # Now read the remaining LUT array from the binary stream still left in
        # the serial buffer
        try:
            ans_bytes = self.ser.read(size=self.config.N_LUT * 2)
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
            LUT_wave = np.array(
                struct.unpack("<" + "H" * self.config.N_LUT, ans_bytes),
                dtype=np.uint16,
            )
        except:
            pft("'%s' I/O ERROR: Can't unpack bytes LUT" % self.name)
            self.ser.flushInput()
            return False

        self.config.LUT_wave = LUT_wave

        if not was_paused:
            self.turn_on()

        return True

    # --------------------------------------------------------------------------
    #   query & set
    # --------------------------------------------------------------------------

    def query_ref(self) -> bool:
        """Send command "ref?" to the Arduino lock-in amp to retrieve the
        output reference signal `ref_X` settings.

        This method will update members:
            `config.ref_freq`
            `config.ref_V_offset`
            `config.ref_V_ampl`
            `config.ref_waveform`
            `config.N_LUT`

        Returns:
            True if successful, False otherwise.
        """
        print("\nObtained `ref_X`\n")

        success, ans_str = self.safe_query("?")
        if success:
            try:
                ans_list = ans_str.split("\t")

                # fmt: off
                self.config.ref_freq     = float(ans_list[0])
                self.config.ref_V_offset = float(ans_list[1])
                self.config.ref_V_ampl   = float(ans_list[2])
                self.config.ref_waveform = Waveform[ans_list[3]]
                self.config.N_LUT        = int(ans_list[4])
                # fmt: on
            except Exception as err:
                pft(err)
                return False
        else:
            return False

        # fmt: off
        print("  freq     : {:>8,.2f}  Hz".format(self.config.ref_freq))
        print("  offset   : %8.3f  V" % self.config.ref_V_offset)
        print("  ampl     : %8.3f  V" % self.config.ref_V_ampl)
        print("  waveform : %8s"      % self.config.ref_waveform.name)
        print("  N_LUT    : %8i"      % self.config.N_LUT)
        # fmt: on

        return True

    def set_ref_freq(self, ref_freq) -> bool:
        """Set the frequency [Hz] for the output reference signal `ref_X`
        at the Arduino lock-in amp. The actual obtained frequency might differ.
        The Arduino will automatically compute the new LUT.

        This method will update members:
            `config.ref_freq`
            `config.ref_V_offset`
            `config.ref_V_ampl`
            `config.ref_waveform`
            `config.N_LUT`
            `config.is_LUT_dirty`
            `config.LUT_wave`

        Returns:
            True if successful, False otherwise.
        """
        was_paused = self.lockin_paused
        if not was_paused:
            self.turn_off()

        success, _ans_str = self.query("freq %f" % ref_freq)
        if not success or not self.query_ref() or not self.query_LUT():
            return False

        if not was_paused:
            self.turn_on()

        return True

    def set_ref_V_offset(self, ref_V_offset) -> bool:
        """Set the voltage offset [V] for the output reference signal `ref_X`
        at the Arduino lock-in amp.
        The Arduino will automatically compute the new LUT.

        This method will update members:
            `config.ref_freq`
            `config.ref_V_offset`
            `config.ref_V_ampl`
            `config.ref_waveform`
            `config.N_LUT`
            `config.is_LUT_dirty`
            `config.LUT_wave`

        Returns:
            True if successful, False otherwise.
        """
        was_paused = self.lockin_paused
        if not was_paused:
            self.turn_off()

        success, _ans_str = self.query("offs %f" % ref_V_offset)
        if not success or not self.query_ref() or not self.query_LUT():
            return False

        if not was_paused:
            self.turn_on()

        return True

    def set_ref_V_ampl(self, ref_V_ampl) -> bool:
        """Set the voltage amplitude [V] for the output reference signal `ref_X`
        at the Arduino lock-in amp.
        The Arduino will automatically compute the new LUT.

        This method will update members:
            `config.ref_freq`
            `config.ref_V_offset`
            `config.ref_V_ampl`
            `config.ref_waveform`
            `config.N_LUT`
            `config.is_LUT_dirty`
            `config.LUT_wave`

        Returns:
            True if successful, False otherwise.
        """
        was_paused = self.lockin_paused
        if not was_paused:
            self.turn_off()

        success, _ans_str = self.query("ampl %f" % ref_V_ampl)
        if not success or not self.query_ref() or not self.query_LUT():
            return False

        if not was_paused:
            self.turn_on()

        return True

    def set_ref_waveform(self, ref_waveform: Waveform) -> bool:
        """Set the waveform type for the output reference signal `ref_X`
        at the Arduino lock-in amp.
        The Arduino will automatically compute the new LUT.

        This method will update members:
            `config.ref_freq`
            `config.ref_V_offset`
            `config.ref_V_ampl`
            `config.ref_waveform`
            `config.N_LUT`
            `config.is_LUT_dirty`
            `config.LUT_wave`

        Returns:
            True if successful, False otherwise.
        """
        was_paused = self.lockin_paused
        if not was_paused:
            self.turn_off()

        success, _ans_str = self.query("wave %f" % ref_waveform.value)
        if not success or not self.query_ref() or not self.query_LUT():
            return False

        if not was_paused:
            self.turn_on()

        return True

    # --------------------------------------------------------------------------
    #   read_until_EOM
    # --------------------------------------------------------------------------

    def read_until_EOM(self, size: Optional[int] = None) -> bytes:
        """Reads from the serial port until the EOM sentinel is found, the size
        is exceeded or until timeout occurs.

        Contrary to `serial.read_until()` which reads 1 byte at a time, here we
        read chunks of `2*N_BYTES_EOM`. This is way more efficient for the OS
        and drastically reduces a non-responsive GUI and dropped I/O (even
        though they are running in separate threads?!). Any left-over bytes
        after the EOM will be remembered and prefixed to the next
        `read_until_EOM()` operation.

        Returns:
            The read contents as type `bytes`.
        """
        line = bytearray()
        line[:] = self.read_until_left_over_bytes

        # fmt: off
        timeout = serial.Timeout(self.ser._timeout)  # pylint: disable=protected-access
        # fmt: on

        while True:
            try:
                c = self.ser.read(2 * self.config.N_BYTES_EOM)
            except:
                # Remain silent
                break

            if c:
                line += c
                line_tail = line[-4 * self.config.N_BYTES_EOM :]
                i_found_terminator = line_tail.find(self.config.EOM)
                if i_found_terminator > -1:
                    N_left_over_bytes_after_EOM = (
                        len(line_tail)
                        - i_found_terminator
                        - self.config.N_BYTES_EOM
                    )

                    if N_left_over_bytes_after_EOM:
                        left_over_bytes = line_tail[
                            -N_left_over_bytes_after_EOM:
                        ]
                        line = line[:-N_left_over_bytes_after_EOM]
                        # print(N_left_over_bytes_after_EOM)
                        # print(left_over_bytes)
                    else:
                        left_over_bytes = bytearray()

                    self.read_until_left_over_bytes = left_over_bytes
                    break
                if size is not None and len(line) >= size:
                    break
            else:
                break

            if timeout.expired():
                break

        return bytes(line)

    # --------------------------------------------------------------------------
    #   listen_to_lockin_amp
    # --------------------------------------------------------------------------

    def listen_to_lockin_amp(
        self,
    ) -> Tuple[
        bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[int]
    ]:
        """Reads incoming data packets coming from the lock-in amp. This method
        is blocking until it receives an EOM (end-of-message) sentinel or until
        it times out.

        Returns:
            Tuple (
                success: bool
                time: numpy.array
                ref_X: numpy.array
                ref_Y: numpy.array
                sig_I: numpy.array
                counter: Optional(int)
            )
        """
        c = self.config  # Shorthand alias

        ans_bytes = self.read_until_EOM()
        # dprint("EOM found with %i bytes and..." % len(ans_bytes))
        if not ans_bytes[: c.N_BYTES_SOM] == c.SOM:
            dprint("'%s' I/O ERROR: No SOM found" % self.name)
            return False, [np.nan], [np.nan], [np.nan], [np.nan], None

        # dprint("SOM okay")
        if not len(ans_bytes) == c.N_BYTES_TX_BUFFER:
            dprint(
                "'%s' I/O ERROR: Expected %i bytes but received %i"
                % (self.name, c.N_BYTES_TX_BUFFER, len(ans_bytes))
            )
            return False, [np.nan], [np.nan], [np.nan], [np.nan], None

        # fmt: off
        ans_bytes = ans_bytes[c.N_BYTES_SOM : -c.N_BYTES_EOM] # Remove sentinels
        bytes_counter   = ans_bytes[0:4]    # Header
        bytes_millis    = ans_bytes[4:8]    # Header
        bytes_micros    = ans_bytes[8:10]   # Header
        bytes_idx_phase = ans_bytes[10:12]  # Header
        bytes_sig_I     = ans_bytes[12:]    # Body
        # fmt: on

        try:
            counter = struct.unpack("<" + c.binary_type_counter, bytes_counter)
            millis = struct.unpack("<" + c.binary_type_millis, bytes_millis)
            micros = struct.unpack("<" + c.binary_type_micros, bytes_micros)
            idx_phase = struct.unpack(
                "<" + c.binary_type_idx_phase, bytes_idx_phase
            )

            sig_I = np.array(
                struct.unpack(
                    "<" + c.binary_type_sig_I * c.BLOCK_SIZE, bytes_sig_I
                ),
                dtype=c.return_type_sig_I,
            )
        except:
            dprint("'%s' I/O ERROR: Can't unpack bytes" % self.name)
            return False, [np.nan], [np.nan], [np.nan], [np.nan], None

        # fmt: off
        counter   = counter[0]
        millis    = millis[0]
        micros    = micros[0]
        idx_phase = idx_phase[0]
        # fmt: on

        # dprint("%i %i" % (millis, micros))
        t0 = millis * 1000 + micros
        time = np.arange(0, c.BLOCK_SIZE)
        time = t0 + time * c.SAMPLING_PERIOD * 1e6
        time = np.asarray(time, dtype=c.return_type_time, order="C")

        idxs_phase = np.arange(idx_phase, idx_phase + c.N_LUT) % c.N_LUT
        phi = 2 * np.pi * idxs_phase / c.N_LUT

        # DEBUG test: Add artificial phase delay between ref_X/Y and sig_I
        """
        if 0:
            phase_delay_deg = 50
            phi = np.unwrap(phi + phase_delay_deg / 180 * np.pi)
        """

        sig_I = sig_I * c.A_REF / (2 ** c.ADC_INPUT_BITS - 1)

        if c.ref_waveform == Waveform.Cosine:
            lut_X = 0.5 * (1 + np.cos(phi))
            lut_Y = 0.5 * (1 + np.sin(phi))

        elif c.ref_waveform == Waveform.Square:
            lut_X = round_C_style(
                ((1.75 * c.N_LUT - idxs_phase) % c.N_LUT) / (c.N_LUT - 1)
            )
            # lut_Y = round_C_style(((1.50 * c.N_LUT - idxs_phase) % c.N_LUT) /
            #                      (c.N_LUT - 1))
            lut_Y = np.interp(
                np.arange(c.N_LUT) + c.N_LUT / 4,
                np.arange(c.N_LUT * 2),
                np.tile(lut_X, 2),
            )

        elif c.ref_waveform == Waveform.Triangle:
            lut_X = 2 * np.abs(idxs_phase / c.N_LUT - 0.5)
            lut_Y = 2 * np.abs(
                ((idxs_phase - c.N_LUT / 4) % c.N_LUT) / c.N_LUT - 0.5
            )

        elif c.ref_waveform == Waveform.Unknown:
            lut_X = np.full(np.nan, c.N_LUT)
            lut_Y = np.full(np.nan, c.N_LUT)

        lut_X = (c.ref_V_offset - c.ref_V_ampl) + 2 * c.ref_V_ampl * lut_X
        lut_Y = (c.ref_V_offset - c.ref_V_ampl) + 2 * c.ref_V_ampl * lut_Y
        lut_X.clip(0, c.A_REF)
        lut_Y.clip(0, c.A_REF)

        ref_X_tiled = np.tile(lut_X, int(np.ceil(c.BLOCK_SIZE / c.N_LUT)))
        ref_Y_tiled = np.tile(lut_Y, int(np.ceil(c.BLOCK_SIZE / c.N_LUT)))

        ref_X = np.asarray(
            ref_X_tiled[: c.BLOCK_SIZE], dtype=c.return_type_ref_XY, order="C"
        )

        ref_Y = np.asarray(
            ref_Y_tiled[: c.BLOCK_SIZE], dtype=c.return_type_ref_XY, order="C"
        )

        return True, time, ref_X, ref_Y, sig_I, counter


# ------------------------------------------------------------------------------
#   Main: Will show a demo when run from the terminal
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    alia = Alia()
    alia.auto_connect("port_data.txt")

    if not alia.is_alive:
        sys.exit(0)

    alia.begin(
        ref_freq=220,
        ref_V_offset=2,
        ref_V_ampl=1,
        ref_waveform=Waveform.Cosine,
    )

    alia.turn_on()

    reply = alia.listen_to_lockin_amp()
    print("success: %s" % reply[0])
    print("time   : %s" % reply[1])
    print("ref_X  : %s" % reply[2])
    print("ref_Y  : %s" % reply[3])
    print("sig_I  : %s" % reply[4])

    alia.set_ref_freq(330)
    alia.set_ref_V_offset(1.5)
    alia.set_ref_V_ampl(0.5)
    alia.set_ref_waveform(Waveform.Square)

    reply = alia.listen_to_lockin_amp()
    print("\nsuccess: %s" % reply[0])
    print("time   : %s" % reply[1])
    print("ref_X  : %s" % reply[2])
    print("ref_Y  : %s" % reply[3])
    print("sig_I  : %s" % reply[4])

    alia.turn_off()
    alia.close()
