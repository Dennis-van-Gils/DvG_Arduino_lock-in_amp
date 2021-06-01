#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module to communicate with an Arduino lock-in amplifier device over a serial
connection.
"""
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/DvG_Arduino_lock-in_amp"
__date__ = "31-05-2021"
__version__ = "1.5.0"  # Inbetween version
# pylint: disable=bare-except, broad-except, pointless-string-statement, invalid-name

import sys
import struct
from enum import Enum
from typing import AnyStr, Optional, Tuple
import time as Time

import serial
import numpy as np

from dvg_devices import Arduino_protocol_serial
from dvg_debug_functions import dprint, print_fancy_traceback as pft


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

        # Data types to decode from binary stream
        binary_type_time  = "I"  # [uint32_t]
        binary_type_ref_X = "H"  # [uint16_t]
        binary_type_sig_I = "h"  # [int16_t]

        # Return types
        return_type_time  = int  # Signed to allow for flexible arithmetic
        return_type_ref_X = float
        return_type_sig_I = float

        # Lock-in amplifier CONSTANTS
        SAMPLING_PERIOD   = 0  # [s]
        BLOCK_SIZE        = 0  # Number of samples send per TX_buffer
        N_BYTES_TX_BUFFER = 0  # [data bytes] Expected number of bytes for each
                               # correctly received TX_buffer from the Arduino
        DAC_OUTPUT_BITS   = 0  # [bits]
        ADC_INPUT_BITS    = 0  # [bits]
        A_REF             = 0  # [V] Analog voltage reference of the Arduino

        # Derived settings
        Fs        = 0          # [Hz] Sampling rate
        F_Nyquist = 0          # [Hz] Nyquist frequency
        T_SPAN_TX_BUFFER = 0   # [s]  Time interval spanned by a single TX_buffer

        # Waveform look-up table (LUT) settings
        N_LUT = 0              # Number of samples covering a full period

        # Reference signal parameters
        ref_freq     = 0       # [Hz] Frequency
        ref_V_offset = 0       # [V]  Voltage offset
        ref_V_ampl   = 0       # [V]  Voltage amplitude
        ref_waveform = Waveform.Unknown  # Waveform enum
        # fmt: on

    def __init__(
        self,
        name="Alia",
        long_name="Arduino lock-in amplifier",
        connect_to_specific_ID=None,
        baudrate=1e6,
        read_timeout=1,
        write_timeout=1,
    ):
        super().__init__(
            name=name,
            long_name=long_name,
            connect_to_specific_ID=connect_to_specific_ID,
        )

        # Hack to provide legacy support
        self.set_ID_validation_query(
            ID_validation_query=self.ID_validation_query,
            valid_ID_broad="Arduino lock-in amp",
            valid_ID_specific=None,
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
        freq: Optional[float] = None,
        V_offset: Optional[float] = None,
        V_ampl: Optional[float] = None,
        waveform: Optional[Waveform] = None,
    ) -> bool:
        """
        Args:
            waveform (None):
                Not used. Provides future support.

        Returns:
            True if successful, False otherwise.
        """
        success, _foo, _bar = self.turn_off()
        if not success:
            return False

        # Shorthand alias
        c = self.config

        print("Lock-in constants")
        print("─────────────────\n")
        success, ans_str = self.safe_query("config?")
        if success:
            try:
                ans_list = ans_str.split("\t")
                # fmt: off
                c.SAMPLING_PERIOD   = float(ans_list[0]) * 1e-6
                c.BLOCK_SIZE        = int(ans_list[1])
                c.N_BYTES_TX_BUFFER = int(ans_list[2])
                c.N_LUT             = int(ans_list[3])
                c.DAC_OUTPUT_BITS   = int(ans_list[4])
                c.ADC_INPUT_BITS    = int(ans_list[5])
                c.A_REF             = float(ans_list[6])
                c.ref_V_offset      = float(ans_list[7])
                c.ref_V_ampl        = float(ans_list[8])
                c.ref_freq          = float(ans_list[9])
                # fmt: on
            except Exception as err:
                pft(err)
                return False
        else:
            return False

        c.Fs = round(1.0 / c.SAMPLING_PERIOD, 6)
        c.F_Nyquist = round(c.Fs / 2, 6)
        c.T_SPAN_TX_BUFFER = c.BLOCK_SIZE * c.SAMPLING_PERIOD

        def fancy(name, value, value_format, unit=""):
            format_str = "{:>16s}  %s  {:<s}" % value_format
            print(format_str.format(name, value, unit))

        fancy("Fs", c.Fs, "{:>9,.2f}", "Hz")
        fancy("F_Nyquist", c.F_Nyquist, "{:>9,.2f}", "Hz")
        fancy("sampling period", c.SAMPLING_PERIOD * 1e6, "{:>9.3f}", "us")
        fancy("block size", c.BLOCK_SIZE, "{:>9d}", "samples")
        fancy("TX buffer", c.N_BYTES_TX_BUFFER, "{:>9d}", "bytes")
        fancy("TX buffer", c.T_SPAN_TX_BUFFER, "{:>9.3f}", "s")
        fancy(
            "TX data rate",
            c.N_BYTES_TX_BUFFER * c.Fs / c.BLOCK_SIZE / 1024,
            "{:>9,.1f}",
            "kb/s",
        )
        fancy("DAC output", c.DAC_OUTPUT_BITS, "{:>9d}", "bit")
        fancy("ADC input", c.ADC_INPUT_BITS, "{:>9d}", "bit")
        fancy("A_ref", c.A_REF, "{:>9.3f}", "V")

        self.set_ref(freq, V_offset, V_ampl, waveform)

        print("┌─────────────────────────┐")
        print("│     All systems GO!     │")
        print("└─────────────────────────┘\n")
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

    def turn_on(self, reset_timer: bool = False) -> bool:
        """

        Args:
            reset_timer (bool):
                Not used. Provides future support.

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
    #   set_ref
    # --------------------------------------------------------------------------

    def set_ref(
        self,
        freq: Optional[float] = None,
        V_offset: Optional[float] = None,
        V_ampl: Optional[float] = None,
        waveform: Optional[Waveform] = None,
    ) -> bool:
        """Request new parameters to be set of the output reference signal
        `ref_X` at the Arduino. The Arduino will compute the new LUT, based on
        the obtained parameters, and will send the new LUT and obtained
        `ref_X` parameters back. The actually obtained parameters might differ
        from the requested ones, noticably the frequency.

        This method will update members:
            `config.ref_freq`
            `config.ref_V_offset`
            `config.ref_V_ampl`
            `config.N_LUT`

        Args:
            freq (float):
                The requested frequency in Hz.

            V_offset (float):
                The requested voltage offset in V.

            V_ampl (float):
                The requested voltage amplitude in V.

            waveform (None):
                Not used. Provides future support.

        Returns:
            True if successful, False otherwise.
        """
        was_paused = self.lockin_paused
        if not was_paused:
            self.turn_off()

        if freq is not None:
            success, ans_str = self.safe_query("ref_freq %f" % freq)
            if success:
                self.config.ref_freq = float(ans_str)
            else:
                return False

        if V_offset is not None:
            success, ans_str = self.safe_query("ref_V_offset %f" % V_offset)
            if success:
                self.config.ref_V_offset = float(ans_str)
            else:
                return False

        if V_ampl is not None:
            success, ans_str = self.safe_query("ref_V_ampl %f" % V_ampl)
            if success:
                self.config.ref_V_ampl = float(ans_str)
            else:
                return False

        if not was_paused:
            self.turn_on()

        # fmt: off
        print("\nReference signal `ref_X`")
        print("────────────────────────\n")
        print("            requested   obtained")
        if freq is not None:
            print("      freq  {:>9,.2f}  {:>9,.2f}  Hz"
                .format(freq, self.config.ref_freq))
        else:
            print("      freq  {:>9s}  {:>9,.2f}  Hz"
                .format("-", self.config.ref_freq))
        if V_offset is not None:
            print("    offset  {:>9,.3f}  {:>9,.3f}  V"
                .format(V_offset, self.config.ref_V_offset))
        else:
            print("    offset  {:>9s}  {:>9,.3f}  V"
                .format("-", self.config.ref_V_offset))
        if V_ampl is not None:
            print("      ampl  {:>9,.3f}  {:>9,.3f}  V"
                .format(V_ampl, self.config.ref_V_ampl))
        else:
            print("      ampl  {:>9s}  {:>9,.3f}  V"
                .format("-", self.config.ref_V_ampl))
        """
        if waveform is not None:
            print("  waveform  {:>9s}  {:>9s}"
                .format(waveform.name, self.config.ref_waveform.name))
        else:
            print("  waveform  {:>9s}  {:>9s}"
                .format("-", self.config.ref_waveform.name))
        """
        print("     N_LUT  {:>9s}  {:>9d}\n".format("-", self.config.N_LUT))
        # fmt: on

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

        # fmt: off
        timeout = serial.Timeout(self.ser._timeout)  # pylint: disable=protected-access
        # fmt: on

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
        bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[int]
    ]:
        """Reads incoming data packets coming from the lock-in amp. This method
        is blocking until it receives an EOM (end-of-message) sentinel or until
        it times out.

        Returns:
            Tuple (
                success: bool
                counter: ---- NOT USED ----
                time   : numpy.array, units [us]
                ref_X  : numpy.array, units [V]
                ref_Y  : numpy.array, units [V]
                sig_I  : numpy.array, units [V]
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

        # fmt: off
        end_byte_time  = c.BLOCK_SIZE * struct.calcsize(c.binary_type_time)
        end_byte_ref_X = end_byte_time + c.BLOCK_SIZE * struct.calcsize(
            c.binary_type_ref_X
        )
        end_byte_sig_I = end_byte_ref_X + c.BLOCK_SIZE * struct.calcsize(
            c.binary_type_sig_I
        )
        ans_bytes   = ans_bytes[c.N_BYTES_SOM : -c.N_BYTES_EOM]
        bytes_time  = ans_bytes[0:end_byte_time]
        bytes_ref_X = ans_bytes[end_byte_time:end_byte_ref_X]
        bytes_sig_I = ans_bytes[end_byte_ref_X:end_byte_sig_I]
        # fmt: on

        try:
            time = np.array(
                struct.unpack(
                    "<" + c.binary_type_time * c.BLOCK_SIZE, bytes_time
                ),
                dtype=c.return_type_time,
            )
            ref_X_phase = np.array(
                struct.unpack(
                    "<" + c.binary_type_ref_X * c.BLOCK_SIZE, bytes_ref_X
                ),
                dtype=c.return_type_ref_X,
            )
            sig_I = np.array(
                struct.unpack(
                    "<" + c.binary_type_sig_I * c.BLOCK_SIZE, bytes_sig_I
                ),
                dtype=c.return_type_sig_I,
            )
        except:
            dprint("'%s' I/O ERROR: Can't unpack bytes" % self.name)
            return failed

        phi = 2 * np.pi * ref_X_phase / c.N_LUT

        # DEBUG test: Add artificial phase delay between ref_X/Y and sig_I
        """
        if 0:
            phase_delay_deg = 50
            phi = np.unwrap(phi + phase_delay_deg / 180 * np.pi)
        """

        ref_X = (c.ref_V_offset + c.ref_V_ampl * np.cos(phi)).clip(0, c.A_REF)
        ref_Y = (c.ref_V_offset + c.ref_V_ampl * np.sin(phi)).clip(0, c.A_REF)
        sig_I = sig_I / (2 ** c.ADC_INPUT_BITS - 1) * c.A_REF
        sig_I = sig_I * 2  # Compensate for differential mode of Arduino

        return True, np.nan, time, ref_X, ref_Y, sig_I


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

    alia.begin(freq=220, V_offset=0.25, V_ampl=0.33)
    # alia.begin()
    alia.turn_on()

    print_reply(alia.listen_to_lockin_amp())
    print_reply(alia.listen_to_lockin_amp())
    alia.set_ref(freq=250, V_offset=1.5, V_ampl=0.5)
    print_reply(alia.listen_to_lockin_amp())
    print_reply(alia.listen_to_lockin_amp())

    alia.turn_off()
    alia.close()
