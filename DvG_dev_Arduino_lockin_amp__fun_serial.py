#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module to communicate with an Arduino lock-in amplifier device over a serial
connection.
"""
__author__      = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__         = "https://github.com/Dennis-van-Gils/DvG_Arduino_lock-in_amp"
__date__        = "21-08-2019"
__version__     = "1.0.0"

import sys
import struct

import serial
import numpy as np
from enum import Enum

import DvG_dev_Arduino__fun_serial as Arduino_functions
from DvG_debug_functions import dprint, print_fancy_traceback as pft

def round_C_style(values: np.ndarray):
    sign_backup = np.sign(values)
    values = np.abs(values)
    
    values_trunc = np.trunc(values)
    values_frac  = values % 1
    
    frac_rounded = np.zeros_like(values)
    frac_rounded[values_frac >= 0.5] = 1
    
    values_rounded = sign_backup * (values_trunc + frac_rounded)
    
    return (values_rounded)
 
class Waveform(Enum):
    Unknown  = -1
    Cosine   = 0
    Square   = 1
    Sawtooth = 2
    Triangle = 3

class Arduino_lockin_amp(Arduino_functions.Arduino):
    class Config():
        # Serial communication sentinels: start and end of message
        SOM = b'\x00\x80\x00\x80\x00\x80\x00\x80\x00\x80'
        EOM = b'\xff\x7f\x00\x00\xff\x7f\x00\x00\xff\x7f'
        N_BYTES_SOM = len(SOM)
        N_BYTES_EOM = len(EOM) 
        
        # Data types to decode from binary streams
        binary_type_counter   = 'I'   # [uint32_t] TX_buffer header
        binary_type_millis    = 'I'   # [uint32_t] TX_buffer header
        binary_type_micros    = 'H'   # [uint16_t] TX_buffer header
        binary_type_idx_phase = 'H'   # [uint16_t] TX_buffer header
        binary_type_sig_I     = 'H'   # [uint16_t] TX_buffer body
        
        # Return types
        return_type_time   = np.float64   # Ensure signed to allow for flexible arithmetic
        return_type_ref_XY = np.float64
        return_type_sig_I  = np.float64
        
        # Microcontroller unit (mcu) info
        mcu_firmware = ''   # Firmware version
        mcu_model    = ''   # Chipset model
        mcu_uid      = ''   # Unique identifier of the chip (serial number)
        
        # Lock-in amplifier CONSTANTS
        SAMPLING_PERIOD   = 0   # [s]
        BLOCK_SIZE        = 0   # Number of samples send per TX_buffer
        N_BYTES_TX_BUFFER = 0   # [data bytes] Expected number of bytes for each
                                # correctly received TX_buffer from the Arduino
        DAC_OUTPUT_BITS   = 0   # [bits]
        ADC_INPUT_BITS    = 0   # [bits]
        A_REF             = 0   # [V] Analog voltage reference of the Arduino
        MIN_N_LUT         = 0   # Minimum allowed number of LUT samples
        MAX_N_LUT         = 0   # Maximum allowed number of LUT samples
        
        # Derived settings
        Fs        = 0         # [Hz] Sampling rate
        F_Nyquist = 0         # [Hz] Nyquist frequency
        T_SPAN_TX_BUFFER = 0  # [s]  Time interval spanned by a single TX_buffer
        
        # Waveform look-up table (LUT) settings
        N_LUT = 0             # Number of samples covering a full period
        LUT_wave = np.array([], dtype=np.uint16)
        # LUT_wave will contain a copy of the LUT array of the current reference
        # signal waveform as used on the Arduino side. This array will be used
        # to reconstruct the ref_X and ref_Y signals, based on the phase index
        # that is sent in the header of each TX_buffer. The unit of each element
        # in the array is the bit-value that is sent out over the DAC of the
        # Arduino. Hence, multiply by A_REF/(2**ADC_INPUT_BITS - 1) to get
        # units of [V].
                
        # Reference signal settings
        ref_freq     = 0      # [Hz] Frequency
        ref_V_offset = 0      # [V]  Voltage offset
        ref_V_ampl   = 0      # [V]  Voltage amplitude
        ref_waveform = Waveform.Unknown # Waveform enum
    
    def __init__(self, 
                 name="Lockin",
                 baudrate=1.2e6,
                 read_timeout=1,
                 write_timeout=1,
                 read_term_char='\n',
                 write_term_char='\n'):
        super(Arduino_lockin_amp, self).__init__()
        
        self.name = name
        self.baudrate = baudrate
        self.read_timeout  = read_timeout
        self.write_timeout = write_timeout
        self.read_term_char  = read_term_char
        self.write_term_char = write_term_char
        
        self.read_until_left_over_bytes = bytearray()
        
        self.config = self.Config()
        self.lockin_paused = True

    def begin(self, ref_freq=None, ref_V_offset=None, ref_V_ampl=None,
              ref_waveform=None):
        """ Prepare the lock-in amp for operation. The start-up state is off.
        If the optional parameters ref_freq, ref_V_offset, ref_V_ampl or
        ref_waveform are not passed, the pre-existing values known to the
        Arduino will be used instead, i.e. it will pick up where it left.
        
        Returns:
            success
        """
        [success, __foo, __bar] = self.turn_off()
        if not success: return False
        
        # Shorthand alias
        c = self.config
        
        print("Retrieving 'mcu?'")
        [success, ans_str] = self.safe_query("mcu?")
        if success:
            try:
                ans_list = ans_str.split('\t')
                c.mcu_firmware = ans_list[0]
                c.mcu_model    = ans_list[1]
                c.mcu_uid      = ans_list[2]
            except Exception as err:
                raise(err)
                return False
        else: return False
        print("  firmware: %s" % c.mcu_firmware)
        print("  model   : %s" % c.mcu_model)
        print("  uid     : %s" % c.mcu_uid)
        
        print("\nRetrieving 'const?'")
        [success, ans_str] = self.safe_query("const?")
        if success:
            try:
                ans_list = ans_str.split('\t')
                # Round SAMPLING PERIOD to nanosecond resolution to
                # prevent e.g. 1.0 being stored as 0.9999999999
                c.SAMPLING_PERIOD   = (round(float(ans_list[0])*1e-6, 9))
                c.BLOCK_SIZE        = int(ans_list[1])
                c.N_BYTES_TX_BUFFER = int(ans_list[2])
                c.DAC_OUTPUT_BITS   = int(ans_list[3])
                c.ADC_INPUT_BITS    = int(ans_list[4] )
                c.A_REF             = float(ans_list[5])
                c.MIN_N_LUT         = int(ans_list[6])
                c.MAX_N_LUT         = int(ans_list[7])
            except Exception as err:
                raise(err)
                return False
        else: return False
        print("  SAMPLING_PERIOD  : %-10.3f [us]"    % (c.SAMPLING_PERIOD*1e6))
        print("  BLOCK_SIZE       : %-10i [samples]" % c.BLOCK_SIZE)
        print("  N_BYTES_TX_BUFFER: %-10i [bytes]"   % c.N_BYTES_TX_BUFFER)
        print("  DAC_OUTPUT_BITS  : %-10i [bit]"     % c.DAC_OUTPUT_BITS)
        print("  ADC_INPUT_BITS   : %-10i [bit]"     % c.ADC_INPUT_BITS)
        print("  A_REF            : %-10.3f [V]"     % c.A_REF)
        print("  MIN_N_LUT        : %-10i [samples]" % c.MIN_N_LUT)
        print("  MAX_N_LUT        : %-10i [samples]" % c.MAX_N_LUT)
        
        c.Fs = round(1.0/c.SAMPLING_PERIOD, 6)
        c.F_Nyquist = round(c.Fs/2, 6)
        c.T_SPAN_TX_BUFFER = (c.BLOCK_SIZE * c.SAMPLING_PERIOD)
        
        print("  T_SPAN_TX_BUFFER : %-10.6f [s]"   % c.T_SPAN_TX_BUFFER)
        print("  Fs               : %-10.6f [kHz]" % (c.Fs/1e3))
        print("  F_Nyquist        : %-10.6f [kHz]" % (c.F_Nyquist/1e3))
        
        """The following commands '_freq', '_offs', '_ampl' and '_wave' that
        will be sent to the Arduino will not update the LUT automatically, nor
        will there be any reply back from the Arduino when correctly received.
        Instead, we send the wished-for settings for the reference signal all
        bunched up in a row, and only then will we compute the LUT
        corresponding to these settings. This prevents a lot of overhead by
        not having to wait for the LUT computation every time a single setting
        gets changed, as would be the case when using the methods
        'set_ref_{freq/V_offs/V_ampl/waveform}'.
        """
        if any(x is not None for x in (ref_freq, ref_V_offset, ref_V_ampl,
                                       ref_waveform)):
            print("\nRequesting set reference signal")

        if ref_freq != None:
            print("  freq             : %-10.3f [Hz]" % ref_freq)
            if not self.write("_freq %f" % ref_freq)          : return False
        if ref_V_offset != None:
            print("  offset           : %-10.3f [V]" % ref_V_offset)
            if not self.write("_offs %f" % ref_V_offset)      : return False
        if ref_V_ampl != None:
            print("  ampl             : %-10.3f [V]" % ref_V_ampl)
            if not self.write("_ampl %f" % ref_V_ampl)        : return False
        if ref_waveform != None:
            print("  waveform         : %-10s" % ref_waveform.name)
            if not self.write("_wave %i" % ref_waveform.value): return False        
        
        if not self.compute_LUT(): return False
        if not self.query_LUT()  : return False
        if not self.query_ref()  : return False
        
        print("\n--- All systems GO! ---\n")        
        return True
    
    def safe_query(self, msg_str, timeout_warning_style=1):
        """
        Returns:
            [success, ans_str]
        """
        was_paused = self.lockin_paused
        if not was_paused: self.turn_off()
        
        [success, ans_str] = self.query(msg_str, timeout_warning_style)
            
        if success and not was_paused: self.turn_on()
        return [success, ans_str]
        
    def turn_on(self):
        """
        Returns:
            success
        """
        success = self.write("on")
        if success:
            self.lockin_paused = False
            self.read_until_left_over_bytes = bytearray()
        
        return success
        
    def turn_off(self, timeout_warning_style=1):
        """
        Returns:
            success
            was_off
            ans_bytes: for debugging purposes
        """
        success = False
        was_off = True
        ans_bytes = b''
        
        # First ensure the lock-in amp will switch off if not already so.
        self.ser.flushInput() # Essential to clear potential large amount of
                              # binary data waiting in the buffer to be read
        if self.write("off", timeout_warning_style):
            self.ser.flushOutput() # Send out 'off' as fast as possible
            
            # Check for acknowledgement reply        
            try:
                ans_bytes = self.ser.read_until("off\n".encode())
                #print(len(ans_bytes))
                #print("found off: ", end ='')
                #print(ans_bytes[-4:])
            except (serial.SerialTimeoutException,
                    serial.SerialException) as err:
                # Note though: The Serial library does not throw an
                # exception when it actually times out! We will check for
                # zero received bytes as indication for timeout, later.
                pft(err, 3)
            except Exception as err:
                pft(err, 3)
                sys.exit(1)
            else:
                if (len(ans_bytes) == 0):
                    # Received 0 bytes, probably due to a timeout.
                    if timeout_warning_style == 1:
                        pft("Received 0 bytes. Read probably timed out.", 3)
                    elif timeout_warning_style == 2:
                        raise(serial.SerialTimeoutException)
                else:
                    try:
                        was_off = (ans_bytes[-12:] == b"already_off\n")
                    except:
                        pass
                    success = True
                    self.lockin_paused = True
    
        return [success, was_off, ans_bytes]
    
    def compute_LUT(self):
        """ Send command 'compute_LUT' to the Arduino lock-in amp to make the
        planned ref_freq, ref_V_offset, ref_V_ampl and ref_waveform become
        effective when the lock-in amp is turned on.
        
        Returns:
            success
        """

        return self.write("c");

        """
        print("\nComputing waveform on Arduino... ", end='')
        [success, ans_str] = self.safe_query("compute_LUT")
        if success:
            # The first reply from the Arduino is the message
            # "Computing LUT..\n" which we will ignore.
            pass
        else: return False
        
        # The second reply of the Arduino will be received when it has finished
        # computing the LUT. This can take a long time, so we enlarge the
        # serial timeout for this operation.
        timeout_backup = self.ser.timeout
        self.ser.timeout = 120
        try:
            ans_bytes = self.ser.read_until(b'\n')
        except Exception as err:
            self.ser.timeout = timeout_backup
            raise(err)
            return False
        self.ser.timeout = timeout_backup
        
        if (len(ans_bytes) == 0):
            # Received 0 bytes, probably due to a timeout.
            pft("'%s' I/O ERROR: Timed out computing LUT" % self.name)
            return False
        
        print(ans_bytes.decode().strip()) # "done in ### ms"
        return True
        """
    
    def query_LUT(self):
        """
        Returns:
            success
        """
        
        # The query "lut?" will first return an ASCII encoded line, terminated
        # by a newline character, and it will then send a binary stream
        # terminated with another newline character.
        # Hence, the upcoming query will only return the first ASCII line with
        # the binary stream still left in the serial buffer.
        [success, ans_str] = self.safe_query("lut?")
        if success:
            try:
                ans_list = ans_str.split('\t')
                self.config.N_LUT = int(ans_list[0])
            except Exception as err:
                raise(err)
                return False
        else: return False
        
        # Read the binary stream still left in the serial buffer
        try:
            ans_bytes = self.ser.read(size=self.config.N_LUT * 2 + 1)
        except:
            pft("'%s' I/O ERROR: Can't read bytes LUT" % self.name)
            self.ser.flushInput()
            return False
        
        if (len(ans_bytes) == 0):
            # Received 0 bytes, probably due to a timeout.
            pft("'%s' I/O ERROR: Timed out reading LUT" % self.name)
            self.ser.flushInput()
            return False
        
        try:
            LUT_wave = np.array(
                    struct.unpack('<' + 'H'*self.config.N_LUT, ans_bytes[:-1]),
                    dtype=np.uint16)
        except:
            pft("'%s' I/O ERROR: Can't unpack bytes LUT" % self.name)
            self.ser.flushInput()
            return False
        
        self.config.LUT_wave = LUT_wave
        return True
    
    def query_ref(self):
        """
        Returns:
            success
        """
        print("\nRetrieving 'ref?'")
        [success, ans_str] = self.safe_query("ref?")
        if success:
            try:
                ans_list = ans_str.split('\t')
                self.config.ref_freq     = float(ans_list[0])
                self.config.ref_V_offset = float(ans_list[1])
                self.config.ref_V_ampl   = float(ans_list[2])
                self.config.ref_waveform = Waveform[ans_list[3]]
                self.config.N_LUT        = int(ans_list[4])
            except Exception as err:
                raise(err)
                return False
        else: return False
        
        print("  freq             : %-10.3f [Hz]" % self.config.ref_freq)
        print("  offset           : %-10.3f [V]"  % self.config.ref_V_offset)
        print("  ampl             : %-10.3f [V]"  % self.config.ref_V_ampl)
        print("  waveform         : %-10s"        % self.config.ref_waveform.name)
        print("  N_LUT            : %-10i"        % self.config.N_LUT)
        return True
        
    def set_ref_freq(self, ref_freq):
        """
        Returns:
            success
        """
        was_paused = self.lockin_paused
        if not was_paused: self.turn_off()
            
        if not self.write("_freq %f" % ref_freq): return False
        if not self.compute_LUT(): return False
        if not self.query_LUT()  : return False
        if not self.query_ref()  : return False
        
        if not was_paused: self.turn_on()        
        return True
    
    def set_ref_V_offset(self, ref_V_offset):
        """
        Returns:
            success
        """
        was_paused = self.lockin_paused
        if not was_paused: self.turn_off()
            
        if not self.write("_offs %f" % ref_V_offset): return False
        if not self.compute_LUT(): return False
        if not self.query_LUT()  : return False
        if not self.query_ref()  : return False
        
        if not was_paused: self.turn_on()        
        return True
    
    def set_ref_V_ampl(self, ref_V_ampl):
        """
        Returns:
            success
        """
        was_paused = self.lockin_paused
        if not was_paused: self.turn_off()
            
        if not self.write("_ampl %f" % ref_V_ampl): return False
        if not self.compute_LUT(): return False
        if not self.query_LUT()  : return False
        if not self.query_ref()  : return False
        
        if not was_paused: self.turn_on()        
        return True
    
    def set_ref_waveform(self, ref_waveform: Waveform):
        """
        Returns:
            success
        """
        was_paused = self.lockin_paused
        if not was_paused: self.turn_off()
            
        if not self.write("_wave %f" % ref_waveform.value): return False
        if not self.compute_LUT(): return False
        if not self.query_LUT()  : return False
        if not self.query_ref()  : return False
        
        if not was_paused: self.turn_on()        
        return True
    
    def read_until_EOM(self, size=None):
        """Reads from the serial port until the EOM sentinel is found, the size
        is exceeded or until timeout occurs.
        
        Contrary to 'serial.read_until' which reads 1 byte at a time, here we
        read chunks of 2*N_BYTES_EOM. This is way more efficient for the OS
        and drastically reduces a non-responsive GUI and dropped I/O (even
        though they are running in separate threads?!). Any left-over bytes
        after the EOM will be remembered and prefixed to the next read_until_EOM
        operation.
        """
        line = bytearray()
        line[:] = self.read_until_left_over_bytes
        timeout = serial.Timeout(self.ser._timeout)
        while True:
            try:
                c = self.ser.read(2*self.config.N_BYTES_EOM)
            except:
                # Remain silent
                break
        
            if c:
                line += c
                line_tail = line[-4*self.config.N_BYTES_EOM:]
                i_found_terminator = line_tail.find(self.config.EOM)
                if i_found_terminator > -1:
                    N_left_over_bytes_after_EOM = (
                            len(line_tail) - i_found_terminator - 
                            self.config.N_BYTES_EOM)
                    
                    if N_left_over_bytes_after_EOM:
                        left_over_bytes = line_tail[-N_left_over_bytes_after_EOM:]
                        line = line[:-N_left_over_bytes_after_EOM]
                        #print(N_left_over_bytes_after_EOM)
                        #print(left_over_bytes)
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
        
    def listen_to_lockin_amp(self):
        """Reads incoming data packets coming from the lock-in amp. This method
        is blocking until it receives an 'end-of-message' sentinel or until it
        times out.
        
        Returns:
            success
            time
            ref_X
            ref_Y
            sig_I
        """
        c = self.config  # Shorthand alias
        
        ans_bytes = self.read_until_EOM()
        #dprint("EOM found with %i bytes and..." % len(ans_bytes))
        if not (ans_bytes[:c.N_BYTES_SOM] == c.SOM):
            dprint("'%s' I/O ERROR: No SOM found" % self.name)
            return [False, [np.nan], [np.nan], [np.nan], [np.nan]]
        
        #dprint("SOM okay")
        if not(len(ans_bytes) == c.N_BYTES_TX_BUFFER):
            dprint("'%s' I/O ERROR: Expected %i bytes but received %i" %
                   (self.name, c.N_BYTES_TX_BUFFER, len(ans_bytes)))
            return [False, [np.nan], [np.nan], [np.nan], [np.nan]]

        ans_bytes = ans_bytes[c.N_BYTES_SOM : -c.N_BYTES_EOM] # Remove sentinels
        bytes_counter   = ans_bytes[0:4]    # Header
        bytes_millis    = ans_bytes[4:8]    # Header
        bytes_micros    = ans_bytes[8:10]   # Header
        bytes_idx_phase = ans_bytes[10:12]  # Header
        bytes_sig_I     = ans_bytes[12:]    # Body
        
        try:
            counter = struct.unpack('<' + c.binary_type_counter, bytes_counter)
            millis  = struct.unpack('<' + c.binary_type_millis , bytes_millis)
            micros  = struct.unpack('<' + c.binary_type_micros , bytes_micros)
            idx_phase = struct.unpack('<' + c.binary_type_idx_phase,
                                      bytes_idx_phase)
            
            sig_I = np.array(struct.unpack('<' +
                             c.binary_type_sig_I * c.BLOCK_SIZE, bytes_sig_I),
                             dtype=c.return_type_sig_I)
        except:
            dprint("'%s' I/O ERROR: Can't unpack bytes" % self.name)
            return [False, [np.nan], [np.nan], [np.nan], [np.nan]]
        
        counter   = counter[0]
        millis    = millis[0]
        micros    = micros[0]
        idx_phase = idx_phase[0]
        
        #dprint("%i %i" % (millis, micros))
        t0 = millis * 1000 + micros
        time = np.arange(0, c.BLOCK_SIZE)
        time = t0 + time * c.SAMPLING_PERIOD * 1e6
        time = np.asarray(time, dtype=c.return_type_time, order='C')
        
        idxs_phase = np.arange(idx_phase, idx_phase + c.N_LUT) % c.N_LUT
        phi = 2 * np.pi * idxs_phase / c.N_LUT
        
        # DEBUG test: Add artificial phase delay between ref_X/Y and sig_I
        """
        if 0:
            phase_delay_deg = 50
            phi = np.unwrap(phi + phase_delay_deg / 180 * np.pi)
        """
        
        sig_I = sig_I * c.A_REF / (2**c.ADC_INPUT_BITS - 1)
        
        if c.ref_waveform == Waveform.Cosine:
            lut_X = 0.5 * (1 + np.cos(phi))
            lut_Y = 0.5 * (1 + np.sin(phi))
        
        elif c.ref_waveform == Waveform.Square:
            lut_X = round_C_style(((1.75 * c.N_LUT - idxs_phase) % c.N_LUT) /
                                  (c.N_LUT - 1))
            lut_Y = round_C_style(((1.50 * c.N_LUT - idxs_phase) % c.N_LUT) /
                                  (c.N_LUT - 1))
            
        elif c.ref_waveform == Waveform.Sawtooth:
            lut_X = 1 - idxs_phase / (c.N_LUT - 1)
            lut_Y = 1 - ((idxs_phase - c.N_LUT / 4) % c.N_LUT) / (c.N_LUT - 1)

        elif c.ref_waveform == Waveform.Triangle:
            lut_X = 2 * np.abs(idxs_phase / c.N_LUT - 0.5)
            lut_Y = 2 * np.abs(((idxs_phase - c.N_LUT / 4) % c.N_LUT) / c.N_LUT - 0.5)
            
        elif c.ref_waveform == Waveform.Unknown:
            lut_X = np.full(np.nan, c.N_LUT)
            lut_Y = np.full(np.nan, c.N_LUT)

        lut_X = (c.ref_V_offset - c.ref_V_ampl) + 2 * c.ref_V_ampl * lut_X
        lut_Y = (c.ref_V_offset - c.ref_V_ampl) + 2 * c.ref_V_ampl * lut_Y
        lut_X.clip(0, c.A_REF)
        lut_Y.clip(0, c.A_REF)
        
        ref_X_tiled = np.tile(lut_X, np.int(np.ceil(c.BLOCK_SIZE / c.N_LUT)))
        ref_Y_tiled = np.tile(lut_Y, np.int(np.ceil(c.BLOCK_SIZE / c.N_LUT)))
        
        ref_X = np.asarray(ref_X_tiled[:c.BLOCK_SIZE],
                           dtype=c.return_type_ref_XY, order='C')
        
        ref_Y = np.asarray(ref_Y_tiled[:c.BLOCK_SIZE],
                           dtype=c.return_type_ref_XY, order='C')        
            
        return [True, time, ref_X, ref_Y, sig_I]
