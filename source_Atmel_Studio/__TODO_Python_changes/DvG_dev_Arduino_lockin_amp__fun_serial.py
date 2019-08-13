#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module to communicate with an Arduino lock-in amp device over a serial
connection.
"""
__author__      = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__         = "https://github.com/Dennis-van-Gils/DvG_Arduino_lock-in_amp"
__date__        = "28-05-2019"
__version__     = "1.0.0"

import sys
import struct

import serial
import numpy as np

import DvG_dev_Arduino__fun_serial as Arduino_functions
from DvG_debug_functions import dprint, print_fancy_traceback as pft

class Arduino_lockin_amp(Arduino_functions.Arduino):
    class Config():
        # Serial communication sentinels: start and end of message
        SOM = b'\x00\x80\x00\x80\x00\x80\x00\x80\x00\x80'
        EOM = b'\xff\x7f\x00\x00\xff\x7f\x00\x00\xff\x7f'
        N_BYTES_SOM = len(SOM)
        N_BYTES_EOM = len(EOM) 
        
        # Data types to decode from binary stream
        binary_type_millis    = 'I'   # uint32_t
        binary_type_micros    = 'H'   # uint16_t
        binary_type_idx_phase = 'H'   # uint16_t
        binary_type_N_LUT     = 'H'   # uint16_t
        binary_type_sig_I     = 'H'   # uint16_t
        
        # Return types
        #return_type_time  = np.int64   # Signed to allow for flexible arithmetic
        #return_type_ref_X = np.float64
        #return_type_ref_Y = np.float64
        return_type_sig_I = np.float64
        
        #ISR_CLOCK               = 0    # [s]
        SAMPLING_PERIOD         = 0    # [s]
        Fs                      = 0    # [Hz]
        F_Nyquist               = 0    # [Hz]
        BLOCK_SIZE              = 0    # [number of samples per variable]
        T_SPAN_BUFFER           = 0    # [s]
        N_BYTES_TRANSMIT_BUFFER = 0    # [data bytes]
        N_LUT                   = 0    # [number of samples over a full waveform period]
        ANALOG_WRITE_RESOLUTION = 0    # [bits]
        ANALOG_READ_RESOLUTION  = 0    # [bits]
        A_REF                   = 0    # [V] Analog voltage reference of Arduino
        ref_V_offset            = 0    # [V] Voltage offset of cosine reference signal
        ref_V_ampl              = 0    # [V] Voltage amplitude of cosine reference signal
        ref_freq                = 0    # [Hz] Frequency of cosine reference signal
    
    def __init__(self, 
                 name="Lockin",
                 baudrate=1e6,
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

    def begin(self, ref_freq=None, ref_V_offset=None, ref_V_ampl=None):
        """
        Returns:
            success
        """
        [success, __foo, __bar] = self.turn_off()
        if not success:
            return False
        
        if not self.get_config():
            return False
        
        if ref_freq != None:
            if not self.set_ref_freq(ref_freq):
                return False
        if ref_V_offset != None:
            if not self.set_ref_V_offset(ref_V_offset):
                return False
        if ref_V_ampl != None:
            if not self.set_ref_V_ampl(ref_V_ampl):
                return False
        
        return True
    
    def safe_query(self, msg_str, timeout_warning_style=1):
        was_paused = self.lockin_paused
        
        if not was_paused:
            self.turn_off()
        
        [success, ans_str] = self.query(msg_str, timeout_warning_style)
            
        if success and not was_paused:
            self.turn_on()
            
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
    
    def get_config(self):
        """
        Returns:
            success
        """
        [success, ans_str] = self.safe_query("config?")
        if success:
            try:
                ans_list = ans_str.split('\t')
                self.config.SAMPLING_PERIOD         = float(ans_list[0]) * 1e-6
                self.config.BLOCK_SIZE              = int(ans_list[1])
                self.config.N_BYTES_TRANSMIT_BUFFER = int(ans_list[2])
                self.config.ANALOG_WRITE_RESOLUTION = int(ans_list[3])
                self.config.ANALOG_READ_RESOLUTION  = int(ans_list[4] )
                self.config.A_REF                   = float(ans_list[5])
                self.config.N_LUT                   = int(ans_list[6])
                self.config.ref_V_offset            = float(ans_list[7])
                self.config.ref_V_ampl              = float(ans_list[8])
                self.config.ref_freq                = float(ans_list[9])
                
                self.config.Fs = 1.0/self.config.SAMPLING_PERIOD
                self.config.F_Nyquist = self.config.Fs/2
                self.config.T_SPAN_BUFFER = (self.config.BLOCK_SIZE *
                                             self.config.SAMPLING_PERIOD)
                return True
            except Exception as err:
                raise(err)
                return False
        
        return False
    
    def set_ref_freq(self, ref_freq):
        """
        Returns:
            success
        """
        [success, ans_str] = self.safe_query("ref_freq %f" % ref_freq)
        if success:
            self.config.ref_freq = float(ans_str)
        return success
    
    def set_ref_V_offset(self, ref_V_offset):
        """
        Returns:
            success
        """
        [success, ans_str] = self.safe_query("ref_V_offset %f" % ref_V_offset)
        if success:
            self.config.ref_V_offset = float(ans_str)        
        return success
    
    def set_ref_V_ampl(self, ref_V_ampl):
        """
        Returns:
            success
        """
        [success, ans_str] = self.safe_query("ref_V_ampl %f" % ref_V_ampl)
        if success:
            self.config.ref_V_ampl = float(ans_str)        
        return success
    
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
        empty = np.array([np.nan])
        c = self.config  # Shorthand alias
        
        ans_bytes = self.read_until_EOM()
        #dprint("EOM found with %i bytes and..." % len(ans_bytes))
        if not (ans_bytes[:c.N_BYTES_SOM] == c.SOM):
            dprint("'%s' I/O ERROR: No SOM found" % self.name)
            return [False, empty, empty, empty, empty]
        
        #dprint("SOM okay")
        if not(len(ans_bytes) == c.N_BYTES_TRANSMIT_BUFFER):
            dprint("'%s' I/O ERROR: Expected %i bytes, received %i" %
                   (self.name, c.N_BYTES_TRANSMIT_BUFFER, len(ans_bytes)))
            return [False, empty, empty, empty, empty]

        """    
        end_byte_time  = c.BLOCK_SIZE * struct.calcsize(c.binary_type_time)
        end_byte_ref_X = (end_byte_time + c.BLOCK_SIZE *
                          struct.calcsize(c.binary_type_ref_X))
        end_byte_sig_I = (end_byte_ref_X + c.BLOCK_SIZE *
                          struct.calcsize(c.binary_type_sig_I))
        ans_bytes   = ans_bytes[c.N_BYTES_SOM  : -c.N_BYTES_EOM]
        bytes_time  = ans_bytes[0              : end_byte_time]
        bytes_ref_X = ans_bytes[end_byte_time  : end_byte_ref_X]
        bytes_sig_I = ans_bytes[end_byte_ref_X : end_byte_sig_I]
        try:
            time        = np.array(struct.unpack('<' +
                            c.binary_type_time * c.BLOCK_SIZE, bytes_time),
                            dtype=c.return_type_time)
            ref_X_phase = np.array(struct.unpack('<' + 
                            c.binary_type_ref_X * c.BLOCK_SIZE, bytes_ref_X),
                            dtype=c.return_type_ref_X)
            sig_I       = np.array(struct.unpack('<' +
                            c.binary_type_sig_I * c.BLOCK_SIZE, bytes_sig_I),
                            dtype=c.return_type_sig_I)
        except:
            dprint("'%s' I/O ERROR: Can't unpack bytes" % self.name)
            return [False, empty, empty, empty, empty]
        """
        ans_bytes = ans_bytes[c.N_BYTES_SOM  : -c.N_BYTES_EOM]
        bytes_millis    = ans_bytes[0:4]
        bytes_micros    = ans_bytes[4:6]
        bytes_idx_phase = ans_bytes[6:8]
        bytes_N_LUT     = ans_bytes[8:10]
        bytes_sig_I     = ans_bytes[10:]
        
        try:
            millis    = struct.unpack('<' + c.binary_type_millis, bytes_millis)
            micros    = struct.unpack('<' + c.binary_type_micros, bytes_micros)
            idx_phase = struct.unpack('<' + c.binary_type_idx_phase, bytes_idx_phase)
            N_LUT     = struct.unpack('<' + c.binary_type_N_LUT, bytes_N_LUT)
            sig_I = np.array(struct.unpack('<' +
                             c.binary_type_sig_I * c.BLOCK_SIZE, bytes_sig_I),
                             dtype=c.return_type_sig_I)
        except:
            dprint("'%s' I/O ERROR: Can't unpack bytes" % self.name)
            return [False, empty, empty, empty, empty]
        
        millis    = millis[0]
        micros    = micros[0]
        idx_phase = idx_phase[0]
        N_LUT     = N_LUT[0]
        
        #dprint("%i %i" % (millis, micros))
        t0 = millis * 1000 + micros
        time = np.arange(0, c.BLOCK_SIZE)
        time = t0 + time * c.SAMPLING_PERIOD * 1e6
        time = np.asarray(time, dtype=np.int64)
        
        idxs_phase = np.arange(idx_phase, idx_phase + c.BLOCK_SIZE)
        phi = 2 * np.pi * idxs_phase / N_LUT
        
        # DEBUG test: Add artificial phase delay between ref_X/Y and sig_I
        if 0:
            phase_delay_deg = 50
            phi = np.unwrap(phi + phase_delay_deg / 180 * np.pi)
        
        ref_X = (c.ref_V_offset + c.ref_V_ampl * np.cos(phi)).clip(0,c.A_REF)
        ref_Y = (c.ref_V_offset + c.ref_V_ampl * np.sin(phi)).clip(0,c.A_REF)
        sig_I = sig_I / (2**c.ANALOG_READ_RESOLUTION - 1) * c.A_REF
        #sig_I = sig_I * 2  # Compensate for differential mode of Arduino
        
        return [True, time, ref_X, ref_Y, sig_I]
