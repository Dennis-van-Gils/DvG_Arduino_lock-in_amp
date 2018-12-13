#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module to communicate with an Arduino lock-in amp device over a serial
connection.

Dennis van Gils
09-12-2018
"""

import sys
import serial

import DvG_dev_Arduino__fun_serial as Arduino_functions
from DvG_debug_functions import print_fancy_traceback as pft

SOM = bytes([0x00, 0x00, 0x00, 0x00, 0xee]) # Start of message
EOM = bytes([0x00, 0x00, 0x00, 0x00, 0xff]) # End of message

class Arduino_lockin_amp(Arduino_functions.Arduino):
    def __init__(self, 
                 name="Lockin",
                 baudrate=3e5,
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
        
        self.ISR_CLOCK = 0          # [s]
        self.BUFFER_SIZE = 0        # [number of samples]
        self.ref_freq = 0           # [Hz]
        self.lockin_paused = True

    def begin(self, ref_freq=None):
        """
        Returns:
            success
        """
        [success, foo, bar] = self.turn_off()
        if success:
            if self.get_ISR_CLOCK():
                if self.get_BUFFER_SIZE():
                    if ref_freq == None:
                        if self.get_ref_freq():
                            return True
                    else:
                        if self.set_ref_freq(ref_freq):
                            return True
        
        return False
    
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
    
    def get_ISR_CLOCK(self):
        """
        Returns:
            success
        """
        [success, ans_str] = self.safe_query("ISR_CLOCK?")
        if success:
            self.ISR_CLOCK = int(ans_str)/1e6   # transform [us] to [s]
        return success
    
    def get_BUFFER_SIZE(self):
        """
        Returns:
            success
        """
        [success, ans_str] = self.safe_query("BUFFER_SIZE?")
        if success: 
            self.BUFFER_SIZE = int(ans_str)
        return success
    
    def get_ref_freq(self):
        """
        Returns:
            success
        """
        [success, ans_str] = self.safe_query("ref?")
        if success:
            self.ref_freq = float(ans_str)
        return success
    
    def set_ref_freq(self, freq):
        """
        Returns:
            success
        """
        [success, ans_str] = self.safe_query("ref %f" % freq)
        if success:
            self.ref_freq = float(ans_str)        
        return success
    
    def listen_to_lockin_amp(self):
        """Reads incoming data packets coming from the lock-in amp. This method
        is blocking until it receives an 'end-of-message' sentinel or until it
        times out.
        
        Returns:
            success
            ans_bytes
        """
        ans_bytes = self.ser.read_until(EOM)
        if (ans_bytes[:5] == SOM):
            ans_bytes = ans_bytes[5:-5] # Remove EOM & SOM
            return [True, ans_bytes]
        
        return [False, b'']