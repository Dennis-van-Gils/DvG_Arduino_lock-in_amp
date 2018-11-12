#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module to communicate with an Arduino lock-in amp device over a serial
connection.
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
                 baudrate=1500000,
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
        
        self.query = self.lockin_amp_query
    
    def lockin_amp_query(self, msg_str, timeout_warning_style=1):
        """Send a message to the serial device and subsequently read the reply.

        Args:
            msg_str (str):
                Message to be sent to the serial device.
            timeout_warning_style (int, optional):
                Work-around for the Serial library not throwing an exception
                when read has timed out.
                1 (default): Will print a traceback error message on screen and
                continue.
                2: Will raise the exception again.

        Returns:
            success (bool):
                True if successful, False otherwise.
            ans_str (str):
                Reply received from the device. [None] if unsuccessful.
        """
        success = False
        ans_str = None

        # First ensure the lock-in amp will switch off if not already so.
        if self.write("off", timeout_warning_style):
            self.ser.flushOutput()
            
            # Check for acknowledgement reply        
            try:
                ans_bytes = self.ser.read_until("off\n".encode())
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
                        ans_str = ans_bytes.decode('utf8').strip()
                    except UnicodeDecodeError as err:
                        # Print error and struggle on
                        pft(err, 3)
                    except Exception as err:
                        pft(err, 3)
                        sys.exit(1)
                    else:
                        success = True

        if not(success):
            return [success, ans_str]
        
        # There could be a lock-in amp on the serial port as the device
        # responded with 'off' as expected of a lock-in amp. 
        # Now send the query contained in the message string.
        success = False  # Must reset success for the next stage to come
        
        if self.write(msg_str, timeout_warning_style):
            try:
                ans_bytes = self.ser.read_until(self.read_term_char.encode())
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
                        ans_str = ans_bytes.decode('utf8').strip()
                    except UnicodeDecodeError as err:
                        # Print error and struggle on
                        pft(err, 3)
                    except Exception as err:
                        pft(err, 3)
                        sys.exit(1)
                    else:
                        success = True

        return [success, ans_str]