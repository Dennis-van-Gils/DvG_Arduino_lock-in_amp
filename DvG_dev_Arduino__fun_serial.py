#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module to communicate with an Arduino(-like) device over a serial connection.
* Provides automatic scanning over all serial ports for the Arduino.
* Mimicks the [PyVISA](https://pypi.org/project/PyVISA/) library  by providing
  ``query`` and ``query_ascii_values`` methods, which write a message to the
  Arduino and return back its reply.

The Arduino should be programmed to respond to a so-called 'identity' query over
the serial connection. It must reply to ``query('id?')`` with an ASCII string
response. Choosing a unique identity response per each Arduino in your project
allows for auto-connecting to these Arduinos without specifying the serial port.

Only ASCII based communication is supported so-far. Binary encoded communication
will be possible as well after a few modifications to this library have been
made (work in progress).

#### On the Arduino side
I also provide a C++ library for the Arduino(-like) device. It provides
listening to a serial port for commands and act upon them. This library can be
used in conjunction (but not required) with this Python library.
See [DvG_SerialCommand](https://github.com/Dennis-van-Gils/DvG_SerialCommand).

Classes:
    Arduino(...):
        Manages serial communication with an Arduino(-like) device.
    
        Methods:
            close():
                Close the serial connection.
            connect_at_port(...)
                Try to establish a connection on this serial port.
            scan_ports(...)
                Scan over all serial ports and try to establish a connection.
            auto_connect(...)
                Try the last used serial port, or scan over all when it fails.
            write(...)
                Write a string to the serial port.
            query(...)
                Write a string to the serial port and return the reply.
            query_ascii_values(...)
                Write a string to the serial port and return the reply, parsed
                into a list of floats.
        
        Important member:
            ser: serial.Serial instance belonging to the Arduino
"""
__author__      = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__         = "https://github.com/Dennis-van-Gils/DvG_dev_Arduino"
__date__        = "23-08-2018"
__version__     = "1.0.0"

import sys
import serial
import serial.tools.list_ports
from pathlib import Path

from DvG_debug_functions import print_fancy_traceback as pft

class Arduino():
    def __init__(self, name="Ard", baudrate=9600,
                 read_timeout=1, write_timeout=1,
                 read_term_char='\n', write_term_char='\n'):
        # Reference to the serial.Serial device instance when a connection has
        # been established
        self.ser = None
        
        # Given name for display and debugging purposes
        self.name = name
        
        # Response of self.query('id?') received from the Arduino.
        # Note that the Arduino should be programmed to respond to such a
        # query if you want the following functionality:
        # By giving each Arduino in your project an unique identity response,
        # one can scan over all serial ports to automatically connect to the
        # Arduino with the proper identity response, by either calling
        #   self.scan_ports(match_identity='your identity here')
        # or
        #   self.auto_connect(path_config, match_identity='your identity here')
        self.identity = None
        
        # Serial communication settings
        self.baudrate = baudrate
        self.read_timeout  = read_timeout
        self.write_timeout = write_timeout
        self.read_term_char  = read_term_char
        self.write_term_char = write_term_char

        # Is the connection to the device alive?
        self.is_alive = False
        
        # Placeholder for keeping track of future automated data acquisition as
        # used by e.g. DvG_dev_Arduino__pyqt_lib.py
        self.update_counter = 0

        # Placeholder for a future mutex instance needed for proper
        # multithreading (e.g. instance of QtCore.Qmutex())
        self.mutex = None

    # --------------------------------------------------------------------------
    #   close
    # --------------------------------------------------------------------------

    def close(self):
        """Close the serial port, disregarding any exceptions
        """
        # Prevent Windows, thinking to be smart, from keeping the port open in
        # case the connection got lost
        try: self.ser.cancel_read()
        except: pass
        try: self.ser.cancel_write()
        except: pass
    
        try: self.ser.close()
        except: pass
        
        self.is_alive = False

    # --------------------------------------------------------------------------
    #   connect_at_port
    # --------------------------------------------------------------------------

    def connect_at_port(self, port_str, match_identity=None,
                        print_trying_message=True):
        """Open the port at address 'port_str' and try to establish a
        connection. Subsequently, an identity query is send to the device. 
        Optionally, if a 'match_identity' string is passed, only connections to
        devices with a matching identity response are accepted.

        Args:
            port_str (str):
                Serial port address to open
            match_identity (str, optional):
                Identity string of the Arduino to establish a connection to.
                When empty or None any device is accepted. Defaults to None.
            print_trying_message (bool, optional):
                When True then a 'trying to open' message is printed to the
                terminal. Defaults to True.

        Returns: True if successful, False otherwise.
        """
        self.is_alive = False

        if match_identity == '': match_identity = None
        if print_trying_message:
            if match_identity is None:
                print("Connect to Arduino")
            else:
                print("Connect to Arduino with identity '%s'" %
                      match_identity)

        print("  @ %-5s: " % port_str, end='')
        try:
            # Open the serial port
            self.ser = serial.Serial(port=port_str,
                                     baudrate=self.baudrate,
                                     timeout=self.read_timeout,
                                     write_timeout=self.write_timeout)
        except (serial.SerialException):
            print("Could not open port")
            return False
        except:
            raise

        try:
            # Query the identity string.
            self.is_alive = True
            [success, identity_str] = self.query("id?", timeout_warning_style=2)
        except:
            print("Identity query 'id?' failed")
            if self.ser is not None: self.ser.close()
            self.is_alive = False
            return False

        if success:
            self.identity = identity_str
            print("ID '%s' : " % identity_str, end='')
            if match_identity is None:
                # Found any device
                print("Success!")
                print("  Name: '%s'\n" % self.name)
                self.is_alive = True
                return True
            elif identity_str.lower() == match_identity.lower():
                # Found the Arduino with matching identity
                print("Success!")
                print("  Name: '%s'\n" % self.name)
                self.is_alive = True
                return True
            else:
                print("Wrong identity")
                self.ser.close()
                self.is_alive = False
                return False

        print("Wrong or no device")
        if self.ser is not None: self.ser.close()
        self.is_alive = False
        return False

    # --------------------------------------------------------------------------
    #   scan_ports
    # --------------------------------------------------------------------------

    def scan_ports(self, match_identity=None):
        """Scan over all serial ports and try to establish a connection. A query
        for the identity string is send over all ports. The port that gives the
        proper response (and optionally has a matching identity string) must be
        the Arduino we're looking for.

        Args:
            match_identity (str, optional):
                Identity string of the Arduino to establish a connection to.
                When empty or None any Arduino is accepted. Defaults to None.

        Returns: True if successful, False otherwise.
        """
        if match_identity == '': match_identity = None
        if match_identity is None:
            print("Scanning ports for any Arduino")
        else:
            print(("Scanning ports for an Arduino with "
                   "identity '%s'") % match_identity)

        # Ports is a list of tuples
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            port_str = p[0]
            if self.connect_at_port(port_str, match_identity, False):
                return True
            else:
                continue

        # Scanned over all the ports without finding a match
        print("\n  ERROR: Device not found")
        return False

    # --------------------------------------------------------------------------
    #   auto_connect
    # --------------------------------------------------------------------------

    def auto_connect(self, path_config, match_identity=None):
        """
        """
        # Try to open the config file containing the port to open. Do not panic
        # if the file does not exist or cannot be read. We will then scan over
        # all ports as alternative.
        port_str = read_port_config_file(path_config)

        # If the config file was read successfully then we can try to open the
        # port listed in the config file and connect to the device.
        if port_str is not None:
            success = self.connect_at_port(port_str, match_identity)
        else:
            success = False

        # Check if we failed establishing a connection
        if not success:
            # Now scan over all ports and try to connect to the device
            print('') # Terminal esthetics
            success = self.scan_ports(match_identity)
            if success:
                # Store the result of a successful connection after a port scan
                # in the config file. Do not panic if we cannot create the
                # config file.
                write_port_config_file(path_config, self.ser.portstr)

        return success

    # --------------------------------------------------------------------------
    #   write
    # --------------------------------------------------------------------------

    def write(self, msg_str, timeout_warning_style=1):
        """Send a message to the serial device.

        Args:
            msg_str (str):
                String to be sent to the serial device.
            timeout_warning_style (int, optional):
                1 (default): Will print a traceback error message on screen and
                continue.
                2: Will raise the exception again.

        Returns: True if successful, False otherwise.
        """
        success = False

        if not self.is_alive:
            pft("Device is not connected yet or already closed.", 3)
        else:
            try:
                self.ser.write((msg_str + self.write_term_char).encode())
            except (serial.SerialTimeoutException,
                    serial.SerialException) as err:
                if timeout_warning_style == 1:
                    pft(err, 3)
                elif timeout_warning_style == 2:
                    raise(err)
            except Exception as err:
                pft(err, 3)
                sys.exit(1)
            else:
                success = True
        
        return success

    # --------------------------------------------------------------------------
    #   query
    # --------------------------------------------------------------------------

    def query(self, msg_str, timeout_warning_style=1):
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

    def query_ascii_values(self, msg_str="", separator='\t'):
        """Send a message to the serial device and subsequently read the reply.
        Expects a reply from the Arduino in the form of an ASCII string
        containing a list of numeric values. These values will be parsed into a
        list of floats and returned.
        
        Returns:
            success (bool):
                True if successful, False otherwise.
            ans_floats (list):
                Reply received from the device and parsed into a list of floats.
                [None] if unsuccessful.
        """
        [success, ans_str] = self.query(msg_str)
        
        if success and not(ans_str == ''):
            try:
                ans_floats = list(map(float, ans_str.split(separator)))
            except ValueError as err:
                # Print error and struggle on
                pft(err, 3)
            except Exception as err:
                pft(err, 3)
                sys.exit(1)
            else:
                return [True, ans_floats]
            
        return [False, []]
    
# ------------------------------------------------------------------------------
#   read_port_config_file
# ------------------------------------------------------------------------------

def read_port_config_file(filepath):
    """Try to open the config textfile containing the port to open. Do not panic
    if the file does not exist or cannot be read.

    Args:
        filepath (pathlib.Path):
            Path to the config file, e.g. Path("config/port.txt")

    Returns:
        The port name string when the config file is read out successfully,
        None otherwise.
    """
    if isinstance(filepath, Path):
        if filepath.is_file():
            try:
                with filepath.open() as f:
                    port_str = f.readline().strip()
                return port_str
            except:
                pass    # Do not panic and remain silent

    return None

# ------------------------------------------------------------------------------
#   write_port_config_file
# ------------------------------------------------------------------------------

def write_port_config_file(filepath, port_str):
    """Try to write the port name string to the config textfile. Do not panic if
    the file cannot be created.

    Args:
        filepath (pathlib.Path):
            Path to the config file, e.g. Path("config/port.txt")
        port_str (string):
            Serial port string to save to file.
            
    Returns: True when successful, False otherwise.
    """
    if isinstance(filepath, Path):
        if not filepath.parent.is_dir():
            # Subfolder does not exists yet. Create.
            try:
                filepath.parent.mkdir()
            except:
                pass    # Do not panic and remain silent

        try:
            # Write the config file
            filepath.write_text(port_str)
        except:
            pass        # Do not panic and remain silent
        else:
            return True

    return False

# ------------------------------------------------------------------------------
#   Main: Will show a demo when run from the terminal
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    ard = Arduino(name="Ard", baudrate=9600)
    
    ard.auto_connect(path_config=Path("last_used_port.txt"),
                     match_identity="My Arduino")
    #ard.scan_ports(match_identity="My Arduino")
    #ard.scan_ports()
    
    if not(ard.is_alive):
        sys.exit(1)
    
    print(ard.query("?"))
    print(ard.query_ascii_values("?", '\t'))
    
    ard.close()
