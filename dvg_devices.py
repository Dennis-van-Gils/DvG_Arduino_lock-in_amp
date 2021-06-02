#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/python-dvg-devices"
__date__ = "15-07-2020"
__version__ = "0.2.1"

import sys
import time
from typing import AnyStr, Callable
from pathlib import Path

# Use of `ast.literal_eval` got removed in v0.2.2 because it chokes on `nan`
# from ast import literal_eval

import serial
import serial.tools.list_ports

from dvg_debug_functions import dprint, ANSI, print_fancy_traceback as pft


class SerialDevice:
    """This class provides higher-level general I/O methods for a serial device,
    by wrapping the excellent `pySerial
    <https://pyserial.readthedocs.io/en/latest/pyserial.html>`_ library.

    The following functionality is offered:

    * TODO: mention write(), query(), query_ascii_values() and close().

    * Scanning over all serial ports to autoconnect to the desired serial
      device, based on the device's reply to a validation query.

    * Autoconnecting to the device, based on the last known successful port that
      gets written to a configuration textfile.

    Instances of this class will tie in nicely with
    :class:`dvg_qdeviceio.QDeviceIO`.

    Args:
        long_name (:obj:`str`, optional):
            Long display name of the device in a general sense. E.g.,
            `"Keysight N8700 PSU"` or `"Arduino M0 Pro"`.

            Default: `"Serial Device"`

        name (:obj:`str`, optional):
            Short display name for the device. E.g., `"PSU_1"` or `"blinker"`.

            Default: `"Dev_1"`

    .. rubric:: Attributes:

    Attributes:
        long_name (:obj:`str`):
            Long display name of the device in a general sense. E.g,
            `"Keysight N8700 PSU"` or `"Arduino M0 Pro"`.

        name (:obj:`str`):
            Short display name for the device. E.g., `"PSU_1"` or `"blinker"`.

        serial_settings (:obj:`dict`):
            Dictionary of keyword arguments to be passed directly to
            :class:`serial.Serial` at initialization of the serial port when
            trying to connect. Do not specify `port` in this dictionary as it
            will be supplied by this class's machinery.

            Default: `{"baudrate": 9600, "timeout": 2, "write_timeout": 2}`

        ser (:class:`serial.Serial` | :obj:`None`):
            Will be set to a :class:`serial.Serial` device instance when a
            connection has been established. Otherwise: :obj:`None`.

        is_alive (:obj:`bool`):
            Is the connection alive? I.e., Can we communicate?
    """

    def __init__(
        self, name="Dev_1", long_name="Serial Device",
    ):
        self.long_name = long_name
        self.name = name

        # Default serial settings
        self.serial_settings = {
            "baudrate": 9600,
            "timeout": 2,
            "write_timeout": 2,
        }

        # Termination characters, must always be of type `bytes`.
        self._read_termination = "\n".encode()
        self._write_termination = "\n".encode()

        # Wait time during :meth:`query` in case there is no read termination
        # character set. We have to wait long enough to make sure the device has
        # had ample time to send out the reply to the serial-input buffer of
        # the host PC, which we will then read completely in one go.
        self._query_wait_time = 0.1  # [s]

        # See :meth:`set_ID_validation_query`
        self._ID_validation_query = None
        self._valid_ID_broad = None
        self._valid_ID_specific = None
        # Whenever `connect_at_port()` is performing the user-supplied
        # ID_validation_query, any query inside of it /must/ raise an exception
        # on timeout. Instead of relying on the user to set the argument
        # `raises_on_timeout=True` when invoking `query()`, we will use below
        # flag instead to enforce the raise.
        self._force_query_to_raise_on_timeout = False

        self.ser = None
        self.is_alive = False

    # --------------------------------------------------------------------------
    #   set_read_termination
    # --------------------------------------------------------------------------

    def set_read_termination(
        self, termination: AnyStr, query_wait_time: float = 0.1,
    ):
        """Set the termination character(s) for serial read.

        Args:
            termination (:obj:`str` | :obj:`bytes` | :obj:`None`):
                Termination character(s). When set to :obj:`None` or empty the
                I/O operation :meth:`query` will wait ``query_wait_time``
                seconds before reading out the complete serial-input buffer,
                which by then should contain the device's reply to the query.

            query_wait_time (:obj:`float`, optional):
                See above.

                Default: :const:`0.1`
        """
        if termination is None:
            termination = b""

        self._read_termination = (
            termination.encode()
            if isinstance(termination, str)
            else termination
        )

        self._query_wait_time = query_wait_time

    # --------------------------------------------------------------------------
    #   set_write_termination
    # --------------------------------------------------------------------------

    def set_write_termination(self, termination: AnyStr):
        """Set the termination character(s) for serial write.

        Args:
            termination (:obj:`str` | :obj:`bytes` | :obj:`None`):
                Termination character(s).
        """
        if termination is None:
            termination = b""

        self._write_termination = (
            termination.encode()
            if isinstance(termination, str)
            else termination
        )

    # --------------------------------------------------------------------------
    #   set_ID_validation_query
    # --------------------------------------------------------------------------

    def set_ID_validation_query(
        self,
        ID_validation_query: Callable[[], tuple],
        valid_ID_broad: object,
        valid_ID_specific: object = None,
    ):
        """When this method **is not** called, then the following default scheme
        applies:

            During :meth:`connect_at_port`, :meth:`scan_ports` or
            :meth:`auto_connect` a succesful connection to *any* serial device
            will be accepted and be stored in class member :attr:`ser`.

        When this method **is** called, then the following scheme applies:

            During :meth:`connect_at_port`, :meth:`scan_ports` or
            :meth:`auto_connect` a connection to a *desired* device will be
            attempted by performing a query for a device ID over the serial
            connection. The serial connection will be accepted and be stored in
            class member :attr:`ser` whenever the following scheme returns
            succesful:

        Args:
            ID_validation_query (:obj:`~typing.Callable` [[], :obj:`tuple`]):
                Reference to a function to perform an ID validation query
                on the device when connecting. The function should take zero
                arguments and return a tuple consisting of two objects, as will
                be explained further down. Only when the outcome of the
                validation is successful, will the connection be accepted and
                remain open. Otherwise, the connection will be automatically
                closed again.

                The device's reply on the validation query will be tested
                against the two other arguments: ``valid_ID_broad`` (required)
                and ``valid_ID_specific`` (optional).

                The *broad* reply can be used to allow connections to a device
                of a certain manufacturer and model. E.g., in a response to an
                ``*idn?`` validation query, one can test a part of the full
                reply -- say, `"THURLBY THANDAR, QL355TP, 279730, 1.00 – 1.00"`
                -- against `"THURLBY THANDAR, QL"` to allow connections to *any*
                Thurlby Thandar QL-series power supply.

                The *specific* reply can be used to narrow down to a specific
                device, once it has passed the *broad* validation test. In the
                example above, one could test against the series number
                `"279730"`, for instance. When argument ``valid_ID_specific``
                is not supplied, any *broad* match will be accepted as
                connection.

                The function to be passed, being defined as a class method
                inside of a derived :class:`SerialDevice` class, should have a
                general form like:

                .. code-block:: python

                    def my_ID_validation_query(self) -> (object, object):
                        # Expected: reply = "THURLBY THANDAR, QL355TP, 279730, 1.00 – 1.00"
                        _success, reply = self.query("*idn?")
                        reply_broad = ans[:19]                  # "THURLBY THANDAR, QL"
                        reply_specific = ans.split(",")[2]      # "279730", i.e. serial number

                        return (reply_broad, reply_specific)

                When ``ID_validation_query`` is set to :obj:`None`, no
                validation will take place and *any* successful connection will
                be accepted and remain open.

            valid_ID_broad (:obj:`object`):
                Reply to be broadly matched when a function reference is being
                passed onto ``ID_validation_query``.

            valid_ID_specific (:obj:`object`, optional):
                Reply to be specifically matched when a function reference is
                being passed onto ``ID_validation_query``. Note: When set
                to :obj:`None`, any connection that is *broadly* matched will be
                accepted and remain open.

                Default: :obj:`None`
        """
        self._ID_validation_query = ID_validation_query
        self._valid_ID_broad = valid_ID_broad
        self._valid_ID_specific = valid_ID_specific

    # --------------------------------------------------------------------------
    #   write
    # --------------------------------------------------------------------------

    def write(self, msg: AnyStr, raises_on_timeout: bool = False) -> bool:
        """Send a message to the serial device.

        Args:
            msg (:obj:`str` | :obj:`bytes`):
                ASCII string or bytes to be sent to the serial device.

            raises_on_timeout (:obj:`bool`, optional):
                Should an exception be raised when a write timeout occurs?

                Default: :const:`False`

        Returns:
            True if successful, False otherwise.
        """
        if not self.is_alive:
            pft("Device is not connected yet or already closed.", 3)
            return False  # --> leaving

        if isinstance(msg, str):
            msg = msg.encode()

        try:
            self.ser.write(msg + self._write_termination)
        except (serial.SerialTimeoutException, serial.SerialException,) as err:
            if raises_on_timeout:
                raise err  # --> leaving
            else:
                pft(err, 3)
                return False  # --> leaving
        except Exception as err:
            pft(err, 3)
            sys.exit(0)  # --> leaving

        return True

    # --------------------------------------------------------------------------
    #   query
    # --------------------------------------------------------------------------

    def query(
        self,
        msg: AnyStr,
        raises_on_timeout: bool = False,
        returns_ascii: bool = True,
    ) -> tuple:
        """Send a message to the serial device and subsequently read the reply.

        Args:
            msg (:obj:`str` | :obj:`bytes`):
                ASCII string or bytes to be sent to the serial device.

            raises_on_timeout (:obj:`bool`, optional):
                Should an exception be raised when a write or read timeout
                occurs?

                Default: :const:`False`

            returns_ascii (:obj:`bool`, optional):
                When set to :const:`True` the device's reply will be returned as
                an ASCII string. Otherwise, it will return as bytes.

                Default: :const:`True`

        Returns:
            :obj:`tuple`:
                - success (:obj:`bool`):
                    True if successful, False otherwise.

                - reply (:obj:`str` | :obj:`bytes` | :obj:`None`):
                    Reply received from the device, either as ASCII string
                    (default) or as bytes when ``returns_ascii`` was set to
                    :const:`False`. :obj:`None` if unsuccessful.
        """

        # Always ensure that a timeout exception is raised when coming from
        # :meth:`connect_at_port`.
        if self._force_query_to_raise_on_timeout:
            raises_on_timeout = True

        # Send query
        if not self.write(msg, raises_on_timeout=raises_on_timeout):
            return (False, None)  # --> leaving

        # Read reply
        try:
            if self._read_termination == b"":
                self.ser.flush()
                time.sleep(self._query_wait_time)
                reply = self.ser.read(self.ser.inWaiting())
            else:
                reply = self.ser.read_until(self._read_termination)
        except serial.SerialException as err:
            # Note: The Serial library does not throw an exception when it
            # times out in `read`, only when it times out in `write`! We
            # will check for zero received bytes as indication for a read
            # timeout, later. See: https://stackoverflow.com/questions/10978224/serialtimeoutexception-in-python-not-working-as-expected
            pft(err, 3)
            return (False, None)  # --> leaving
        except Exception as err:
            pft(err, 3)
            sys.exit(0)  # --> leaving

        if len(reply) == 0:
            if raises_on_timeout:
                raise serial.SerialException(
                    "Received 0 bytes. Read probably timed out."
                )  # --> leaving
            else:
                pft("Received 0 bytes. Read probably timed out.", 3)
                return (False, None)  # --> leaving

        if returns_ascii:
            try:
                reply = reply.decode("utf8").strip()
            except UnicodeDecodeError as err:
                pft(err, 3)
                return (False, None)  # --> leaving
            except Exception as err:
                pft(err, 3)
                sys.exit(0)  # --> leaving

        return (True, reply)

    # --------------------------------------------------------------------------
    #   query_ascii_values
    # --------------------------------------------------------------------------

    def query_ascii_values(
        self, msg: str, delimiter="\t", raises_on_timeout: bool = False,
    ) -> tuple:
        r"""Send a message to the serial device and subsequently read the reply.
        Expects a reply in the form of an ASCII string containing a list of
        numeric values, separated by a delimiter. These values will be parsed
        into a list and returned.

        Args:
            msg (:obj:`str`):
                ASCII string to be sent to the serial device.

            delimiter (:obj:`str`, optional):
                Delimiter used in the device's reply.

                Default: `"\\t"`

            raises_on_timeout (:obj:`bool`, optional):
                Should an exception be raised when a write or read timeout
                occurs?

                Default: :const:`False`

        Returns:
            :obj:`tuple`:
                - success (:obj:`bool`):
                    True if successful, False otherwise.

                - reply_list (:obj:`list`):
                    Reply received from the device and parsed into a list of
                    separate values. The list is empty if unsuccessful.
        """
        success, reply = self.query(
            msg, raises_on_timeout=raises_on_timeout, returns_ascii=True
        )

        if not success:
            return (False, ())  # --> leaving

        try:
            # NOTE: `ast.literal_eval` chokes when it receives 'nan' so we ditch
            # it and just interpret everything as `float` instead.
            # reply_list = list(map(literal_eval, reply.split(delimiter)))
            reply_list = list(map(float, reply.split(delimiter)))
        except ValueError as err:
            pft(err, 3)
            return (False, ())  # --> leaving
        except Exception as err:
            pft(err, 3)
            sys.exit(0)  # --> leaving

        return (True, reply_list)

    # --------------------------------------------------------------------------
    #   close
    # --------------------------------------------------------------------------

    def close(self, ignore_exceptions=False):
        """Cancel all pending serial operations and close the serial port.
        """
        if self.ser is not None:
            try:
                self.ser.cancel_read()
            except:
                pass
            try:
                self.ser.cancel_write()
            except:
                pass

            try:
                self.ser.close()
            except Exception as err:
                if ignore_exceptions:
                    pass
                else:
                    pft(err, 3)
                    sys.exit(0)

        self.is_alive = False

    # --------------------------------------------------------------------------
    #   connect_at_port
    # --------------------------------------------------------------------------

    def connect_at_port(self, port: str, verbose: bool = True) -> bool:
        """Open the serial port at address ``port`` and try to establish a
        connection.

        If the connection is successful and :meth:`set_ID_validation_query`
        was not set, then this function will return :const:`True`.

        If :meth:`set_ID_validation_query` was set, then this function will
        return :const:`True` or :const:`False` depending on the validation
        scheme as explained in :meth:`set_ID_validation_query`.

        Args:
            port (:obj:`str`):
                Serial port address to open.

            verbose (:obj:`bool`, optional):
                Print a `"Connecting to: "`-message to the terminal?

                Default: :const:`True`

        Returns:
            True if successful, False otherwise.
        """

        def print_success(success_str: str):
            dprint(success_str, ANSI.GREEN)
            dprint(" " * 16 + "--> %s\n" % self.name, ANSI.GREEN)

        if verbose:
            _print_hrule(True)
            if (
                self._ID_validation_query is None
                or self._valid_ID_specific is None
            ):
                dprint("  Connecting to: %s" % self.long_name, ANSI.YELLOW)
            else:
                dprint(
                    "  Connecting to: %s `%s`"
                    % (self.long_name, self._valid_ID_specific),
                    ANSI.YELLOW,
                )
            _print_hrule()

        print("  @ %-11s " % port, end="")
        try:
            # Open the serial port
            self.ser = serial.Serial(port=port, **self.serial_settings)
        except serial.SerialException:
            print("Could not open port.")
            return False  # --> leaving
        except Exception as err:
            pft(err, 3)
            sys.exit(0)  # --> leaving

        if self._ID_validation_query is None:
            # Found any device
            print_success("Any Success!")
            self.is_alive = True
            return True  # --> leaving

        # Optional validation query
        try:
            self._force_query_to_raise_on_timeout = True
            self.is_alive = True  # We must assume communication is possible
            reply_broad, reply_specific = self._ID_validation_query()
        except:
            print("Wrong or no device.")
            self.close(ignore_exceptions=True)
            return False  # --> leaving
        finally:
            self._force_query_to_raise_on_timeout = False

        if reply_broad == self._valid_ID_broad:
            if reply_specific is not None:
                print("Found `%s`: " % reply_specific, end="")

            if self._valid_ID_specific is None:
                # Found a matching device in a broad sense
                print_success("Broad Success!")
                self.is_alive = True
                return True  # --> leaving

            elif reply_specific == self._valid_ID_specific:
                # Found a matching device in a specific sense
                print_success("Specific Success!")
                self.is_alive = True
                return True  # --> leaving

        print("Wrong device.")
        self.close(ignore_exceptions=True)
        return False

    # --------------------------------------------------------------------------
    #   scan_ports
    # --------------------------------------------------------------------------

    def scan_ports(self, verbose: bool = True) -> bool:
        """Scan over all serial ports and try to establish a connection. See
        further the description at :meth:`connect_at_port`.

        Args:
            verbose (:obj:`bool`, optional):
                Print a `"Scanning ports for: "`-message to the terminal?

                Default: :const:`True`

        Returns:
            True if successful, False otherwise.
        """

        if verbose:
            _print_hrule(True)
            if (
                self._ID_validation_query is None
                or self._valid_ID_specific is None
            ):
                dprint("  Scanning ports for: %s" % self.long_name, ANSI.YELLOW)
            else:
                dprint(
                    "  Scanning ports for: %s `%s`"
                    % (self.long_name, self._valid_ID_specific),
                    ANSI.YELLOW,
                )
            _print_hrule()

        # Ports is a list of tuples
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            port = p[0]
            if self.connect_at_port(port, verbose=False):
                return True
            else:
                continue

        # Scanned over all the ports without finding a match
        dprint("  Error: Device not found.\n", ANSI.RED)
        return False

    # --------------------------------------------------------------------------
    #   auto_connect
    # --------------------------------------------------------------------------

    def auto_connect(
        self, filepath_last_known_port: str = "config/port.txt"
    ) -> bool:
        """Try to connect to the device using the last-known successful port
        as got written to the textfile ``filepath_last_known_port`` by the
        previous call to :meth:`auto_connect`.

        When the file does not exist, can not be read or if the desired device
        can not be found at that specific port, then a scan over all ports will
        be performed by automatically invoking :meth:`scan_ports`.

        Args:
            filepath_last_known_port (:obj:`str`):
                Default: `"config/port.txt"`

        Returns:
            True if successful, False otherwise.
        """
        path = Path(filepath_last_known_port)
        port = self._get_last_known_port(path)

        if port is None:
            if self.scan_ports():
                self._store_last_known_port(path, self.ser.portstr)
                return True
            else:
                return False
        else:
            if self.connect_at_port(port):
                self._store_last_known_port(path, self.ser.portstr)
                return True
            else:
                if self.scan_ports(verbose=False):
                    self._store_last_known_port(path, self.ser.portstr)
                    return True
                else:
                    return False

    # -----------------------------------------------------------------------------
    #   _get_last_known_port
    # -----------------------------------------------------------------------------

    def _get_last_known_port(self, path: Path):
        """Try to open the textfile pointed to by ``path``, containing the port
        to open. Do not panic if the file does not exist or cannot be read.

        Args:
            path (:class:`pathlib.Path`):
                Path to the textfile, e.g., ``Path("config/port.txt")``.

        Returns:
            The port name :obj:`str` when the textfile is read out successfully,
            :obj:`None` otherwise.
        """
        if isinstance(path, Path):
            if path.is_file():
                try:
                    with path.open() as f:
                        port = f.readline().strip()
                    return port
                except:
                    pass  # Do not panic and remain silent

        return None

    # -----------------------------------------------------------------------------
    #   _store_last_known_port
    # -----------------------------------------------------------------------------

    def _store_last_known_port(self, path: Path, port_str):
        """Try to write the port name string ``port_str`` to the textfile
        pointed to by ``path``. Do not panic if the file can not be created or
        written to.

        Args:
            path (:class:`pathlib.Path`):
                Path to the textfile, e.g., ``Path("config/port.txt")``.

            port_str (:obj:`str`):
                The port name string to write to file.

        Returns:
            True if successful, False otherwise.
        """
        if isinstance(path, Path):
            if not path.parent.is_dir():
                # Subfolder does not exists yet. Create.
                try:
                    path.parent.mkdir()
                except:
                    pass  # Do not panic and remain silent

            try:
                # Write the config file
                path.write_text(port_str)
            except:
                pass  # Do not panic and remain silent
            else:
                return True

        return False


def _print_hrule(leading_newline=False):
    dprint(("\n" if leading_newline else "") + "-" * 60, ANSI.YELLOW)


class Arduino(SerialDevice):
    """Provides higher-level general I/O methods for communicating with an
    Arduino(-like) board over the serial connection.

    The Arduino could be programmed to respond to a so-called 'identity' query over
    the serial connection. It must reply to ASCII command "id?" with an ASCII string
    response. Choosing a unique identity response per each Arduino in your project
    allows for auto-connecting to your Arduino without having to specify the serial
    port.

    .. rubric:: On the Arduino side:

    I also provide a C++ library for the Arduino(-like) device. It provides
    listening to a serial port for commands and act upon them. This library can be
    used in conjunction (but not required) with this Python module.
    See https://github.com/Dennis-van-Gils/DvG_SerialCommand.

    .. rubric:: Most important methods inherited from :class:`~dvg_devices.BaseDevice.SerialDevice`:

    * :meth:`~dvg_devices.BaseDevice.SerialDevice.connect_at_port`
    * :meth:`~dvg_devices.BaseDevice.SerialDevice.scan_ports`
    * :meth:`~dvg_devices.BaseDevice.SerialDevice.auto_connect`
    * :meth:`~dvg_devices.BaseDevice.SerialDevice.close`
    * :meth:`~dvg_devices.BaseDevice.SerialDevice.write`
    * :meth:`~dvg_devices.BaseDevice.SerialDevice.query`
    * :meth:`~dvg_devices.BaseDevice.SerialDevice.query_ascii_values`

    See :class:`~dvg_devices.BaseDevice.SerialDevice` for the documentation of
    all the methods and attributes that are available. Instances of this class
    will tie in nicely with :class:`~dvg_qdeviceio.QDeviceIO`.
    """

    def __init__(
        self, name="Ard_1", long_name="Arduino", connect_to_specific_ID=None,
    ):
        super().__init__(name=name, long_name=long_name)

        # Default serial settings
        self.serial_settings = {
            "baudrate": 115200,
            "timeout": 2,
            "write_timeout": 2,
        }
        self.set_read_termination("\n")
        self.set_write_termination("\n")

        self.set_ID_validation_query(
            ID_validation_query=self.ID_validation_query,
            valid_ID_broad="Arduino",
            valid_ID_specific=connect_to_specific_ID,
        )

    # --------------------------------------------------------------------------
    #   ID_validation_query
    # --------------------------------------------------------------------------

    def ID_validation_query(self) -> (str, str):
        # Expected: reply = "Arduino, [specific ID]"
        _success, reply = self.query("id?")
        reply = reply.split(",")
        reply_broad = reply[0].strip()  # "Arduino"
        reply_specific = reply[1].strip() if len(reply) > 1 else None

        return (reply_broad, reply_specific)


# ------------------------------------------------------------------------------
#   Main: Will show a demo when run from the terminal
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    ard = Arduino(name="Ard_1", connect_to_specific_ID="Wave generator")

    ard.serial_settings["baudrate"] = 115200
    ard.auto_connect()
    # ard.scan_ports()

    if not ard.is_alive:
        sys.exit(0)

    _success, readings = ard.query_ascii_values("?")
    print(readings)

    _success, reply = ard.query("?")
    print(reply)

    ard.close()
