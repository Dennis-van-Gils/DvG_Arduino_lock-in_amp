#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PyQt5 interface to handle logging data to file particularly well suited for
multithreaded programs.

- Github: https://github.com/Dennis-van-Gils/python-dvg-pyqt-filelogger
- PyPI: https://pypi.org/project/dvg-pyqt-filelogger

Installation::

    pip install dvg-pyqt-filelogger

``class FileLogger(write_header_function: Callable = None, write_data_function: Callable = None):``

    Methods:

        * ``set_write_header_function(...)``
        * ``set_data_header_function(...)``
        * ``record(state)``
        * ``start_recording()``
        * ``stop_recording()``
        * ``update(filepath, mode)``
        * ``write(data)``
        * ``np_savetxt(...)``
        * ``flush()``
        * ``close()``
        * ``is_recording()``
        * ``elapsed()``
        * ``pretty_elapsed()``

    Signals:

        * ``signal_recording_started(str)``
        * ``signal_recording_stopped(pathlib.Path)``
"""
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/python-dvg-pyqt-filelogger"
__date__ = "13-05-2021"
__version__ = "1.1.0"

from typing import AnyStr, Callable
from pathlib import Path
import datetime

from PyQt5 import QtCore
from PyQt5.QtCore import QDateTime
import numpy as np

from dvg_debug_functions import print_fancy_traceback as pft


class FileLogger(QtCore.QObject):
    """Handles logging data to a file particularly well suited for multithreaded
    programs where one thread is writing data to the log and the other thread
    (the main/GUI thread) requests starting and stopping of the log, e.g.,
    by the user pressing a button.

    The methods ``start_recording()``, ``stop_recording()`` and ``record(bool)``
    can be directly called from the main/GUI thread.

    In the logging thread you repeatedly need to call ``update()``. This method
    takes cares of the state machine behind ``FileLogger`` and will perform the
    appropiate action, such as creating a file on disk, creating the header or
    writing new data to the log.

    Args:
        write_header_function (``Callable``, optional):
            Reference to a function that contains your specific code to write a
            header to the log file. This will get called during ``update()``.

            Default: ``None``

        write_data_function (``Callable``, optional):
            Reference to a function that contains your specific code to write
            new data to the log file. This will get called during ``update()``.

            Default: ``None``

        Both of the above functions can contain calls to the following class
        members:

            * ``FileLogger.write()``
            * ``FileLogger.elapsed()``

    NOTE:
        This class lacks a mutex and is hence not threadsafe from the get-go.
        As long as ``update()`` is being called from inside another mutex, such
        as a data-acquisition mutex for instance, it is safe.

    NOTE:
        By design the code in this class will continue on when exceptions occur.
        They are reported to the command line.

    .. rubric:: Signals:

    ``signal_recording_started (str)``:
        Emitted whenever a new recording has started. Useful for, e.g.,
        updating text of a record button.

        Returns:
            The filepath as ``str`` of the newly created log file.

        Type:
            ``PyQt5.QtCore.pyqtSignal()``

    ``signal_recording_stopped (pathlib.Path)``:
        Emitted whenever the recording has stopped. Useful for, e.g., updating
        text of a record button.

        Returns:
            The filepath as ``pathlib.Path()`` of the newly created log file.
            You could use this to, e.g., automatically navigate to the log in
            the file explorer or ask the user for a 'save to' destination.

        Type:
            ``PyQt5.QtCore.pyqtSignal()``

    Example usage:

    .. code-block:: python

        from PyQt5 import QtWidgets as QtWid
        from dvg_pyqt_filelogger import FileLogger

        #  When using a PyQt5 GUI put this inside your ``MainWindow()`` definition:
        # ----

        self.qpbt_record = QtWid.QPushButton(
            text="Click to start recording to file",
            checkable=True
        )
        self.qpbt_record.clicked.connect(lambda state: log.record(state))

        #  Initialize FileLogger at __main__
        # ----

        window = MainWindow()

        log = FileLogger(
            write_header_function=write_header_to_log,
            write_data_function=write_data_to_log
        )
        log.signal_recording_started.connect(
            lambda filepath: window.qpbt_record.setText(
                "Recording to file: %s" % filepath
            )
        )
        log.signal_recording_stopped.connect(
            lambda: window.qpbt_record.setText(
                "Click to start recording to file"
            )
        )

        #  Define these functions in your main module:
        # ----

        def write_header_to_log():
            log.write("elapsed [s]\treading_1\n")

        def write_data_to_log():
            log.write("%.3f\t%.4f\n" % (log.elapsed(), state.reading_1))

        #  Lastly, put this inside your logging thread:
        # ----

        log.update()
    """

    signal_recording_started = QtCore.pyqtSignal(str)
    signal_recording_stopped = QtCore.pyqtSignal(Path)

    def __init__(
        self,
        write_header_function: Callable = None,
        write_data_function: Callable = None,
    ):
        super().__init__(parent=None)

        self._write_header_function = write_header_function
        self._write_data_function = write_data_function

        self._filepath = None  # Will be of type pathlib.Path()
        self._filehandle = None
        self._mode = "a"

        self._timer = QtCore.QElapsedTimer()
        self._start = False
        self._stop = False
        self._is_recording = False

    def __del__(self):
        if self._is_recording:
            self._filehandle.close()

    def set_write_header_function(self, write_header_function: Callable):
        self._write_header_function = write_header_function

    def set_write_data_function(self, write_data_function: Callable):
        self._write_data_function = write_data_function

    @QtCore.pyqtSlot(bool)
    def record(self, state: bool = True):
        """Can be called from any thread.
        """
        if state:
            self.start_recording()
        else:
            self.stop_recording()

    @QtCore.pyqtSlot()
    def start_recording(self):
        """Can be called from any thread.
        """
        self._start = True
        self._stop = False

    @QtCore.pyqtSlot()
    def stop_recording(self):
        """Can be called from any thread.
        """
        self._start = False
        self._stop = True

    def update(self, filepath: str = "", mode: str = "a"):
        """This method will have to get called repeatedly, presumably in the
        thread where logging is required, e.g., the data-generation thread.
        This method takes cares of the state machine behind ``FileLogger`` and
        will perform the appropriate action, such as creating a file on disk,
        creating the header or writing new data to the log.

        Args:
            filepath (``str``):
                Location of the log file in case it has to be created or opened
                for write access.

                Default: ``"{yyMMdd_HHmmss}.txt"`` denoting the current date and time.

            mode (``str``, optional):
                Mode in which the log file is to be opened, see ``open()`` for
                more details. Most common options:

                    * ``w``: Open for writing, truncating the file first.
                    * ``a``: Open for writing, appending to the end of the file
                      if it exists.

                Defaults: ``a``
        """
        if self._start:
            if filepath == "":
                filepath = (
                    QDateTime.currentDateTime().toString("yyMMdd_HHmmss")
                    + ".txt"
                )

            self._filepath = Path(filepath)
            self._mode = mode

            # Reset flags
            self._start = False
            self._stop = False

            if self._create_log():
                self.signal_recording_started.emit(filepath)
                self._is_recording = True
                if self._write_header_function is not None:
                    self._write_header_function()
                self._timer.start()

            else:
                self._is_recording = False

        if self._is_recording and self._stop:
            self.signal_recording_stopped.emit(self._filepath)
            self._timer.invalidate()
            self.close()

        if self._is_recording:
            if self._write_data_function is not None:
                self._write_data_function()

    def _create_log(self) -> bool:
        """Create/open the log file and keep the file handle open.

        Returns True if successful, False otherwise.
        """
        try:
            self._filehandle = open(self._filepath, self._mode)
        except Exception as err:  # pylint: disable=broad-except
            pft(err, 3)
            return False
        else:
            return True

    def write(self, data: AnyStr) -> bool:
        """Write binary or ASCII data to the currently opened log file.

        By design any exceptions occurring in this method will not terminate the
        execution, but it will report the error to the command line and continue
        on instead.

        Returns True if successful, False otherwise.
        """
        try:
            self._filehandle.write(data)
        except Exception as err:  # pylint: disable=broad-except
            pft(err, 3)
            return False
        else:
            return True

    def np_savetxt(self, *args, **kwargs) -> bool:
        """Write 1D or 2D array_like data to the currently opened log file. This
        method passes all arguments directly to ``numpy.savetxt()``, see
        https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html.
        This method outperforms ``FileLogger.write()``, especially when large
        chunks of 2D data are passed (my test shows 8x faster).

        By design any exceptions occurring in this method will not terminate the
        execution, but it will report the error to the command line and continue
        on instead.

        Returns True if successful, False otherwise.
        """
        try:
            np.savetxt(self._filehandle, *args, **kwargs)
        except Exception as err:  # pylint: disable=broad-except
            pft(err, 3)
            return False
        else:
            return True

    @QtCore.pyqtSlot()
    def flush(self):
        """Force-flush the contents in the OS buffer to file as soon as
        possible. Do not call repeatedly, because it causes overhead.
        """
        self._filehandle.flush()

    def close(self):
        if self._is_recording:
            self._filehandle.close()
        self._start = False
        self._stop = False
        self._is_recording = False

    def is_recording(self) -> bool:
        return self._is_recording

    def elapsed(self) -> float:
        """Returns time in seconds (``float``) since start of recording.
        """
        return self._timer.elapsed() / 1e3

    def pretty_elapsed(self) -> str:
        """Returns time as "h:mm:ss" (``str``) since start of recording.
        """
        return str(datetime.timedelta(seconds=int(self.elapsed())))
