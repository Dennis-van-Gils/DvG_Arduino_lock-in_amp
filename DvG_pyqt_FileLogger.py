#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Class FileLogger handles logging data to file, particularly well suited for
multithreaded programs, where one thread is writing data to the log (the logging
thread) and the other thread (the main thread/GUI) handles starting and stopping
of the logging by user interaction (i.e. a button).
    
The boolean members 'starting' and 'stopping' should be directly written to from
the main/GUI thread. (These are boolean atomic operations so no mutex lock is
necessary here.)

In the logging thread one should test for these booleans as demonstrated in the
following example:
    if file_logger.starting:
        if file_logger.create_log(my_current_time, my_path):
            file_logger.write("Time\tValue\n")  # Header
        
    if file_logger.stopping:
        file_logger.close_log()
        
    if file_logger.is_recording:
        elapsed_time = my_current_time - file_logger.start_time
        file_logger.write("%.3f\t%.3f\n" % (elapsed_time, my_value))

Class:
    FileLogger():   
        Methods:
            create_log(...):
                Open new log file and keep file handle open.
            write(...):
                Write data to the open log file.
            close_log():
                Close the log file.
                
        Important members:
            starting (bool):
            stopping (bool):
            is_recording (bool):

        Signals:
            signal_set_recording_text(str):
                Useful for updating text of e.g. a record button when using a
                PyQt GUI. This signal is not emitted in this module itself, and
                you should emit it yourself in your own code when needed.
"""
__author__      = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__         = "https://github.com/Dennis-van-Gils/DvG_PyQt_misc"
__date__        = "23-08-2018"
__version__     = "1.0.0"

from pathlib import Path

from PyQt5 import QtCore

from DvG_debug_functions import print_fancy_traceback as pft

class FileLogger(QtCore.QObject):
    signal_set_recording_text = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__(None)

        self.path_log = None        # pathlib.Path instance to the log
        self.f_log = None           # File handle to log
        
        self.start_time = None      # To keep track of elapsed time since start
        self.starting = False
        self.stopping = False
        self.is_recording = False
        
        # Placeholder for a future mutex instance needed for proper
        # multithreading (e.g. instance of QtCore.Qmutex())
        self.mutex = None
        
    def create_log(self, start_time, path_log: Path, mode='a'):
        """Open new log file and keep file handle open.
        
        Args:
            start_time:
                Timestamp of the start of recording, usefull to keep track of
                the elapsed time while recording.
            path_log [pathlib.Path]:
                Location of the file to write to.
            mode:
                Mode in which the file is openend, see 'open()' for more
                details. Defaults to 'a'. Most common options:
                'w': Open for writing, truncating the file first
                'a': Open for writing, appending to the end of the file if it
                     exists
        
        Returns: True if successful, False otherwise.
        """
        self.path_log = path_log
        self.start_time = start_time
        self.starting = False
        self.stopping = False
        
        try:
            self.f_log = open(path_log, mode)
        except Exception as err:
            pft(err, 3)
            self.is_recording = False
            return False
        else:            
            self.is_recording = True
            return True
    
    def write(self, data):
        """
        Returns: True if successful, False otherwise.
        """
        try:
            self.f_log.write(data)
        except Exception as err:
            pft(err, 3)
            return False
        else:
            return True
        
    def close_log(self):
        if self.is_recording:
            self.f_log.close()
        self.starting = False
        self.stopping = False
        self.is_recording = False
