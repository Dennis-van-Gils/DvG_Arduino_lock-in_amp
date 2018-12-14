#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PyQt5 module to provide multithreaded communication and periodical data
acquisition for an Arduino based lock-in amplifier.
"""
__author__      = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__         = "https://github.com/Dennis-van-Gils/DvG_dev_Arduino"
__date__        = "13-12-2018"
__version__     = "1.2.0"

from PyQt5 import QtCore, QtWidgets as QtWid

import DvG_dev_Arduino_lockin_amp__fun_serial as lockin_functions
import DvG_dev_Base__pyqt_lib as Dev_Base_pyqt_lib

# Show debug info in terminal? Warning: Slow! Do not leave on unintentionally.
DEBUG_worker_DAQ  = False
DEBUG_worker_send = False

# ------------------------------------------------------------------------------
#   Arduino_pyqt
# ------------------------------------------------------------------------------

class Arduino_lockin_amp_pyqt(Dev_Base_pyqt_lib.Dev_Base_pyqt, QtCore.QObject):
    """Manages multithreaded communication and periodical data acquisition for
    an Arduino(-like) device.

    All device I/O operations will be offloaded to 'workers', each running in 
    a newly created thread instead of in the main/GUI thread.
        
        - Worker_DAQ:
            Periodically acquires data from the device.

        - Worker_send:
            Maintains a thread-safe queue where desired device I/O operations
            can be put onto, and sends the queued operations first in first out
            (FIFO) to the device.

    (*): See 'DvG_dev_Base__pyqt_lib.py' for details.

    Args:
        dev:
            Reference to a 'DvG_dev_Arduino__fun_serial.Arduino' instance.
        
        (*) DAQ_update_interval_ms
        (*) DAQ_function_to_run_each_update
        (*) DAQ_critical_not_alive_count
        (*) DAQ_timer_type
        
    Main methods:
        (*) start_thread_worker_DAQ(...)
        (*) start_thread_worker_send(...)
        (*) close_all_threads()
        
        queued_write(...):
            Write a message to the Arduino via the worker_send queue.
        
    Inner-class instances:
        (*) worker_DAQ
        (*) worker_send
        
    Main data attributes:
        (*) DAQ_update_counter
        (*) obtained_DAQ_update_interval_ms
        (*) obtained_DAQ_rate_Hz

    Signals:
        (*) signal_DAQ_updated()
        (*) signal_connection_lost()
    """
    signal_ref_freq_is_set = QtCore.pyqtSignal()
    
    def __init__(self,
                 dev: lockin_functions.Arduino_lockin_amp,
                 DAQ_update_interval_ms=1000,
                 DAQ_function_to_run_each_update=None,
                 DAQ_critical_not_alive_count=3,
                 DAQ_timer_type=QtCore.Qt.PreciseTimer,
                 DAQ_trigger_by=Dev_Base_pyqt_lib.DAQ_trigger.CONTINUOUS,
                 calc_DAQ_rate_every_N_iter=25,
                 parent=None):
        super(Arduino_lockin_amp_pyqt, self).__init__(parent=parent)

        self.attach_device(dev)

        self.create_worker_DAQ(DAQ_update_interval_ms,
                               DAQ_function_to_run_each_update,
                               DAQ_critical_not_alive_count,
                               DAQ_timer_type,
                               DAQ_trigger_by,
                               calc_DAQ_rate_every_N_iter=calc_DAQ_rate_every_N_iter,
                               DEBUG=DEBUG_worker_DAQ)

        self.create_worker_send(alt_process_jobs_function=
                                self.alt_process_jobs_function,
                                DEBUG=DEBUG_worker_send)
        
    def turn_on(self):
        """
        Returns:
            success
        """
        locker = QtCore.QMutexLocker(self.dev.mutex)
        
        if self.dev.turn_on():
            self.worker_DAQ.schedule_suspend(False)
            QtWid.QApplication.processEvents()
            return True        
        
        locker.unlock()
        return False
    
    def turn_off(self):
        """
        Returns:
            success
        """
        self.worker_DAQ.schedule_suspend()
        while not self.worker_DAQ.suspended:
            QtWid.QApplication.processEvents()
        
        locker = QtCore.QMutexLocker(self.dev.mutex)
        [success, foo, bar] = self.dev.turn_off()
        locker.unlock()

        return success
    
    def set_ref_freq(self, ref_freq):
        self.worker_send.queued_instruction("set_ref_freq", ref_freq)
    
    # --------------------------------------------------------------------------
    #   alt_process_jobs_function
    # --------------------------------------------------------------------------

    def alt_process_jobs_function(self, func, args):
        if func == "set_ref_freq":
            ref_freq = args[0]
            
            if not (ref_freq == self.dev.config.ref_freq):
                was_paused = self.dev.lockin_paused
                
                if not was_paused:
                    self.worker_DAQ.schedule_suspend()
                    while not self.worker_DAQ.suspended:
                        QtWid.QApplication.processEvents()
                
                self.dev.set_ref_freq(ref_freq)
                self.signal_ref_freq_is_set.emit()
    
                if not was_paused:
                    self.worker_DAQ.schedule_suspend(False)
                    QtWid.QApplication.processEvents()

        else:
            # Default job handling
            func(*args)