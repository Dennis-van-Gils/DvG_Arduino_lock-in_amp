#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PyQt5 module to provide multithreaded communication and periodical data
acquisition for an Arduino based lock-in amplifier.
"""
__author__      = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__         = "https://github.com/Dennis-van-Gils/DvG_dev_Arduino"
__date__        = "30-07-2019"
__version__     = "1.0.0"

import numpy as np
from scipy.signal import welch
from DvG_RingBuffer import DvG_RingBuffer as RingBuffer
from PyQt5 import QtCore, QtWidgets as QtWid
import time as Time

import DvG_dev_Base__pyqt_lib as Dev_Base_pyqt_lib
import DvG_dev_Arduino_lockin_amp__fun_serial as lockin_functions
from DvG_Buffered_FIR_Filter import Buffered_FIR_Filter

# WORK IN PROGRESS
"""
import pyfftw
# Monkey patch fftpack
np.fft = pyfftw.interfaces.numpy_fft
# Turn on the cache for optimum performance
pyfftw.interfaces.cache.enable()
"""

# ------------------------------------------------------------------------------
#   Arduino_pyqt
# ------------------------------------------------------------------------------

class Arduino_lockin_amp_pyqt(Dev_Base_pyqt_lib.Dev_Base_pyqt, QtCore.QObject):
    """Manages multithreaded communication and periodical data acquisition for
    an Arduino(-like) lock-in amplifier device.

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
    signal_ref_freq_is_set     = QtCore.pyqtSignal()
    signal_ref_V_offset_is_set = QtCore.pyqtSignal()
    signal_ref_V_ampl_is_set   = QtCore.pyqtSignal()
    
    class State():
        def __init__(self, buffer_size, N_buffers_in_deque=0):
            """Reflects the actual readings, parsed into separate variables, of
            the lock-in amplifier. There should only be one instance of the
            State class.
            """
            self.buffers_received = 0
            self.buffer_size        = buffer_size           # [samples]
            self.N_buffers_in_deque = N_buffers_in_deque    # [int]
            self.N_deque = buffer_size * N_buffers_in_deque # [samples]
            
            # Predefine arrays for clarity
            # Keep .time as dtype=np.float64, because it can contain np.nan
            self.time   = np.full(buffer_size, np.nan, dtype=np.float64) # [ms]
            self.ref_X  = np.full(buffer_size, np.nan, dtype=np.float64)
            self.ref_Y  = np.full(buffer_size, np.nan, dtype=np.float64)
            self.sig_I  = np.full(buffer_size, np.nan, dtype=np.float64)

            self.time_1 = np.full(buffer_size, np.nan, dtype=np.float64) # [ms]
            self.filt_I = np.full(buffer_size, np.nan, dtype=np.float64)
            self.mix_X  = np.full(buffer_size, np.nan, dtype=np.float64)
            self.mix_Y  = np.full(buffer_size, np.nan, dtype=np.float64)
            
            self.time_2 = np.full(buffer_size, np.nan, dtype=np.float64) # [ms]
            self.X      = np.full(buffer_size, np.nan, dtype=np.float64)
            self.Y      = np.full(buffer_size, np.nan, dtype=np.float64)
            self.R      = np.full(buffer_size, np.nan, dtype=np.float64)
            self.T      = np.full(buffer_size, np.nan, dtype=np.float64)

            self.sig_I_min = np.nan
            self.sig_I_max = np.nan
            self.sig_I_avg = np.nan
            self.sig_I_std = np.nan
            
            """ Deque arrays needed for proper FIR filtering.
            Each time a complete buffer of BUFFER_SIZE samples is received from
            the lock-in, it will extend the deque array (a thread-safe FIFO
            shift buffer).
            
                i.e. N_buffers_in_deque = 3
                    startup          : deque = [no value; no value ; no value]
                    received buffer 1: deque = [buffer_1; no value ; no value]
                    received buffer 2: deque = [buffer_1; buffer_2 ; no value]
                    received buffer 3: deque = [buffer_1; buffer_2 ; buffer_3]
                    received buffer 4: deque = [buffer_2; buffer_3 ; buffer_4]
                    received buffer 5: deque = [buffer_3; buffer_4 ; buffer_5]
                    etc...
            """
            
            # Create deques
            if self.N_buffers_in_deque > 0:
                # Stage 0: unprocessed data
                p = {'capacity': self.N_deque, 'dtype': np.float64}
                self.deque_time   = RingBuffer(**p)
                self.deque_ref_X  = RingBuffer(**p)
                self.deque_ref_Y  = RingBuffer(**p)
                self.deque_sig_I  = RingBuffer(**p)
                # Stage 1: apply band-stop filter and heterodyne mixing
                self.deque_time_1 = RingBuffer(**p)
                self.deque_filt_I = RingBuffer(**p)
                self.deque_mix_X  = RingBuffer(**p)
                self.deque_mix_Y  = RingBuffer(**p)
                # Stage 2: apply low-pass filter and signal reconstruction
                self.deque_time_2 = RingBuffer(**p)
                self.deque_X      = RingBuffer(**p)
                self.deque_Y      = RingBuffer(**p)
                self.deque_R      = RingBuffer(**p)
                self.deque_T      = RingBuffer(**p)
                
                self.deques = [self.deque_time,
                               self.deque_ref_X,
                               self.deque_ref_Y,
                               self.deque_sig_I,
                               self.deque_time_1,
                               self.deque_filt_I,
                               self.deque_mix_X,
                               self.deque_mix_Y,
                               self.deque_time_2,
                               self.deque_X,
                               self.deque_Y,
                               self.deque_R,
                               self.deque_T]
            
            # Mutex for proper multithreading. If the state variables are not
            # atomic or thread-safe, you should lock and unlock this mutex for
            # each read and write operation.
            self.mutex = QtCore.QMutex()
        
        def reset(self):
            """Clears the received buffer counter and clears all deques.
            """
            locker = QtCore.QMutexLocker(self.mutex)
            
            self.buffers_received = 0
            if self.N_buffers_in_deque > 0:
                for this_deque in self.deques:
                    this_deque.clear()
                    
            locker.unlock()
                
    def __init__(self,
                 dev: lockin_functions.Arduino_lockin_amp,
                 DAQ_update_interval_ms=1000,
                 DAQ_function_to_run_each_update=None,
                 DAQ_critical_not_alive_count=3,
                 DAQ_timer_type=QtCore.Qt.PreciseTimer,
                 DAQ_trigger_by=Dev_Base_pyqt_lib.DAQ_trigger.CONTINUOUS,
                 calc_DAQ_rate_every_N_iter=25,
                 N_buffers_in_deque=41,
                 DEBUG_worker_DAQ=False,
                 DEBUG_worker_send=False,
                 use_CUDA=False,
                 parent=None):
        super(Arduino_lockin_amp_pyqt, self).__init__(parent=parent)

        self.attach_device(dev)

        self.create_worker_DAQ(
                DAQ_update_interval_ms,
                DAQ_function_to_run_each_update,
                DAQ_critical_not_alive_count,
                DAQ_timer_type,
                DAQ_trigger_by,
                calc_DAQ_rate_every_N_iter=calc_DAQ_rate_every_N_iter,
                DEBUG=DEBUG_worker_DAQ)

        self.create_worker_send(
                alt_process_jobs_function=
                self.alt_process_jobs_function,
                DEBUG=DEBUG_worker_send)
        
        self.state = self.State(dev.config.BUFFER_SIZE, N_buffers_in_deque)
        self.use_CUDA = use_CUDA
        
        # Create FIR filter: Band-stop on sig_I
        # TODO: turn 'use_narrower_filter' into a toggle button UI
        use_narrower_filter = False
        if use_narrower_filter:
            firwin_cutoff = [  0.5,
                              49.5,  50.5,
                              99.5, 100.5,
                             149.5, 150.5]
            firwin_window = ("chebwin", 50)
        else:
            firwin_cutoff = [  1.0,
                              49.0,  51.0,
                              99.0, 101.0,
                             149.0, 151.0]
            firwin_window = "blackmanharris"
        self.firf_1_sig_I = Buffered_FIR_Filter(self.state.buffer_size,
                                                self.state.N_buffers_in_deque,
                                                dev.config.Fs,
                                                firwin_cutoff,
                                                firwin_window,
                                                pass_zero=False,
                                                display_name="firf_1_sig_I",
                                                use_CUDA=self.use_CUDA)
    
        # Create FIR filter: Low-pass on mix_X and mix_Y
        # TODO: the extra distance 'roll_off_width' to stay away from
        # f_cutoff should be calculated based on the roll-off width of the
        # filter, instead of hard-coded
        roll_off_width = 2; # [Hz]
        firwin_cutoff = 2*dev.config.ref_freq - roll_off_width
        firwin_window = "blackmanharris"
        self.firf_2_mix_X = Buffered_FIR_Filter(self.state.buffer_size,
                                                self.state.N_buffers_in_deque,
                                                dev.config.Fs,
                                                firwin_cutoff,
                                                firwin_window,
                                                pass_zero=True,
                                                display_name="firf_2_mix_X",
                                                use_CUDA=self.use_CUDA)
        self.firf_2_mix_Y = Buffered_FIR_Filter(self.state.buffer_size,
                                                self.state.N_buffers_in_deque,
                                                dev.config.Fs,
                                                firwin_cutoff,
                                                firwin_window,
                                                pass_zero=True,
                                                display_name="firf_2_mix_Y",
                                                use_CUDA=self.use_CUDA)
        
    def turn_on(self):
        self.worker_send.queued_instruction("turn_on")
    
    def turn_on_immediately(self):
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
        self.worker_send.queued_instruction("turn_off")
    
    def turn_off_immediately(self):
        """
        Returns:
            success
        """
        self.worker_DAQ.schedule_suspend()
        
        tick = Time.time()
        TIMEOUT = 2 # [s]
        while not self.worker_DAQ.suspended:
            QtWid.QApplication.processEvents()
            if Time.time() - tick > TIMEOUT:
                print("Wait for worker_DAQ to reach suspended state timed out. "
                      "Brute forcing turn off.")
                break
        
        locker = QtCore.QMutexLocker(self.dev.mutex)
        [success, foo, bar] = self.dev.turn_off()
        locker.unlock()
        return success
    
    def set_ref_freq(self, ref_freq):
        self.worker_send.queued_instruction("set_ref_freq", ref_freq)
        
    def set_ref_V_offset(self, ref_V_offset):
        self.worker_send.queued_instruction("set_ref_V_offset", ref_V_offset)
        
    def set_ref_V_ampl(self, ref_V_ampl):
        self.worker_send.queued_instruction("set_ref_V_ampl", ref_V_ampl)
    
    # -------------------------------------------------------------------------
    #   alt_process_jobs_function
    # -------------------------------------------------------------------------

    def alt_process_jobs_function(self, func, args):
        if func[:8] == "set_ref_":
            set_value = args[0]
        
            if func == "set_ref_freq":
                current_value = self.dev.config.ref_freq
            elif func == "set_ref_V_offset":
                current_value = self.dev.config.ref_V_offset
            elif func == "set_ref_V_ampl":
                current_value = self.dev.config.ref_V_ampl
            else:
                current_value = 0
        
            if not (set_value == current_value):
                was_paused = self.dev.lockin_paused
                
                if not was_paused:
                    self.worker_DAQ.schedule_suspend()
                    while not self.worker_DAQ.suspended:
                        QtWid.QApplication.processEvents()
        
                if func == "set_ref_freq":
                    self.dev.set_ref_freq(set_value)
                    self.signal_ref_freq_is_set.emit()
                elif func == "set_ref_V_offset":
                    self.dev.set_ref_V_offset(set_value)
                    self.signal_ref_V_offset_is_set.emit()
                elif func == "set_ref_V_ampl":
                    self.dev.set_ref_V_ampl(set_value)
                    self.signal_ref_V_ampl_is_set.emit()
                
                if not was_paused:
                    self.worker_DAQ.schedule_suspend(False)
                    QtWid.QApplication.processEvents()
                    
        elif func == "turn_on":
            self.state.reset()
            if self.dev.turn_on():
                self.worker_DAQ.schedule_suspend(False)
                
        elif func == "turn_off":
            self.worker_DAQ.schedule_suspend()
            while not self.worker_DAQ.suspended:
                QtWid.QApplication.processEvents()
            self.dev.turn_off()
                    
        else:
            # Default job handling
            func(*args)
            
    # -------------------------------------------------------------------------
    #   compute_power_spectrum
    # -------------------------------------------------------------------------

    def compute_power_spectrum(self, deque_in: RingBuffer):
        """Using scipy.signal.welch()
        When scaling='spectrum', Pxx returns units of V^2
        When scaling='density', Pxx returns units of V^2/Hz
        Note: Amplitude ratio in dB: 20 log_10(A1/A2)
              Power     ratio in dB: 10 log_10(P1/P2)
        """
        [f, Pxx] = welch(deque_in,
                         fs=self.dev.config.Fs,
                         window='hanning',
                         nperseg=self.dev.config.Fs,
                         detrend=False,
                         scaling='spectrum')
        
        return [f, 10 * np.log10(Pxx)]
