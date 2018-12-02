#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PyQt5 module to provide the base framework for multithreaded communication
and periodical data acquisition for an I/O device.

MAIN CONTENTS:
--------------

    Class:
        Dev_Base_pyqt(...)
            Methods:
                attach_device(...)
                create_worker_DAQ()
                create_worker_send()
                start_thread_worker_DAQ(...)
                start_thread_worker_send(...)
                close_thread_worker_DAQ()
                close_thread_worker_send()
                close_all_threads()

            Inner-class instances:
                worker_DAQ(...)
                    Methods:
                        wake_up(...)

                worker_send(...):
                    Methods:
                        add_to_queue(...)
                        process_queue()
                        queued_instruction(...)

            Main data attributes:
                DAQ_update_counter
                obtained_DAQ_update_interval_ms
                obtained_DAQ_rate_Hz

            Signals:
                signal_DAQ_updated()
                signal_connection_lost()
"""
__author__      = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__         = "https://github.com/Dennis-van-Gils/DvG_dev_Arduino"
__date__        = "19-09-2018"
__version__     = "1.1.0"

from enum import IntEnum, unique
import queue
import numpy as np
from PyQt5 import QtCore
from DvG_debug_functions import ANSI, dprint, print_fancy_traceback as pft

# Short-hand alias for DEBUG information
def curThreadName(): return QtCore.QThread.currentThread().objectName()

@unique
class DAQ_trigger(IntEnum):
    [INTERNAL_TIMER, EXTERNAL_WAKE_UP_CALL] = range(2)

# ------------------------------------------------------------------------------
#   InnerClassDescriptor
# ------------------------------------------------------------------------------

class InnerClassDescriptor(object):
    """Allows an inner class instance to get the attributes from the outer class
    instance by referring to 'self.outer'. Used in this module by the
    'Worker_DAQ' and 'Worker_send' classes. Usage: @InnerClassDescriptor.
    Not to be used outside of this module.
    """
    def __init__(self, cls):
        self.cls = cls

    def __get__(self, instance, outerclass):
        class Wrapper(self.cls):
            outer = instance
        Wrapper.__name__ = self.cls.__name__
        return Wrapper

# ------------------------------------------------------------------------------
#   Dev_Base_pyqt
# ------------------------------------------------------------------------------

class Dev_Base_pyqt(QtCore.QObject):
    """This class provides the base framework for multithreaded communication
    and periodical data acquisition for an I/O device.

    All device I/O operations will be offloaded to 'workers', each running in
    a newly created thread instead of in the main/GUI thread.

        - Worker_DAQ:
            Periodically acquires data from the device.

        - Worker_send:
            Maintains a thread-safe queue where desired device I/O operations
            can be put onto, and sends the queued operations first in first out
            (FIFO) to the device.

    This class can be mixed into your own specific device_pyqt class definition.
    Hint: Look up 'mixin class' for Python.
    E.g., when writing your own Arduino device pyqt library:
        class Arduino_pyqt(Dev_Base_pyqt_lib.Dev_Base_pyqt, QtCore.QObject):

    Methods:
        attach_device(...)
            Attach a reference to a 'device' instance with I/O methods.

        create_worker_DAQ():
            Create a single instance of 'Worker_DAQ' and transfer it to a newly
            created (PyQt5.QtCore.QThread) thread called 'thread_DAQ'.

        create_worker_send():
            Create a single instance of 'Worker_send' and transfer it to a newly
            created (PyQt5.QtCore.QThread) thread called 'thread_send'.

        start_thread_worker_DAQ(...):
            Start running the event loop of the 'worker_DAQ' thread.
            I.e., start acquiring data periodically from the device.

        start_thread_worker_send(...):
            Start running the event loop of the 'worker_send' thread.
            I.e., start maintaining the desired device I/O operations queue.

        close_thread_worker_DAQ():
            Stop 'worker_DAQ' and close its thread.

        close_thread_worker_send()
            Stop 'worker_send' and close its thread.

        close_all_threads():
            Stop all of any running workers and close their respective threads.
            Safer and more convenient than calling 'close_thread_worker_DAQ' and
            'close_thread_worker_send', individually.

    Inner-class instances:
        worker_DAQ
        worker_send

    Main data attributes:
        dev:
            Reference to a 'device' instance with I/O methods. Needs to be set
            by calling 'attach_device(...)'.

        dev.mutex (PyQt5.QtCore.QMutex):
            Mutex to allow for properly multithreaded device I/O operations.

        DAQ_update_counter:
            Increments every time 'worker_DAQ' updates.

        obtained_DAQ_update_interval_ms:
            Obtained time interval in milliseconds since the previous
            'worker_DAQ' update.

        obtained_DAQ_rate_Hz:
            Obtained acquisition rate of 'worker_DAQ' in Hertz, evaluated every
            second.

    Signals:
        signal_DAQ_updated:
            Emitted by 'worker_DAQ' when 'update' has finished.

        signal_connection_lost:
            Indicates that we lost connection to the device, because one or more
            device I/O operations failed. Emitted by 'worker_DAQ' during
            'update' when 'DAQ_not_alive_counter' is equal to or larger than
            'worker_DAQ.critical_not_alive_count'.
    """
    signal_DAQ_updated     = QtCore.pyqtSignal()
    signal_connection_lost = QtCore.pyqtSignal()

    def __init__(self, parent):
        super(Dev_Base_pyqt, self).__init__(parent=parent)

        self.dev = self.NoAttachedDevice()
        self.worker_DAQ = None
        self.worker_send = None

        self.DAQ_update_counter = 0
        self.DAQ_not_alive_counter = 0

        self.obtained_DAQ_update_interval_ms = np.nan
        self.obtained_DAQ_rate_Hz = np.nan

    class NoAttachedDevice():
        name = "NoAttachedDevice"
        is_alive = False
        mutex = None

    # --------------------------------------------------------------------------
    #   attach_device
    # --------------------------------------------------------------------------

    def attach_device(self, dev):
        """Attach a reference to a 'device' instance with I/O methods.
        """
        if type(self.dev) == self.NoAttachedDevice:
            self.dev = dev
            self.dev.mutex = QtCore.QMutex()
        else:
            pft("Device can be attached only once. Already attached to '%s'." %
                self.dev.name)

    # --------------------------------------------------------------------------
    #   Create workers
    # --------------------------------------------------------------------------

    def create_worker_DAQ(self, *args, **kwargs):
        self.worker_DAQ = self.Worker_DAQ(*args, **kwargs)

        if self.dev.is_alive:
            self.thread_DAQ = QtCore.QThread()
            self.thread_DAQ.setObjectName("%s_DAQ" % self.dev.name)
            self.worker_DAQ.moveToThread(self.thread_DAQ)
            self.thread_DAQ.started.connect(self.worker_DAQ.run)
        else:
            self.thread_DAQ = None

    def create_worker_send(self, *args, **kwargs):
        self.worker_send = self.Worker_send(*args, **kwargs)

        if self.dev.is_alive:
            self.thread_send = QtCore.QThread()
            self.thread_send.setObjectName("%s_send" % self.dev.name)
            self.worker_send.moveToThread(self.thread_send)
            self.thread_send.started.connect(self.worker_send.run)
        else:
            self.thread_send = None

    # --------------------------------------------------------------------------
    #   Start threads
    # --------------------------------------------------------------------------

    def start_thread_worker_DAQ(self, priority=QtCore.QThread.InheritPriority):
        """Start running the event loop of the worker thread.

        Args:
            priority (PyQt5.QtCore.QThread.Priority, optional, default=
                      QtCore.QThread.InheritPriority):
                By default, the 'worker_DAQ' thread runs in the operating system
                at the same thread priority as the main/GUI thread. You can
                change to higher priority by setting 'priority' to, e.g.,
                'QtCore.QThread.TimeCriticalPriority'. Be aware that this is
                resource heavy, so use sparingly.

        Returns True when successful, False otherwise.
        """
        if hasattr(self, 'thread_DAQ'):
            if self.thread_DAQ is not None:
                self.thread_DAQ.start(priority)
                return True
            else:
                print("Worker_DAQ %s: Can't start thread because device is "
                      "not alive." % self.dev.name)
                return False
        else:
            pft("Worker_DAQ %s: Can't start thread because it does not exist. "
                "Did you forget to call 'create_worker_DAQ' first?" %
                self.dev.name)
            return False

    def start_thread_worker_send(self, priority=QtCore.QThread.InheritPriority):
        """Start running the event loop of the worker thread.

        Args:
            priority (PyQt5.QtCore.QThread.Priority, optional, default=
                      QtCore.QThread.InheritPriority):
                By default, the 'worker_send' thread runs in the operating system
                at the same thread priority as the main/GUI thread. You can
                change to higher priority by setting 'priority' to, e.g.,
                'QtCore.QThread.TimeCriticalPriority'. Be aware that this is
                resource heavy, so use sparingly.

        Returns True when successful, False otherwise.
        """
        if hasattr(self, 'thread_send'):
            if self.thread_send is not None:
                self.thread_send.start(priority)
                return True
            else:
                print("Worker_send %s: Can't start thread because device is "
                      "not alive." % self.dev.name)
                return False
        else:
            pft("Worker_send %s: Can't start thread because it does not exist. "
                "Did you forget to call 'create_worker_send' first?" %
                self.dev.name)
            return False

    # --------------------------------------------------------------------------
    #   Close threads
    # --------------------------------------------------------------------------

    def close_thread_worker_DAQ(self):
        if self.thread_DAQ is not None:
            if (self.worker_DAQ.trigger_by ==
                DAQ_trigger.EXTERNAL_WAKE_UP_CALL):
                self.worker_DAQ.stop()
                self.worker_DAQ.qwc.wakeAll()
            self.thread_DAQ.quit()
            print("Closing thread %s " %
                  "{:.<16}".format(self.thread_DAQ.objectName()), end='')
            if self.thread_DAQ.wait(2000): print("done.\n", end='')
            else: print("FAILED.\n", end='')

    def close_thread_worker_send(self):
        if self.thread_send is not None:
            self.worker_send.stop()
            self.worker_send.qwc.wakeAll()
            self.thread_send.quit()
            print("Closing thread %s " %
                  "{:.<16}".format(self.thread_send.objectName()), end='')
            if self.thread_send.wait(2000): print("done.\n", end='')
            else: print("FAILED.\n", end='')

    def close_all_threads(self):
        if hasattr(self, 'thread_DAQ') : self.close_thread_worker_DAQ()
        if hasattr(self, 'thread_send'): self.close_thread_worker_send()

    # --------------------------------------------------------------------------
    #   Worker_DAQ
    # --------------------------------------------------------------------------

    @InnerClassDescriptor
    class Worker_DAQ(QtCore.QObject):
        """This worker acquires data from the device at a fixed update interval.
        It does so by calling a user-supplied function containing your device
        I/O operations (and data parsing, processing or more), every update
        period.

        The worker should be placed inside a separate thread. No direct changes
        to the GUI should be performed inside this class. If needed, use the
        QtCore.pyqtSignal() mechanism to instigate GUI changes.

        The Worker_DAQ routine is robust in the following sense. It can be set
        to quit as soon as a communication error appears, or it could be set to
        allow a certain number of communication errors before it quits. The
        latter can be useful in non-critical implementations where continuity of
        the program is of more importance than preventing drops in data
        transmission. This, obviously, is a work-around for not having to tackle
        the source of the communication error, but sometimes you just need to
        struggle on. E.g., when your Arduino is out in the field and picks up
        occasional unwanted interference/ground noise that messes with your data
        transmission.

        Args:
            DAQ_update_interval_ms:
                Desired data acquisition update interval in milliseconds.

            DAQ_function_to_run_each_update (optional, default=None):
                Reference to a user-supplied function containing the device
                query operations and subsequent data processing, to be invoked
                every DAQ update. It should return True when everything went
                successful, and False otherwise.

                NOTE: No direct changes to the GUI should run inside this
                function! If you do anyhow, expect a penalty in the timing
                stability of this worker.

                E.g. pseudo-code, where 'time' and 'reading_1' are variables
                that live at a higher scope, presumably at main/GUI scope level.

                def my_update_function():
                    # Query the device for its state
                    [success, tmp_state] = dev.query_ascii_values("state?")
                    if not(success):
                        print("Device IOerror")
                        return False

                    # Parse readings into separate variables
                    try:
                        [time, reading_1] = tmp_state
                    except Exception as err:
                        print(err)
                        return False

                    return True

            DAQ_critical_not_alive_count (optional, default=1):
                The worker will allow for up to a certain number of
                communication failures with the device before hope is given up
                and a 'connection lost' signal is emitted. Use at your own
                discretion.

            DAQ_timer_type (PyQt5.QtCore.Qt.TimerType, optional, default=
                            PyQt5.QtCore.Qt.CoarseTimer):
                The update interval is timed to a QTimer running inside
                Worker_DAQ. The accuracy of the timer can be improved by setting
                it to PyQt5.QtCore.Qt.PreciseTimer with ~1 ms granularity, but
                it is resource heavy. Use sparingly.

            DAQ_trigger_by (optional, default=DAQ_trigger.INTERNAL_TIMER):
                TO DO: write description

            DEBUG (bool, optional, default=False):
                Show debug info in terminal? Warning: Slow! Do not leave on
                unintentionally.
        """
        def __init__(self,
                     DAQ_update_interval_ms,
                     DAQ_function_to_run_each_update=None,
                     DAQ_critical_not_alive_count=1,
                     DAQ_timer_type=QtCore.Qt.CoarseTimer,
                     DAQ_trigger_by=DAQ_trigger.INTERNAL_TIMER,
                     DEBUG=False):
            super().__init__(None)
            self.DEBUG = DEBUG
            self.DEBUG_color = ANSI.CYAN

            self.dev = self.outer.dev
            self.update_interval_ms = DAQ_update_interval_ms
            self.function_to_run_each_update = DAQ_function_to_run_each_update
            self.critical_not_alive_count = DAQ_critical_not_alive_count
            self.timer_type = DAQ_timer_type
            self.trigger_by = DAQ_trigger_by

            if self.trigger_by == DAQ_trigger.EXTERNAL_WAKE_UP_CALL:
                self.qwc = QtCore.QWaitCondition()
                self.mutex_wait = QtCore.QMutex()
                self.running = True

            self.calc_DAQ_rate_every_N_iter = max(
                    round(1e3/self.update_interval_ms), 1)
            self.prev_tick_DAQ_update = 0
            self.prev_tick_DAQ_rate = 0

            if self.DEBUG:
                dprint("Worker_DAQ  %s init: thread %s" %
                       (self.dev.name, curThreadName()), self.DEBUG_color)

        @QtCore.pyqtSlot()
        def run(self):
            if self.DEBUG:
                dprint("Worker_DAQ  %s run : thread %s" %
                       (self.dev.name, curThreadName()), self.DEBUG_color)

            # INTERNAL TIMER
            if self.trigger_by == DAQ_trigger.INTERNAL_TIMER:
                self.timer = QtCore.QTimer()
                self.timer.setInterval(self.update_interval_ms)
                self.timer.timeout.connect(self.update)
                self.timer.setTimerType(self.timer_type)
                self.timer.start()

            # EXTERNAL WAKE UP
            elif self.trigger_by == DAQ_trigger.EXTERNAL_WAKE_UP_CALL:
                while self.running:
                    locker_wait = QtCore.QMutexLocker(self.mutex_wait)

                    if self.DEBUG:
                        dprint("Worker_DAQ  %s: waiting for trigger" %
                               self.dev.name, self.DEBUG_color)

                    self.qwc.wait(self.mutex_wait)
                    self.update()

                    locker_wait.unlock()

                if self.DEBUG:
                    dprint("Worker_DAQ  %s: done running" % self.dev.name,
                           self.DEBUG_color)

        @QtCore.pyqtSlot()
        def stop(self):
            """Only useful with DAQ_trigger.EXTERNAL_WAKE_UP_CALL
            """
            self.running = False

        @QtCore.pyqtSlot()
        def update(self):
            locker = QtCore.QMutexLocker(self.dev.mutex)
            self.outer.DAQ_update_counter += 1

            if self.DEBUG:
                dprint("Worker_DAQ  %s: lock %i" %
                       (self.dev.name, self.outer.DAQ_update_counter),
                       self.DEBUG_color)

            # Keep track of the obtained DAQ update interval
            now = QtCore.QDateTime.currentMSecsSinceEpoch()
            if self.outer.DAQ_update_counter > 1:
                self.outer.obtained_DAQ_update_interval_ms = (
                        now - self.prev_tick_DAQ_update)
            self.prev_tick_DAQ_update = now

            # Keep track of the obtained DAQ rate
            # Start at iteration 5 to ensure we have stabilized
            if self.outer.DAQ_update_counter == 5:
                self.prev_tick_DAQ_rate = now
            elif (self.outer.DAQ_update_counter %
                  self.calc_DAQ_rate_every_N_iter == 5):
                self.outer.obtained_DAQ_rate_Hz = (
                        self.calc_DAQ_rate_every_N_iter /
                        (now - self.prev_tick_DAQ_rate) * 1e3)
                self.prev_tick_DAQ_rate = now

            # Check the not alive counter
            if (self.outer.DAQ_not_alive_counter >=
                self.critical_not_alive_count):
                dprint("\nWorker_DAQ %s: Determined device is not alive "
                       "anymore." % self.dev.name)
                self.dev.is_alive = False

                locker.unlock()
                if self.trigger_by == DAQ_trigger.INTERNAL_TIMER:
                    self.timer.stop()
                elif self.trigger_by == DAQ_trigger.EXTERNAL_WAKE_UP_CALL:
                    self.stop()
                self.outer.signal_DAQ_updated.emit()
                self.outer.signal_connection_lost.emit()
                return

            # ----------------------------------
            #   User-supplied DAQ function
            # ----------------------------------

            if not(self.function_to_run_each_update is None):
                if not(self.function_to_run_each_update()):
                    self.outer.DAQ_not_alive_counter += 1

            # ----------------------------------
            #   End user-supplied DAQ function
            # ----------------------------------

            if self.DEBUG:
                dprint("Worker_DAQ  %s: unlocked" % self.dev.name,
                       self.DEBUG_color)

            locker.unlock()
            self.outer.signal_DAQ_updated.emit()

        # ----------------------------------------------------------------------
        #   wake_up
        # ----------------------------------------------------------------------

        def wake_up(self):
            if self.trigger_by == DAQ_trigger.EXTERNAL_WAKE_UP_CALL:
                self.qwc.wakeAll()

    # --------------------------------------------------------------------------
    #   Worker_send
    # --------------------------------------------------------------------------

    @InnerClassDescriptor
    class Worker_send(QtCore.QObject):
        """This worker maintains a thread-safe queue where desired device I/O
        operations, a.k.a. jobs, can be put onto. The worker will send out the
        operations to the device, first in first out (FIFO), until the queue is
        empty again.

        The worker should be placed inside a separate thread. This worker uses
        the QWaitCondition mechanism. Hence, it will only send out all
        operations collected in the queue, whenever the thread it lives in is
        woken up by calling 'Worker_send.process_queue()'. When it has emptied
        the queue, the thread will go back to sleep again.

        No direct changes to the GUI should be performed inside this class. If
        needed, use the QtCore.pyqtSignal() mechanism to instigate GUI changes.

        Args:
            alt_process_jobs_function (optional, default=None):
                Reference to an user-supplied function performing an alternative
                job handling when processing the worker_send queue. The default
                job handling effectuates calling 'func(*args)', where 'func' and
                'args' are retrieved from the worker_send queue, and nothing
                more. The default is sufficient when 'func' corresponds to an
                I/O operation that is an one-way send, i.e. a write operation
                without a reply.

                Instead of just write operations, you can also put query
                operations in the queue and process each reply of the device
                accordingly. This is the purpose of this argument: To provide
                your own 'job processing routines' function. The function you
                supply must take two arguments, where the first argument will be
                'func' and the second argument will be 'args', which is a tuple.
                Both 'func' and 'args' will be retrieved from the worker_send
                queue and passed onto your own function.

                Example of a query operation by sending and checking for a
                special string value of 'func':

                    def my_alt_process_jobs_function(func, args):
                        if func == "query_id?":
                            # Query the device for its identity string
                            [success, ans_str] = self.dev.query("id?")
                            # And store the reply 'ans_str' in another variable
                            # at a higher scope or do stuff with it here.
                        elif:
                            # Default job handling where, e.g.
                            # func = self.dev.write
                            # args = ("toggle LED",)
                            func(*args)

            DEBUG (bool, optional, default=False):
                Show debug info in terminal? Warning: Slow! Do not leave on
                unintentionally.

        Methods:
            add_to_queue(...):
                Put an instruction on the worker_send queue.

            process_queue():
                Trigger processing the worker_send queue.

            queued_instruction(...):
                Put an instruction on the worker_send queue and process the
                queue.
        """

        def __init__(self,
                     alt_process_jobs_function=None,
                     DEBUG=False):
            super().__init__(None)
            self.DEBUG = DEBUG
            self.DEBUG_color = ANSI.YELLOW

            self.dev = self.outer.dev
            self.alt_process_jobs_function = alt_process_jobs_function

            self.update_counter = 0
            self.qwc = QtCore.QWaitCondition()
            self.mutex_wait = QtCore.QMutex()
            self.running = True

            # Use a 'sentinel' value to signal the start and end of the queue
            # to ensure proper multithreaded operation.
            self.sentinel = None
            self.queue = queue.Queue()
            self.queue.put(self.sentinel)

            if self.DEBUG:
                dprint("Worker_send %s init: thread %s" %
                       (self.dev.name, curThreadName()), self.DEBUG_color)

        @QtCore.pyqtSlot()
        def run(self):
            if self.DEBUG:
                dprint("Worker_send %s run : thread %s" %
                       (self.dev.name, curThreadName()), self.DEBUG_color)

            while self.running:
                locker_wait = QtCore.QMutexLocker(self.mutex_wait)

                if self.DEBUG:
                    dprint("Worker_send %s: waiting for trigger" %
                           self.dev.name, self.DEBUG_color)

                self.qwc.wait(self.mutex_wait)
                locker = QtCore.QMutexLocker(self.dev.mutex)
                self.update_counter += 1

                if self.DEBUG:
                    dprint("Worker_send %s: lock %i" %
                           (self.dev.name, self.update_counter),
                           self.DEBUG_color)

                """Process all jobs until the queue is empty. We must iterate 2
                times because we use a sentinel in a FIFO queue. First iter
                removes the old sentinel. Second iter processes the remaining
                queue items and will put back a new sentinel again.
                """
                for i in range(2):
                    for job in iter(self.queue.get_nowait, self.sentinel):
                        func = job[0]
                        args = job[1:]

                        if self.DEBUG:
                            if type(func) == str:
                                dprint("Worker_send %s: %s %s" %
                                       (self.dev.name, func, args),
                                       self.DEBUG_color)
                            else:
                                dprint("Worker_send %s: %s %s" %
                                       (self.dev.name, func.__name__, args),
                                       self.DEBUG_color)

                        if self.alt_process_jobs_function is None:
                            # Default job processing:
                            # Send I/O operation to the device
                            try:
                                func(*args)
                            except Exception as err:
                                pft(err)
                        else:
                            # User-supplied job processing
                            self.alt_process_jobs_function(func, args)

                    # Put sentinel back in
                    self.queue.put(self.sentinel)

                if self.DEBUG:
                    dprint("Worker_send %s: unlocked" % self.dev.name,
                           self.DEBUG_color)

                locker.unlock()
                locker_wait.unlock()

            if self.DEBUG:
                dprint("Worker_send %s: done running" % self.dev.name,
                       self.DEBUG_color)

        @QtCore.pyqtSlot()
        def stop(self):
            self.running = False

        # ----------------------------------------------------------------------
        #   add_to_queue
        # ----------------------------------------------------------------------

        def add_to_queue(self, instruction, pass_args=()):
            """Put an instruction on the worker_send queue.
            E.g. add_to_queue(self.dev.write, "toggle LED")

            Args:
                instruction:
                    Intended to be a reference to a device I/O function such as
                    'self.dev.write'. However, you have the freedom to be
                    creative and put e.g. strings decoding special instructions
                    on the queue as well. Handling such special cases must be
                    programmed by the user by supplying the argument
                    'alt_process_jobs_function', when instantiating
                    'Worker_send', with your own job-processing-routines
                    function. See 'Worker_send' for more details.

                pass_args (optional, default=()):
                    Argument(s) to be passed to the instruction. Must be a
                    tuple, but for convenience any other type will also be
                    accepted if it concerns just a single argument that needs to
                    be passed.
            """
            if type(pass_args) is not tuple: pass_args = (pass_args,)
            self.queue.put((instruction, *pass_args))

        # ----------------------------------------------------------------------
        #   process_queue
        # ----------------------------------------------------------------------

        def process_queue(self):
            """Trigger processing the worker_send queue.
            """
            self.qwc.wakeAll()

        # ----------------------------------------------------------------------
        #   queued_instruction
        # ----------------------------------------------------------------------

        def queued_instruction(self, instruction, pass_args=()):
            """Put an instruction on the worker_send queue and process the
            queue. See 'add_to_queue' for more details.
            """
            self.add_to_queue(instruction, pass_args)
            self.process_queue()
