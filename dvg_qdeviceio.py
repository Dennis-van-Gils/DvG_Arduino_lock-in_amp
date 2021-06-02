#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PyQt5 framework for multithreaded data acquisition and communication with an
I/O device.
"""
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/python-dvg-qdeviceio"
__date__ = "09-05-2021"
__version__ = "0.4.0"
# pylint: disable=protected-access

from enum import IntEnum, unique
import queue
import time

# Code coverage tools 'coverage' and 'pytest-cov' don't seem to correctly trace
# code which is inside methods called from within QThreads, see
# https://github.com/nedbat/coveragepy/issues/686
# To mitigate this problem, I use a custom decorator '@_coverage_resolve_trace'
# to be hung onto those method definitions. This will prepend the decorated
# method code with 'sys.settrace(threading._trace_hook)' when a code
# coverage test is detected. When no coverage test is detected, it will just
# pass the original method untouched.
import sys
import threading
from functools import wraps

import numpy as np
from PyQt5 import QtCore
from dvg_debug_functions import (
    print_fancy_traceback as pft,
    dprint,
    tprint,
    ANSI,
)

running_coverage = "coverage" in sys.modules
if running_coverage:
    print("\nCode coverage test detected\n")


def _coverage_resolve_trace(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        if running_coverage:
            sys.settrace(threading._trace_hook)
        fn(*args, **kwargs)

    return wrapped


# Short-hand alias for DEBUG information
def _cur_thread_name():
    return QtCore.QThread.currentThread().objectName()


@unique
class DAQ_TRIGGER(IntEnum):
    """An enumeration decoding different modes of operation for
    :class:`Worker_DAQ` to perform a data-acquisition (DAQ) update.
    """

    # fmt: off
    INTERNAL_TIMER = 0       #: :ref:`Link to background information <INTERNAL_TIMER>`.
    SINGLE_SHOT_WAKE_UP = 1  #: :ref:`Link to background information <SINGLE_SHOT_WAKE_UP>`.
    CONTINUOUS = 2           #: :ref:`Link to background information <CONTINUOUS>`.
    # fmt: on


# ------------------------------------------------------------------------------
#   QDeviceIO
# ------------------------------------------------------------------------------


class QDeviceIO(QtCore.QObject):
    """This class provides the framework for multithreaded data acquisition
    (DAQ) and communication with an I/O device.

    All device I/O operations will be offloaded to *workers*, each running in
    their dedicated thread. The following workers can be created:

    * :class:`Worker_DAQ` :

        Acquires data from the device, either periodically or aperiodically.

        Created by calling :meth:`create_worker_DAQ`.

    * :class:`Worker_jobs` :

        Maintains a thread-safe queue where desired device I/O operations,
        called *jobs*, can be put onto. It will send out the queued jobs
        first-in, first-out (FIFO) to the device.

        Created by calling :meth:`create_worker_jobs`.

    Tip:
        You can inherit from `QDeviceIO` to build your own subclass that
        hides the specifics of creating :class:`Worker_DAQ` and
        :class:`Worker_jobs` from the user and modifies the default parameter
        values. E.g., when making a `QDeviceIO` subclass specific to your
        Arduino project::

            from dvg_qdeviceio import QDeviceIO, DAQ_TRIGGER

            class Arduino_qdev(QDeviceIO):
                def __init__(
                    self, dev=None, DAQ_function=None, debug=False, **kwargs,
                ):
                    # Pass `dev` onto QDeviceIO() and pass `**kwargs` onto QtCore.QObject()
                    super().__init__(dev, **kwargs)

                    # Set the DAQ to 10 Hz internal timer
                    self.create_worker_DAQ(
                        DAQ_trigger                = DAQ_TRIGGER.INTERNAL_TIMER,
                        DAQ_function               = DAQ_function,
                        DAQ_interval_ms            = 100,  # 100 ms -> 10 Hz
                        critical_not_alive_count   = 3,
                        debug                      = debug,
                    )

                    # Standard jobs handling
                    self.create_worker_jobs(debug=debug)

        Now, the user only has to call the following to get up and running::

            ard_qdev = Arduino_qdev(
                dev=my_Arduino_device,
                DAQ_function=my_DAQ_function
            )
            ard_qdev.start()

    .. _`QDeviceIO_args`:

    Args:
        dev (:obj:`object` | :obj:`None`, optional):
            Reference to a user-supplied *device* class instance containing
            I/O methods. In addition, `dev` should also have the following
            members. If not, they will be injected into the `dev` instance for
            you:

                * **dev.name** (:obj:`str`) -- Short display name for the \
                    device. Default: "myDevice".

                * **dev.mutex** (:class:`PyQt5.QtCore.QMutex`) -- To allow \
                    for properly multithreaded device I/O operations. It will \
                    be used by :class:`Worker_DAQ` and :class:`Worker_jobs`.

                * **dev.is_alive** (:obj:`bool`) -- Device is up and \
                    communicatable? Default: :const:`True`.

            Default: :obj:`None`

        **kwargs:
            All remaining keyword arguments will be passed onto inherited class
            :class:`PyQt5.QtCore.QObject`.

    .. _`QDeviceIO_attributes`:
    .. rubric:: Attributes:

    Attributes:
        dev (:obj:`object` | :obj:`None`):
            Reference to a user-supplied *device* class instance containing
            I/O methods.

        worker_DAQ (:class:`Worker_DAQ` | :obj:`None`):
            Instance of :class:`Worker_DAQ` as created by
            :meth:`create_worker_DAQ`. This worker runs in a dedicated thread.

        worker_jobs (:class:`Worker_jobs` | :obj:`None`):
            Instance of :class:`Worker_jobs` as created by
            :meth:`create_worker_jobs`. This worker runs in a dedicated thread.

        update_counter_DAQ (:obj:`int`):
            Increments every time :attr:`worker_DAQ` tries to update.

        update_counter_jobs (:obj:`int`):
            Increments every time :attr:`worker_jobs` tries to update.

        obtained_DAQ_interval_ms (:obj:`int` | :obj:`numpy.nan`):
            Obtained time interval in milliseconds since the previous
            :attr:`worker_DAQ` update.

        obtained_DAQ_rate_Hz (:obj:`float` | :obj:`numpy.nan`):
            Obtained acquisition rate of :attr:`worker_DAQ` in hertz. It will
            take several DAQ updates for the value to be properly calculated,
            and till that time it will be :obj:`numpy.nan`.

        not_alive_counter_DAQ (:obj:`int`):
            Number of consecutive failed attempts to update :attr:`worker_DAQ`,
            presumably due to device I/O errors. Will be reset to 0 once a
            successful DAQ update occurs. See the
            :obj:`signal_connection_lost()` mechanism.
    """

    signal_DAQ_updated = QtCore.pyqtSignal()
    """:obj:`PyQt5.QtCore.pyqtSignal`: Emitted by :class:`Worker_DAQ` when its
    :attr:`~Worker_DAQ.DAQ_function` has run and finished, either succesfully or
    not.

    Tip:
        It can be useful to connect this signal to a slot containing, e.g.,
        your GUI redraw routine::

            from PyQt5 import QtCore

            @QtCore.pyqtSlot()
            def my_GUI_redraw_routine():
                ...

            qdev.signal_DAQ_updated.connect(my_GUI_redraw_routine)

        where ``qdev`` is your instance of :class:`QDeviceIO`. Don't forget to
        decorate the function definition with a :func:`PyQt5.QtCore.pyqtSlot`
        decorator.
    """

    signal_jobs_updated = QtCore.pyqtSignal()
    """:obj:`PyQt5.QtCore.pyqtSignal`: Emitted by :class:`Worker_jobs` when all
    pending jobs in the queue have been sent out to the device in a response to
    :meth:`send` or :meth:`process_jobs_queue`. See also the tip at
    :obj:`signal_DAQ_updated()`.
    """

    signal_DAQ_paused = QtCore.pyqtSignal()
    """:obj:`PyQt5.QtCore.pyqtSignal`: Emitted by :class:`Worker_DAQ` to confirm
    the worker has entered the `paused` state in a response to
    :meth:`Worker_DAQ.pause`. See also the tip at :obj:`signal_DAQ_updated()`.
    """

    signal_connection_lost = QtCore.pyqtSignal()
    """:obj:`PyQt5.QtCore.pyqtSignal`: Emitted by :class:`Worker_DAQ` to
    indicate that we have lost connection to the device. This happens when `N`
    consecutive device I/O operations have failed, where `N` equals the argument
    :obj:`critical_not_alive_count` as passed to method
    :meth:`create_worker_DAQ`. See also the tip at :obj:`signal_DAQ_updated()`.
    """

    # Necessary for INTERNAL_TIMER
    _request_worker_DAQ_stop = QtCore.pyqtSignal()

    # Necessary for CONTINUOUS
    _request_worker_DAQ_pause = QtCore.pyqtSignal()
    _request_worker_DAQ_unpause = QtCore.pyqtSignal()

    def __init__(self, dev=None, **kwargs):
        super().__init__(**kwargs)  # Pass **kwargs onto QtCore.QObject()

        self.dev = self._NoDevice()
        if dev is not None:
            self.attach_device(dev)

        self._thread_DAQ = None
        self._thread_jobs = None

        self.worker_DAQ = None
        self.worker_jobs = None

        self.update_counter_DAQ = 0
        self.update_counter_jobs = 0
        self.not_alive_counter_DAQ = 0

        self.obtained_DAQ_interval_ms = np.nan
        self.obtained_DAQ_rate_Hz = np.nan

        self._qwc_worker_DAQ_started = QtCore.QWaitCondition()
        self._qwc_worker_jobs_started = QtCore.QWaitCondition()

        self._qwc_worker_DAQ_stopped = QtCore.QWaitCondition()
        self._qwc_worker_jobs_stopped = QtCore.QWaitCondition()
        self._mutex_wait_worker_DAQ = QtCore.QMutex()
        self._mutex_wait_worker_jobs = QtCore.QMutex()

    class _NoDevice:
        name = "NoDevice"

    # --------------------------------------------------------------------------
    #   attach_device
    # --------------------------------------------------------------------------

    def attach_device(self, dev):
        """Attach a reference to a user-supplied *device* class instance
        containing I/O methods. In addition, `dev` should also have the
        following members. If not, they will be injected into the `dev`
        instance for you:

            * **dev.name** (:obj:`str`) -- Short display name for the \
                device. Default: "myDevice".

            * **dev.mutex** (:class:`PyQt5.QtCore.QMutex`) -- To allow \
                for properly multithreaded device I/O operations. It will \
                be used by :class:`Worker_DAQ` and :class:`Worker_jobs`.

            * **dev.is_alive** (:obj:`bool`) -- Device is up and \
                communicatable? Default: :const:`True`.

        Args:
            dev (:obj:`object`):
                Reference to a user-supplied *device* class instance containing
                I/O methods.
        """
        if not hasattr(dev, "name"):
            dev.name = "myDevice"

        if not hasattr(dev, "mutex"):
            dev.mutex = QtCore.QMutex()

        if not hasattr(dev, "is_alive"):
            dev.is_alive = True  # Assume the device is alive from the start

        if type(self.dev) == self._NoDevice:
            self.dev = dev
        else:
            pft(
                "Device can be attached only once. Already attached to '%s'."
                % self.dev.name
            )
            sys.exit(22)

    # --------------------------------------------------------------------------
    #   Create workers
    # --------------------------------------------------------------------------

    def create_worker_DAQ(self, **kwargs):
        """Create and configure an instance of :class:`Worker_DAQ` and transfer
        it to a new :class:`PyQt5.QtCore.QThread`.

        Args:
            **kwargs
                Will be passed directly to :class:`Worker_DAQ` as initialization
                parameters, :ref:`see here <Worker_DAQ_args>`.
        """
        if type(self.dev) == self._NoDevice:
            pft(
                "Can't create worker_DAQ, because there is no device attached."
                " Did you forget to call 'attach_device()' first?"
            )
            sys.exit(99)

        self.worker_DAQ = Worker_DAQ(qdev=self, **kwargs)
        self._request_worker_DAQ_stop.connect(self.worker_DAQ._stop)
        self._request_worker_DAQ_pause.connect(self.worker_DAQ.pause)
        self._request_worker_DAQ_unpause.connect(self.worker_DAQ.unpause)

        self._thread_DAQ = QtCore.QThread()
        self._thread_DAQ.setObjectName("%s_DAQ" % self.dev.name)
        self._thread_DAQ.started.connect(self.worker_DAQ._do_work)
        self.worker_DAQ.moveToThread(self._thread_DAQ)

        if hasattr(self.worker_DAQ, "_timer"):
            self.worker_DAQ._timer.moveToThread(self._thread_DAQ)

    def create_worker_jobs(self, **kwargs):
        """Create and configure an instance of :class:`Worker_jobs` and transfer
        it to a new :class:`PyQt5.QtCore.QThread`.

        Args:
            **kwargs
                Will be passed directly to :class:`Worker_jobs` as initialization
                parameters, :ref:`see here <Worker_jobs_args>`.
        """
        if type(self.dev) == self._NoDevice:
            pft(
                "Can't create worker_jobs, because there is no device attached."
                " Did you forget to call 'attach_device()' first?"
            )
            sys.exit(99)

        self.worker_jobs = Worker_jobs(qdev=self, **kwargs)

        self._thread_jobs = QtCore.QThread()
        self._thread_jobs.setObjectName("%s_jobs" % self.dev.name)
        self._thread_jobs.started.connect(self.worker_jobs._do_work)
        self.worker_jobs.moveToThread(self._thread_jobs)

    # --------------------------------------------------------------------------
    #   Start workers
    # --------------------------------------------------------------------------

    def start(
        self,
        DAQ_priority=QtCore.QThread.InheritPriority,
        jobs_priority=QtCore.QThread.InheritPriority,
    ) -> bool:
        """Start the event loop of all of any created workers.

        Args:
            DAQ_priority (:obj:`PyQt5.QtCore.QThread.Priority`, optional):
                By default, the *worker* threads run in the operating system
                at the same thread priority as the *main/GUI* thread. You can
                change to higher priority by setting `priority` to, e.g.,
                :const:`PyQt5.QtCore.QThread.TimeCriticalPriority`. Be aware that this
                is resource heavy, so use sparingly.

                Default: :const:`PyQt5.QtCore.QThread.InheritPriority`.

            jobs_priority (:obj:`PyQt5.QtCore.QThread.Priority`, optional):
                Like `DAQ_priority`.

                Default: :const:`PyQt5.QtCore.QThread.InheritPriority`.

        Returns:
            True if successful, False otherwise.
        """
        success = True

        if self._thread_jobs is not None:
            success &= self.start_worker_jobs(priority=jobs_priority)

        if self._thread_DAQ is not None:
            success &= self.start_worker_DAQ(priority=DAQ_priority)

        return success

    def start_worker_DAQ(self, priority=QtCore.QThread.InheritPriority) -> bool:
        """Start the data acquisition with the device by starting the event loop
        of the :attr:`worker_DAQ` thread.

        Args:
            priority (:const:`PyQt5.QtCore.QThread.Priority`, optional):
                See :meth:`start` for details.

        Returns:
            True if successful, False otherwise.
        """
        if self._thread_DAQ is None:
            pft(
                "Worker_DAQ  %s: Can't start thread, because it does not exist. "
                "Did you forget to call 'create_worker_DAQ()' first?"
                % self.dev.name
            )
            sys.exit(404)  # --> leaving

        elif not self.dev.is_alive:
            dprint(
                "Worker_DAQ  %s: WARNING - Device is not alive.\n"
                % self.dev.name,
                ANSI.RED,
            )
            return False  # --> leaving

        if self.worker_DAQ.debug:
            tprint(
                "Worker_DAQ  %s: start requested..." % self.dev.name,
                ANSI.WHITE,
            )

        self._thread_DAQ.start(priority)

        # Wait for worker_DAQ to confirm having started
        locker_wait = QtCore.QMutexLocker(self._mutex_wait_worker_DAQ)
        self._qwc_worker_DAQ_started.wait(self._mutex_wait_worker_DAQ)
        locker_wait.unlock()

        if self.worker_DAQ._DAQ_trigger == DAQ_TRIGGER.SINGLE_SHOT_WAKE_UP:
            # Wait a tiny amount of extra time for the worker to have entered
            # 'self._qwc.wait(self._mutex_wait)' of method '_do_work()'.
            # Unfortunately, we can't use
            #   'QTimer.singleShot(500, confirm_has_started(self))'
            # inside the '_do_work()' routine, because it won't never resolve
            # due to the upcoming blocking 'self._qwc.wait(self._mutex_wait)'.
            # Hence, we use a blocking 'time.sleep()' here. Also note we can't
            # use 'QtCore.QCoreApplication.processEvents()' instead of
            # 'time.sleep()', because it involves a QWaitCondition and not an
            # signal event.
            time.sleep(0.05)

        if self.worker_DAQ._DAQ_trigger == DAQ_TRIGGER.CONTINUOUS:
            # We expect a 'signal_DAQ_paused' being emitted at start-up by this
            # worker. Make sure this signal gets processed as soon as possible,
            # and prior to any other subsequent actions the user might request
            # from this worker after having returned back from the user's call
            # to 'start_worker_DAQ()'.
            QtCore.QCoreApplication.processEvents()

        return True

    def start_worker_jobs(
        self, priority=QtCore.QThread.InheritPriority
    ) -> bool:
        """Start maintaining the jobs queue by starting the event loop of the
        :attr:`worker_jobs` thread.

        Args:
            priority (:obj:`PyQt5.QtCore.QThread.Priority`, optional):
                See :meth:`start` for details.

        Returns:
            True if successful, False otherwise.
        """
        if self._thread_jobs is None:
            pft(
                "Worker_jobs %s: Can't start thread because it does not exist. "
                "Did you forget to call 'create_worker_jobs()' first?"
                % self.dev.name
            )
            sys.exit(404)  # --> leaving

        elif not self.dev.is_alive:
            dprint(
                "Worker_jobs %s: WARNING - Device is not alive.\n"
                % self.dev.name,
                ANSI.RED,
            )
            return False  # --> leaving

        if self.worker_jobs.debug:
            tprint(
                "Worker_jobs %s: start requested..." % self.dev.name,
                ANSI.WHITE,
            )

        self._thread_jobs.start(priority)

        # Wait for worker_jobs to confirm having started
        locker_wait = QtCore.QMutexLocker(self._mutex_wait_worker_jobs)
        self._qwc_worker_jobs_started.wait(self._mutex_wait_worker_jobs)
        locker_wait.unlock()

        # Wait a tiny amount of extra time for the worker to have entered
        # 'self._qwc.wait(self._mutex_wait)' of method '_do_work()'.
        # Unfortunately, we can't use
        #   'QTimer.singleShot(500, confirm_has_started(self))'
        # inside the '_do_work()' routine, because it won't never resolve
        # due to the upcoming blocking 'self._qwc.wait(self._mutex_wait)'.
        # Hence, we use a blocking 'time.sleep()' here. Also note we can't
        # use 'QtCore.QCoreApplication.processEvents()' instead of
        # 'time.sleep()', because it involves a QWaitCondition and not an
        # signal event.
        time.sleep(0.05)

        return True

    # --------------------------------------------------------------------------
    #   Quit workers
    # --------------------------------------------------------------------------

    def quit(self) -> bool:
        """Stop all of any running workers and close their respective threads.

        Returns:
            True if successful, False otherwise.
        """
        return self.quit_worker_DAQ() & self.quit_worker_jobs()

    def quit_worker_DAQ(self) -> bool:
        """Stop :attr:`worker_DAQ` and close its thread.

        Returns:
            True if successful, False otherwise.
        """

        if self._thread_DAQ is None or not self.worker_DAQ._has_started:
            return True

        if self._thread_DAQ.isFinished():
            # CASE: Device has had a 'connection_lost' event during run-time,
            # which already stopped and closed the thread.
            print(
                "Closing thread %s already closed."
                % "{:.<16}".format(self._thread_DAQ.objectName())
            )
            return True

        if not self.worker_DAQ._has_stopped:
            if self.worker_DAQ.debug:
                tprint(
                    "Worker_DAQ  %s: stop requested..." % self.dev.name,
                    ANSI.WHITE,
                )

            if self.worker_DAQ._DAQ_trigger == DAQ_TRIGGER.INTERNAL_TIMER:
                # The QTimer inside the INTERNAL_TIMER '_do_work()'-routine
                # /must/ be stopped from within the worker_DAQ thread. Hence,
                # we must use a signal from out of this different thread.
                self._request_worker_DAQ_stop.emit()

            elif (
                self.worker_DAQ._DAQ_trigger == DAQ_TRIGGER.SINGLE_SHOT_WAKE_UP
            ):
                # The QWaitCondition inside the SINGLE_SHOT_WAKE_UP '_do_work()'
                # routine will likely have locked worker_DAQ. Hence, a
                # '_request_worker_DAQ_stop' signal as above might not get
                # handled by worker_DAQ when emitted from out of this thread.
                # Instead, we directly call '_stop()' from out of this different
                # thread, which is perfectly fine for SINGLE_SHOT_WAKE_UP as per
                # my design.
                self.worker_DAQ._stop()

            elif self.worker_DAQ._DAQ_trigger == DAQ_TRIGGER.CONTINUOUS:
                # We directly call '_stop()' from out of this different thread,
                # which is perfectly fine for CONTINUOUS as per my design.
                self.worker_DAQ._stop()

            # Wait for worker_DAQ to confirm having stopped
            locker_wait = QtCore.QMutexLocker(self._mutex_wait_worker_DAQ)
            self._qwc_worker_DAQ_stopped.wait(self._mutex_wait_worker_DAQ)
            locker_wait.unlock()

        self._thread_DAQ.quit()
        print(
            "Closing thread %s "
            % "{:.<16}".format(self._thread_DAQ.objectName()),
            end="",
        )
        if self._thread_DAQ.wait(2000):
            print("done.\n", end="")
            return True
        else:
            print("FAILED.\n", end="")  # pragma: no cover
            return False  # pragma: no cover

    def quit_worker_jobs(self) -> bool:
        """Stop :attr:`worker_jobs` and close its thread.

        Returns:
            True if successful, False otherwise.
        """

        if self._thread_jobs is None or not self.worker_jobs._has_started:
            return True

        if self._thread_jobs.isFinished():
            # CASE: Device has had a 'connection_lost' event during run-time,
            # which already stopped the worker and closed the thread.
            print(
                "Closing thread %s already closed."
                % "{:.<16}".format(self._thread_jobs.objectName())
            )
            return True

        if not self.worker_jobs._has_stopped:
            if self.worker_jobs.debug:
                tprint(
                    "Worker_jobs %s: stop requested..." % self.dev.name,
                    ANSI.WHITE,
                )

            # The QWaitCondition inside the SINGLE_SHOT_WAKE_UP '_do_work()'-
            # routine will likely have locked worker_DAQ. Hence, a
            # '_request_worker_DAQ_stop' signal might not get handled by
            # worker_DAQ when emitted from out of this thread. Instead,
            # we directly call '_stop()' from out of this different thread,
            # which is perfectly fine as per my design.
            self.worker_jobs._stop()

            # Wait for worker_jobs to confirm having stopped
            locker_wait = QtCore.QMutexLocker(self._mutex_wait_worker_jobs)
            self._qwc_worker_jobs_stopped.wait(self._mutex_wait_worker_jobs)
            locker_wait.unlock()

        self._thread_jobs.quit()
        print(
            "Closing thread %s "
            % "{:.<16}".format(self._thread_jobs.objectName()),
            end="",
        )
        if self._thread_jobs.wait(2000):
            print("done.\n", end="")
            return True
        else:
            print("FAILED.\n", end="")  # pragma: no cover
            return False  # pragma: no cover

    # --------------------------------------------------------------------------
    #   worker_DAQ related
    # --------------------------------------------------------------------------

    @QtCore.pyqtSlot()
    def pause_DAQ(self):
        """Only useful in mode :const:`DAQ_TRIGGER.CONTINUOUS`. Request
        :attr:`worker_DAQ` to pause and stop listening for data. After
        :attr:`worker_DAQ` has achieved the `paused` state, it will emit
        :obj:`signal_DAQ_paused()`.
        """
        if self.worker_DAQ is not None:
            self._request_worker_DAQ_pause.emit()

    @QtCore.pyqtSlot()
    def unpause_DAQ(self):
        """Only useful in mode :const:`DAQ_TRIGGER.CONTINUOUS`. Request
        :attr:`worker_DAQ` to resume listening for data. Once
        :attr:`worker_DAQ` has successfully resumed, it will emit
        :obj:`signal_DAQ_updated()` for every DAQ update.
        """
        if self.worker_DAQ is not None:
            self._request_worker_DAQ_unpause.emit()

    @QtCore.pyqtSlot()
    def wake_up_DAQ(self):
        """Only useful in mode :const:`DAQ_TRIGGER.SINGLE_SHOT_WAKE_UP`.
        Request :attr:`worker_DAQ` to wake up and perform a single update,
        i.e. run :attr:`~Worker_DAQ.DAQ_function` once. It will emit
        :obj:`signal_DAQ_updated()` after :attr:`~Worker_DAQ.DAQ_function` has
        run, either successful or not.
        """
        if self.worker_DAQ is not None:
            self.worker_DAQ.wake_up()

    # --------------------------------------------------------------------------
    #   worker_jobs related
    # --------------------------------------------------------------------------

    @QtCore.pyqtSlot()
    def send(self, instruction, pass_args=()):
        """Put a job on the :attr:`worker_jobs` queue and send out the full
        queue first-in, first-out to the device until empty. Once finished, it
        will emit :obj:`signal_jobs_updated()`.

        Args:
            instruction (:obj:`function` | *other*):
                Intended to be a reference to a device I/O method such as
                ``dev.write()``. Any arguments to be passed to the I/O method
                need to be set in the :attr:`pass_args` parameter.

                You have the freedom to be creative and put, e.g., strings
                decoding special instructions on the queue as well. Handling
                such special cases must be programmed by supplying the argument
                :obj:`jobs_function` with your own alternative
                job-processing-routine function during the initialization of
                :class:`Worker_jobs`. :ref:`See here <Worker_jobs_args>`.

            pass_args (:obj:`tuple` | *other*, optional):
                Arguments to be passed to the instruction. Must be given as a
                :obj:`tuple`, but for convenience any other type will also be
                accepted if it just concerns a single argument.

                Default: :obj:`()`.

        Example::

            qdev.send(dev.write, "toggle LED")

        where ``qdev`` is your :class:`QDeviceIO` class instance and ``dev`` is
        your *device* class instance containing I/O methods.
        """
        if self.worker_jobs is not None:
            self.worker_jobs.send(instruction, pass_args)

    @QtCore.pyqtSlot()
    def add_to_jobs_queue(self, instruction, pass_args=()):
        """Put a job on the :attr:`worker_jobs` queue.

        See :meth:`send` for details on the parameters.
        """
        if self.worker_jobs is not None:
            self.worker_jobs.add_to_queue(instruction, pass_args)

    @QtCore.pyqtSlot()
    def process_jobs_queue(self):
        """Send out the full :attr:`worker_jobs` queue first-in, first-out to
        the device until empty. Once finished, it will emit
        :obj:`signal_jobs_updated()`.
        """
        if self.worker_jobs is not None:
            self.worker_jobs.process_queue()


# ------------------------------------------------------------------------------
#   Worker_DAQ
# ------------------------------------------------------------------------------


class Worker_DAQ(QtCore.QObject):
    """This worker acquires data from the I/O device, either periodically or
    aperiodically. It does so by calling a user-supplied function, passed as
    initialization parameter :ref:`DAQ_function <arg_DAQ_function>`, containing
    device I/O operations and subsequent data processing, every time the worker
    *updates*. There are different modes of operation for this worker to perform
    an *update*. This is set by initialization parameter
    :ref:`DAQ_trigger <arg_DAQ_trigger>`.

    An instance of this worker will be created and placed inside a new thread by
    a call to :meth:`QDeviceIO.create_worker_DAQ`.

    The *Worker_DAQ* routine is robust in the following sense. It can be set
    to quit as soon as a communication error appears, or it could be set to
    allow a certain number of communication errors before it quits. The
    latter can be useful in non-critical implementations where continuity of
    the program is of more importance than preventing drops in data
    transmission. This, obviously, is a work-around for not having to tackle
    the source of the communication error, but sometimes you just need to
    struggle on. E.g., when your Arduino is out in the field and picks up
    occasional unwanted interference/ground noise that messes with your data
    transmission. See initialization parameter :obj:`critical_not_alive_count`.

    .. _`Worker_DAQ_args`:

    Args:
        qdev (:class:`QDeviceIO`):
            Reference to the parent :class:`QDeviceIO` class instance,
            automatically set when being initialized by
            :meth:`QDeviceIO.create_worker_DAQ`.

            .. _`arg_DAQ_trigger`:

        DAQ_trigger (:obj:`int`, optional):
            Mode of operation. See :class:`DAQ_TRIGGER`.

            Default: :const:`DAQ_TRIGGER.INTERNAL_TIMER`.

            .. _`arg_DAQ_function`:

        DAQ_function (:obj:`function` | :obj:`None`, optional):
            Reference to a user-supplied function containing the device
            I/O operations and subsequent data processing, to be invoked
            every DAQ update.

            Default: :obj:`None`.

            Important:
                The function must return :const:`True` when the communication
                with the device was successful, and :const:`False` otherwise.

            Warning:
                **Neither directly change the GUI, nor print to the terminal
                from out of this function.** Doing so might temporarily suspend
                the function and could mess with the timing stability of the
                worker. (You're basically undermining the reason to have
                multithreading in the first place). That could be acceptable,
                though, when you need to print debug or critical error
                information to the terminal, but be aware about the possible
                negative effects.

                Instead, connect to :meth:`QDeviceIO.signal_DAQ_updated` from
                out of the *main/GUI* thread to instigate changes to the
                terminal/GUI when needed.

            Example:
                Pseudo-code, where ``time`` and ``temperature`` are variables
                that live at a higher scope, presumably at the *main* scope
                level. The function ``dev.query_temperature()`` contains the
                device I/O operations, e.g., sending out a query over RS232 and
                collecting the device reply. In addition, the function notifies
                if the communication was successful. Hence, the return values of
                ``dev.query_temperature()`` are ``success`` as boolean and
                ``reply`` as a tuple containing a time stamp and a temperature
                reading. ::

                    def my_DAQ_function():
                        [success, reply] = dev.query_temperature()
                        if not(success):
                            print("Device IOerror")
                            return False    # Return failure

                        # Parse readings into separate variables and store them
                        try:
                            [time, temperature] = reply
                        except Exception as err:
                            print(err)
                            return False    # Return failure

                        return True         # Return success

        DAQ_interval_ms (:obj:`int`, optional):
            Only useful in mode :const:`DAQ_TRIGGER.INTERNAL_TIMER`. Desired
            data-acquisition update interval in milliseconds.

            Default: :const:`100`.

        DAQ_timer_type (:obj:`PyQt5.QtCore.Qt.TimerType`, optional):
            Only useful in mode :const:`DAQ_TRIGGER.INTERNAL_TIMER`.
            The update interval is timed to a :class:`PyQt5.QtCore.QTimer`
            running inside :class:`Worker_DAQ`. The default value
            :const:`PyQt5.QtCore.Qt.PreciseTimer` tries to ensure the best
            possible timer accuracy, usually ~1 ms granularity depending on the
            OS, but it is resource heavy so use sparingly. One can reduce the
            CPU load by setting it to less precise timer types
            :const:`PyQt5.QtCore.Qt.CoarseTimer` or
            :const:`PyQt5.QtCore.Qt.VeryCoarseTimer`.

            Default: :const:`PyQt5.QtCore.Qt.PreciseTimer`.

            .. _`arg_critical_not_alive_count`:

        critical_not_alive_count (:obj:`int`, optional):
            The worker will allow for up to a certain number of consecutive
            communication failures with the device, before hope is given up
            and a :meth:`QDeviceIO.signal_connection_lost` is emitted. Use at
            your own discretion.

            Default: :const:`1`.

        debug (:obj:`bool`, optional):
            Print debug info to the terminal? Warning: Slow! Do not leave on
            unintentionally.

            Default: :const:`False`.

        **kwargs:
            All remaining keyword arguments will be passed onto inherited class
            :class:`PyQt5.QtCore.QObject`.

    .. rubric:: Attributes:

    Attributes:
        qdev (:class:`QDeviceIO`):
            Reference to the parent :class:`QDeviceIO` class instance.

        dev (:obj:`object` | :obj:`None`):
            Reference to the user-supplied *device* class instance containing
            I/O methods, automatically set when calling
            :meth:`QDeviceIO.create_worker_DAQ`. It is a shorthand for
            :obj:`self.qdev.dev`.

        DAQ_function (:obj:`function` | :obj:`None`):
            See the similarly named :ref:`initialization parameter
            <arg_DAQ_function>`.

        critical_not_alive_count (:obj:`int`):
            See the similarly named :ref:`initialization parameter
            <arg_critical_not_alive_count>`.
    """

    def __init__(
        self,
        qdev,
        DAQ_trigger=DAQ_TRIGGER.INTERNAL_TIMER,
        DAQ_function=None,
        DAQ_interval_ms=100,
        DAQ_timer_type=QtCore.Qt.PreciseTimer,
        critical_not_alive_count=1,
        debug=False,
        **kwargs,
    ):
        super().__init__(**kwargs)  # Pass **kwargs onto QtCore.QObject()
        self.debug = debug
        self.debug_color = ANSI.CYAN

        self.qdev = qdev
        self.dev = None if qdev is None else qdev.dev

        self._DAQ_trigger = DAQ_trigger
        self.DAQ_function = DAQ_function
        self._DAQ_interval_ms = DAQ_interval_ms
        self._DAQ_timer_type = DAQ_timer_type
        self.critical_not_alive_count = critical_not_alive_count

        self._has_started = False
        self._has_stopped = False

        # Keep track of the obtained DAQ interval and DAQ rate using
        # QElapsedTimer (QET)
        self._QET_interval = QtCore.QElapsedTimer()
        self._QET_rate = QtCore.QElapsedTimer()
        # Accumulates the number of DAQ updates passed since the previous DAQ
        # rate evaluation
        self._rate_accumulator = 0

        # Members specifically for INTERNAL_TIMER
        if self._DAQ_trigger == DAQ_TRIGGER.INTERNAL_TIMER:
            self._timer = QtCore.QTimer()
            self._timer.setInterval(DAQ_interval_ms)
            self._timer.setTimerType(DAQ_timer_type)
            self._timer.timeout.connect(self._perform_DAQ)

        # Members specifically for SINGLE_SHOT_WAKE_UP
        elif self._DAQ_trigger == DAQ_TRIGGER.SINGLE_SHOT_WAKE_UP:
            self._running = True
            self._qwc = QtCore.QWaitCondition()
            self._mutex_wait = QtCore.QMutex()

        # Members specifically for CONTINUOUS
        # Note: At start-up, the worker will directly go into a paused state
        # and trigger a 'signal_DAQ_paused' PyQt signal
        elif self._DAQ_trigger == DAQ_TRIGGER.CONTINUOUS:
            self._running = True
            self._pause = None  # Will be set at init of '_do_work()' when 'start_worker_DAQ()' is called
            self._paused = None  # Will be set at init of '_do_work()' when 'start_worker_DAQ()' is called

        if self.debug:
            tprint(
                "Worker_DAQ  %s: init @ thread %s"
                % (self.dev.name, _cur_thread_name()),
                self.debug_color,
            )

    @_coverage_resolve_trace
    @QtCore.pyqtSlot()
    def _do_work(self):
        # fmt: off
        # Uncomment block to enable Visual Studio Code debugger to have access
        # to this thread. DO NOT LEAVE BLOCK UNCOMMENTED: Running it outside of
        # the debugger causes crashes.
        """
        if self.debug:
            import pydevd
            pydevd.settrace(suspend=False)
        """
        # fmt: on

        init = True

        def confirm_has_started(self):
            # Wait a tiny amount of extra time for QDeviceIO to have entered
            # 'self._qwc_worker_###_started.wait(self._mutex_wait_worker_###)'
            # of method 'start_worker_###()'.
            time.sleep(0.05)

            if self.debug:
                tprint(
                    "Worker_DAQ  %s: has started" % self.dev.name,
                    self.debug_color,
                )

            # Send confirmation
            self.qdev._qwc_worker_DAQ_started.wakeAll()
            self._has_started = True

        if self.debug:
            tprint(
                "Worker_DAQ  %s: starting @ thread %s"
                % (self.dev.name, _cur_thread_name()),
                self.debug_color,
            )

        # INTERNAL_TIMER
        if self._DAQ_trigger == DAQ_TRIGGER.INTERNAL_TIMER:
            self._timer.start()
            confirm_has_started(self)

        # SINGLE_SHOT_WAKE_UP
        elif self._DAQ_trigger == DAQ_TRIGGER.SINGLE_SHOT_WAKE_UP:
            locker_wait = QtCore.QMutexLocker(self._mutex_wait)
            locker_wait.unlock()

            while self._running:
                locker_wait.relock()

                if self.debug:
                    tprint(
                        "Worker_DAQ  %s: waiting for wake-up trigger"
                        % self.dev.name,
                        self.debug_color,
                    )

                if init:
                    confirm_has_started(self)
                    init = False

                self._qwc.wait(self._mutex_wait)

                if self.debug:
                    tprint(
                        "Worker_DAQ  %s: has woken up" % self.dev.name,
                        self.debug_color,
                    )

                # Needed check to prevent _perform_DAQ() at final wake up
                # when _stop() has been called
                if self._running:
                    self._perform_DAQ()

                locker_wait.unlock()

            if self.debug:
                tprint(
                    "Worker_DAQ  %s: has stopped" % self.dev.name,
                    self.debug_color,
                )

            # Wait a tiny amount for the other thread to have entered the
            # QWaitCondition lock, before giving a wakingAll().
            QtCore.QTimer.singleShot(
                100, self.qdev._qwc_worker_DAQ_stopped.wakeAll,
            )
            self._has_stopped = True

        # CONTINUOUS
        elif self._DAQ_trigger == DAQ_TRIGGER.CONTINUOUS:
            while self._running:
                QtCore.QCoreApplication.processEvents()  # Essential to fire and process signals

                if init:
                    self._pause = True
                    self._paused = True

                    if self.debug:
                        tprint(
                            "Worker_DAQ  %s: starting up paused"
                            % self.dev.name,
                            self.debug_color,
                        )

                    self.qdev.signal_DAQ_paused.emit()

                    confirm_has_started(self)
                    init = False

                if self._pause:  # == True
                    if self._pause != self._paused:
                        if self.debug and not init:
                            tprint(
                                "Worker_DAQ  %s: has paused" % self.dev.name,
                                self.debug_color,
                            )
                        self.qdev.signal_DAQ_paused.emit()
                        self._paused = True

                    time.sleep(0.01)  # Do not hog the CPU while paused

                else:  # == False
                    if self._pause != self._paused:
                        if self.debug:
                            tprint(
                                "Worker_DAQ  %s: has unpaused" % self.dev.name,
                                self.debug_color,
                            )
                        self._paused = False

                    self._perform_DAQ()

            if self.debug:
                tprint(
                    "Worker_DAQ  %s: has stopped" % self.dev.name,
                    self.debug_color,
                )

            # Wait a tiny amount for 'create_worker_DAQ()', which is running
            # in a different thread than this one, to have entered the
            # QWaitCondition lock, before giving a wakingAll().
            QtCore.QTimer.singleShot(
                100, self.qdev._qwc_worker_DAQ_stopped.wakeAll,
            )
            self._has_stopped = True

    @_coverage_resolve_trace
    @QtCore.pyqtSlot()
    def _perform_DAQ(self):
        locker = QtCore.QMutexLocker(self.dev.mutex)
        self.qdev.update_counter_DAQ += 1

        if self.debug:
            tprint(
                "Worker_DAQ  %s: lock   # %i"
                % (self.dev.name, self.qdev.update_counter_DAQ),
                self.debug_color,
            )

        # Keep track of the obtained DAQ interval and DAQ rate
        if not self._QET_interval.isValid():
            self._QET_interval.start()
            self._QET_rate.start()
        else:
            # Obtained DAQ interval
            self.qdev.obtained_DAQ_interval_ms = self._QET_interval.restart()

            # Obtained DAQ rate
            self._rate_accumulator += 1
            dT = self._QET_rate.elapsed()

            if dT >= 1000:  # Evaluate every N elapsed milliseconds. Hard-coded.
                self._QET_rate.restart()
                try:
                    self.qdev.obtained_DAQ_rate_Hz = (
                        self._rate_accumulator / dT * 1e3
                    )
                except ZeroDivisionError:  # pragma: no cover
                    self.qdev.obtained_DAQ_rate_Hz = np.nan  # pragma: no cover

                self._rate_accumulator = 0

        # ----------------------------------
        #   User-supplied DAQ function
        # ----------------------------------

        if self.DAQ_function is not None:
            try:
                success = self.DAQ_function()
            except Exception as err:  # pylint: disable=broad-except
                pft(err)
                dprint(
                    "@ Worker_DAQ %s\n" % self.dev.name, ANSI.RED,
                )
            else:
                if success:
                    # Did return True, hence was successfull
                    # --> Reset the 'not alive' counter
                    self.qdev.not_alive_counter_DAQ = 0
                else:
                    # Did return False, hence was unsuccessfull
                    self.qdev.not_alive_counter_DAQ += 1

        # ----------------------------------
        #   End user-supplied DAQ function
        # ----------------------------------

        if self.debug:
            tprint(
                "Worker_DAQ  %s: unlock # %i"
                % (self.dev.name, self.qdev.update_counter_DAQ),
                self.debug_color,
            )

        locker.unlock()

        # Check the not alive counter
        if self.qdev.not_alive_counter_DAQ >= self.critical_not_alive_count:
            dprint(
                "Worker_DAQ  %s: Lost connection to device." % self.dev.name,
                ANSI.RED,
            )
            self.dev.is_alive = False
            self._stop()
            self.qdev.signal_connection_lost.emit()
            return

        self.qdev.signal_DAQ_updated.emit()

    @QtCore.pyqtSlot()
    def _stop(self):
        """Stop the worker to prepare for quitting the worker thread.
        """
        if self.debug:
            tprint("Worker_DAQ  %s: stopping" % self.dev.name, self.debug_color)

        if self._DAQ_trigger == DAQ_TRIGGER.INTERNAL_TIMER:
            # NOTE: The timer /must/ be stopped from the worker_DAQ thread!
            self._timer.stop()

            if self.debug:
                tprint(
                    "Worker_DAQ  %s: has stopped" % self.dev.name,
                    self.debug_color,
                )

            # Wait a tiny amount for the other thread to have entered the
            # QWaitCondition lock, before giving a wakingAll().
            QtCore.QTimer.singleShot(
                100, self.qdev._qwc_worker_DAQ_stopped.wakeAll,
            )
            self._has_stopped = True

        elif self._DAQ_trigger == DAQ_TRIGGER.SINGLE_SHOT_WAKE_UP:
            self._running = False
            self._qwc.wakeAll()  # Wake up for the final time

        elif self._DAQ_trigger == DAQ_TRIGGER.CONTINUOUS:
            self._running = False

    # ----------------------------------------------------------------------
    #   pause / unpause
    # ----------------------------------------------------------------------

    @QtCore.pyqtSlot()
    def pause(self):
        """Only useful in mode :const:`DAQ_TRIGGER.CONTINUOUS`. Pause
        the worker to stop listening for data. After :attr:`worker_DAQ` has
        achieved the `paused` state, it will emit :obj:`signal_DAQ_paused()`.

        This method should not be called from another thread. Connect this slot
        to a signal instead.
        """
        if self._DAQ_trigger == DAQ_TRIGGER.CONTINUOUS:
            if self.debug:
                tprint(
                    "Worker_DAQ  %s: pause requested..." % self.dev.name,
                    ANSI.WHITE,
                )

            # The possible undefined behavior of changing variable '_pause'
            # from out of another thread gets handled acceptably correct in
            # '_do_work()' as per my design.
            self._pause = True

    @QtCore.pyqtSlot()
    def unpause(self):
        """Only useful in mode :const:`DAQ_TRIGGER.CONTINUOUS`. Unpause
        the worker to resume listening for data. Once :attr:`worker_DAQ` has
        successfully resumed, it will emit :obj:`signal_DAQ_updated()` for every
        DAQ update.

        This method should not be called from another thread. Connect this slot
        to a signal instead.
        """
        if self._DAQ_trigger == DAQ_TRIGGER.CONTINUOUS:
            if self.debug:
                tprint(
                    "Worker_DAQ  %s: unpause requested..." % self.dev.name,
                    ANSI.WHITE,
                )

            # The possible undefined behavior of changing variable '_pause'
            # from out of another thread gets handled acceptably correct in
            # '_do_work()' as per my design.
            self._pause = False

    # ----------------------------------------------------------------------
    #   wake_up
    # ----------------------------------------------------------------------

    @QtCore.pyqtSlot()
    def wake_up(self):
        """Only useful in mode :const:`DAQ_TRIGGER.SINGLE_SHOT_WAKE_UP`. See the
        description at :meth:`QDeviceIO.wake_up_DAQ`.

        This method can be called from another thread.
        """
        if self._DAQ_trigger == DAQ_TRIGGER.SINGLE_SHOT_WAKE_UP:
            if self.debug:
                tprint(
                    "Worker_DAQ  %s: wake-up requested..." % self.dev.name,
                    ANSI.WHITE,
                )

            self._qwc.wakeAll()


# ------------------------------------------------------------------------------
#   Worker_jobs
# ------------------------------------------------------------------------------


class Worker_jobs(QtCore.QObject):
    """This worker maintains a thread-safe queue where desired device I/O
    operations, called *jobs*, can be put onto. The worker will send out the
    operations to the device, first-in, first-out (FIFO), until the queue is
    empty again. The manner in which each job gets handled is explained by
    initialization parameter :ref:`jobs_function <arg_jobs_function>`.

    An instance of this worker will be created and placed inside a new thread by
    a call to :meth:`QDeviceIO.create_worker_jobs`.

    This worker uses the :class:`PyQt5.QtCore.QWaitCondition` mechanism. Hence,
    it will only send out all pending jobs on the queue, whenever the thread is
    woken up by a call to :meth:`Worker_jobs.process_queue()`. When it has
    emptied the queue, the thread will go back to sleep again.

    .. _`Worker_jobs_args`:

    Args:
        qdev (:class:`QDeviceIO`):
            Reference to the parent :class:`QDeviceIO` class instance,
            automatically set when being initialized by
            :meth:`QDeviceIO.create_worker_jobs`.

            .. _`arg_jobs_function`:

        jobs_function (:obj:`function` | :obj:`None`, optional): Routine to be
            performed per job.

            Default: :obj:`None`.

            When omitted and, hence, left set to the default value :obj:`None`,
            it will perform the default job handling routine, which goes as
            follows:

                ``func`` and ``args`` will be retrieved from the jobs
                queue and their combination ``func(*args)`` will get executed.
                Respectively, *func* and *args* correspond to *instruction* and
                *pass_args* of methods :meth:`send` and :meth:`add_to_queue`.

            The default is sufficient when ``func`` corresponds to an
            I/O operation that is an one-way send, i.e. a write operation
            with optionally passed arguments, but without a reply from the
            device.

            Alternatively, you can pass it a reference to a user-supplied
            function performing an alternative job handling routine. This
            allows you to get creative and put, e.g., special string messages on
            the queue that decode into, e.g.,

            - multiple write operations to be executed as one block,
            - query operations whose return values can be acted upon accordingly,
            - extra data processing in between I/O operations.

            The function you supply must take two arguments, where the first
            argument is to be ``func`` and the second argument is to be
            ``args`` of type :obj:`tuple`. Both ``func`` and ``args`` will be
            retrieved from the jobs queue and passed onto your supplied
            function.

            Warning:
                **Neither directly change the GUI, nor print to the terminal
                from out of this function.** Doing so might temporarily suspend
                the function and could mess with the timing stability of the
                worker. (You're basically undermining the reason to have
                multithreading in the first place). That could be acceptable,
                though, when you need to print debug or critical error
                information to the terminal, but be aware about this warning.

                Instead, connect to :meth:`QDeviceIO.signal_jobs_updated` from
                out of the *main/GUI* thread to instigate changes to the
                terminal/GUI when needed.

            Example::

                def my_jobs_function(func, args):
                    if func == "query_id?":
                        # Query the device for its identity string
                        [success, ans_str] = dev.query("id?")
                        # And store the reply 'ans_str' in another variable
                        # at a higher scope or do stuff with it here.
                    else:
                        # Default job handling where, e.g.
                        # func = dev.write
                        # args = ("toggle LED",)
                        func(*args)

        debug (:obj:`bool`, optional):
            Print debug info to the terminal? Warning: Slow! Do not leave on
            unintentionally.

            Default: :const:`False`.

        **kwargs:
            All remaining keyword arguments will be passed onto inherited class
            :class:`PyQt5.QtCore.QObject`.

    .. rubric:: Attributes:

    Attributes:
        qdev (:class:`QDeviceIO`):
            Reference to the parent :class:`QDeviceIO` class instance.

        dev (:obj:`object` | :obj:`None`):
            Reference to the user-supplied *device* class instance containing
            I/O methods, automatically set when calling
            :meth:`QDeviceIO.create_worker_jobs`. It is a shorthand for
            :obj:`self.qdev.dev`.

        jobs_function (:obj:`function` | :obj:`None`):
            See the similarly named :ref:`initialization parameter
            <arg_jobs_function>`.
    """

    def __init__(
        self, qdev, jobs_function=None, debug=False, **kwargs,
    ):
        super().__init__(**kwargs)  # Pass **kwargs onto QtCore.QObject()
        self.debug = debug
        self.debug_color = ANSI.YELLOW

        self.qdev = qdev
        self.dev = None if qdev is None else qdev.dev

        self.jobs_function = jobs_function
        self._has_started = False
        self._has_stopped = False

        self._running = True
        self._qwc = QtCore.QWaitCondition()
        self._mutex_wait = QtCore.QMutex()

        # Use a 'sentinel' value to signal the start and end of the queue
        # to ensure proper multithreaded operation.
        self._sentinel = None
        self._queue = queue.Queue()
        self._queue.put(self._sentinel)

        if self.debug:
            tprint(
                "Worker_jobs %s: init @ thread %s"
                % (self.dev.name, _cur_thread_name()),
                self.debug_color,
            )

    @_coverage_resolve_trace
    @QtCore.pyqtSlot()
    def _do_work(self):
        # fmt: off
        # Uncomment block to enable Visual Studio Code debugger to have access
        # to this thread. DO NOT LEAVE BLOCK UNCOMMENTED: Running it outside of
        # the debugger causes crashes.
        """
        if self.debug:
            import pydevd
            pydevd.settrace(suspend=False)
        """
        # fmt: on

        init = True

        def confirm_has_started(self):
            # Wait a tiny amount of extra time for QDeviceIO to have entered
            # 'self._qwc_worker_###_started.wait(self._mutex_wait_worker_###)'
            # of method 'start_worker_###()'.
            time.sleep(0.05)

            if self.debug:
                tprint(
                    "Worker_jobs %s: has started" % self.dev.name,
                    self.debug_color,
                )

            # Send confirmation
            self.qdev._qwc_worker_jobs_started.wakeAll()
            self._has_started = True

        if self.debug:
            tprint(
                "Worker_jobs %s: starting @ thread %s"
                % (self.dev.name, _cur_thread_name()),
                self.debug_color,
            )

        locker_wait = QtCore.QMutexLocker(self._mutex_wait)
        locker_wait.unlock()

        while self._running:
            locker_wait.relock()

            if self.debug:
                tprint(
                    "Worker_jobs %s: waiting for wake-up trigger"
                    % self.dev.name,
                    self.debug_color,
                )

            if init:
                confirm_has_started(self)
                init = False

            self._qwc.wait(self._mutex_wait)

            if self.debug:
                tprint(
                    "Worker_jobs %s: has woken up" % self.dev.name,
                    self.debug_color,
                )

            # Needed check to prevent _perform_jobs() at final wake up
            # when _stop() has been called
            if self._running:
                self._perform_jobs()

            locker_wait.unlock()

        if self.debug:
            tprint(
                "Worker_jobs %s: has stopped" % self.dev.name, self.debug_color,
            )

        # Wait a tiny amount for the other thread to have entered the
        # QWaitCondition lock, before giving a wakingAll().
        QtCore.QTimer.singleShot(
            100, self.qdev._qwc_worker_jobs_stopped.wakeAll,
        )
        self._has_stopped = True

    @_coverage_resolve_trace
    @QtCore.pyqtSlot()
    def _perform_jobs(self):
        locker = QtCore.QMutexLocker(self.dev.mutex)
        self.qdev.update_counter_jobs += 1

        if self.debug:
            tprint(
                "Worker_jobs %s: lock   # %i"
                % (self.dev.name, self.qdev.update_counter_jobs),
                self.debug_color,
            )

        # Process all jobs until the queue is empty. We must iterate 2 times
        # because we use a sentinel in a FIFO queue. First iter removes the old
        # sentinel. Second iter processes the remaining queue items and will put
        # back a new sentinel again.
        for _i in range(2):
            for job in iter(self._queue.get_nowait, self._sentinel):
                func = job[0]
                args = job[1:]

                if self.debug:
                    if type(func) == str:
                        tprint(
                            "Worker_jobs %s: %s %s"
                            % (self.dev.name, func, args),
                            self.debug_color,
                        )
                    else:
                        tprint(
                            "Worker_jobs %s: %s %s"
                            % (self.dev.name, func.__name__, args),
                            self.debug_color,
                        )

                if self.jobs_function is None:
                    # Default job processing:
                    # Send I/O operation to the device
                    try:
                        func(*args)
                    except Exception as err:  # pylint: disable=broad-except
                        pft(err)
                        dprint(
                            "@ Worker_jobs %s\n" % self.dev.name, ANSI.RED,
                        )
                else:
                    # User-supplied job processing
                    self.jobs_function(func, args)

            # Put sentinel back in
            self._queue.put(self._sentinel)

        if self.debug:
            tprint(
                "Worker_jobs %s: unlock # %i"
                % (self.dev.name, self.qdev.update_counter_jobs),
                self.debug_color,
            )

        locker.unlock()
        self.qdev.signal_jobs_updated.emit()

    @QtCore.pyqtSlot()
    def _stop(self):
        """Stop the worker to prepare for quitting the worker thread
        """
        if self.debug:
            tprint(
                "Worker_jobs %s: stopping" % self.dev.name, self.debug_color,
            )

        self._running = False
        self._qwc.wakeAll()  # Wake up for the final time

    # ----------------------------------------------------------------------
    #   send
    # ----------------------------------------------------------------------

    def send(self, instruction, pass_args=()):
        """See the description at :meth:`QDeviceIO.send`.

        This method can be called from another thread.
        """
        self.add_to_queue(instruction, pass_args)
        self.process_queue()

    # ----------------------------------------------------------------------
    #   add_to_queue
    # ----------------------------------------------------------------------

    def add_to_queue(self, instruction, pass_args=()):
        """See the description at :meth:`QDeviceIO.add_to_jobs_queue`.

        This method can be called from another thread.
        """
        if type(pass_args) is not tuple:
            pass_args = (pass_args,)
        self._queue.put((instruction, *pass_args))

    # ----------------------------------------------------------------------
    #   process_queue
    # ----------------------------------------------------------------------

    def process_queue(self):
        """See the description at :meth:`QDeviceIO.process_jobs_queue`.

        This method can be called from another thread.
        """
        if self.debug:
            tprint(
                "Worker_jobs %s: wake-up requested..." % self.dev.name,
                ANSI.WHITE,
            )

        self._qwc.wakeAll()
