#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Class ChartHistory manages two data arrays `x` and `y` that serve as history
buffers for displaying e.g. a strip chart or Lissajous curve. It is
thread-safe and relies on the PyQtGraph library for fast plotting to screen.
Intended multithreaded operation: One thread does the data acquisition and
pushes new data points into the history buffers by calling `add_new_reading(s)`,
and another thread performs the GUI refresh and redraws the data behind the plot
by calling `update_curve`.

New readings are stored in an history array of fixed size. Newest readings
are placed at the end of the array. Array full? -> FIFO.

Class:
    ChartHistory(chart_history_length, plot_data_item):
        Args:
            chart_history_length:
                Number of data points to store in each history buffer.
            plot_data_item:
                Instance of `pyqtgraph.PlotDataItem` to plot out the buffered
                data to.

        Methods:
            apply_downsampling(...):
                Calls the downsampling routines of PyQtGraph, but in the future
                I will provide my own routines for downsampling here.
            add_new_reading(...):
                Add single data point (x, y) to the history buffers.
            add_new_readings(...):
                Add lists of data points (list_x, list_y) to the history
                buffers.
            update_curve():
                Update the data behind the curve and redraw.
            clear():
                Clear buffers.

        Important member:
            x_axis_divisor:
                If the x-data is time, you can use this divisor value to
                transform the x-axis units from e.g. milliseconds to seconds or
                minutes.
            y_axis_divisor:
                Same functionality as x_axis_divisor
"""
__author__      = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__         = "https://github.com/Dennis-van-Gils/DvG_PyQt_misc"
__date__        = "14-09-2018"
__version__     = "1.0.0"

import collections

import numpy as np
from PyQt5 import QtCore
import pyqtgraph as pg

class ChartHistory(object):
    def __init__(self,
                 chart_history_length,
                 plot_data_item: pg.PlotDataItem=None):
        self.chart_history_length = chart_history_length
        self.curve = plot_data_item   # Instance of [pyqtgraph.PlotDataItem]
        self.mutex = QtCore.QMutex()  # For the case of multithreaded access

        # If the x-data is time, you can use this divisor value to transform
        # the x-axis units from e.g. milliseconds to seconds or minutes.
        self.x_axis_divisor = 1
        self.y_axis_divisor = 1

        self._x = collections.deque(maxlen=chart_history_length)
        self._y = collections.deque(maxlen=chart_history_length)
        self._x_snapshot = [0]
        self._y_snapshot = [0]

        # Performance boost: Do not plot data outside of visible range
        self.curve.clipToView = True

        # Default to no downsampling
        self.curve.setDownsampling(ds=1, auto=False, method='mean')

    def apply_downsampling(self, do_apply=True, ds=4):
        if do_apply:
            # Speed up plotting, needed for keeping the GUI responsive when
            # using large datasets
            self.curve.setDownsampling(ds=ds, auto=False, method='mean')
        else:
            self.curve.setDownsampling(ds=1, auto=False, method='mean')

    def add_new_reading(self, x, y):
        locker = QtCore.QMutexLocker(self.mutex)
        self._x.append(x)
        self._y.append(y)
        locker.unlock()

    def add_new_readings(self, x_list, y_list):
        locker = QtCore.QMutexLocker(self.mutex)
        self._x.extend(x_list)
        self._y.extend(y_list)
        locker.unlock()

    def update_curve(self):
        """Creates a snapshot of the buffered data, which is a fast operation,
        followed by updating the data behind the curve and redrawing it, which
        is a slow operation. Hence, the use of a snapshot creation, which is
        locked my a mutex, followed by a the mutex unlocked redrawing.
        """

        # First create a snapshot of the buffered data. Fast.
        locker = QtCore.QMutexLocker(self.mutex)
        self._x_snapshot = np.copy(self._x)
        self._y_snapshot = np.copy(self._y)
        #print("numel x: %d, numel y: %d" %
        #      (self._x_snapshot.size, self._y_snapshot.size))
        locker.unlock()

        # Now update the data behind the curve and redraw the curve. Slow
        if self.curve is not None:
            if ((len(self._x_snapshot) == 0) or
                (np.alltrue(np.isnan(self._y_snapshot)))):
                self.curve.setData([0], [0])
            else:
                self.curve.setData((self._x_snapshot - self._x_snapshot[-1])
                                   / float(self.x_axis_divisor),
                                   self._y_snapshot
                                   / float(self.y_axis_divisor))

    def clear(self):
        locker = QtCore.QMutexLocker(self.mutex)
        self._x.clear()
        self._y.clear()
        locker.unlock()
