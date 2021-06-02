#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""*PyQtGraph library providing thread-safe plot curves with underlying (ring)
buffers:* ``HistoryChartCurve``, ``BufferedPlotCurve`` & ``PlotCurve``.

- Github: https://github.com/Dennis-van-Gils/python-dvg-pyqtgraph-threadsafe
- PyPI: https://pypi.org/project/dvg-pyqtgraph-threadsafe

Installation::

    pip install dvg-pyqtgraph-threadsafe

Classes ``HistoryChartCurve``, ``BufferedPlotCurve`` & ``PlotCurve`` wrap around
a ``pyqtgraph.PlotDataItem`` instance, called a *curve* for convenience. Data
can be safely appended or set from out of any thread.

The (x, y)-curve data is buffered internally to the class, relying on either a
circular/ring buffer or a regular array buffer:

    HistoryChartCurve
        Ring buffer. The plotted x-data will be shifted such that the
        right-side is always set to 0. I.e., when `x` denotes time, the data is
        plotted backwards in time, hence the name *history* chart. The most
        recent data is on the right-side of the ring buffer.

    BufferedPlotCurve
        Ring buffer. Data will be plotted as is. Can also act as a Lissajous
        figure.

    PlotCurve
        Regular array buffer. Data will be plotted as is.

Usage:

    .. code-block:: python

        import sys
        from PyQt5 import QtWidgets
        import pyqtgraph as pg
        from dvg_pyqtgraph_threadsafe import HistoryChartCurve

        class MainWindow(QtWidgets.QWidget):
            def __init__(self, parent=None, **kwargs):
                super().__init__(parent, **kwargs)

                self.gw = pg.GraphicsLayoutWidget()
                self.plot_1 = self.gw.addPlot()

                # Create a HistoryChartCurve and have it wrap around a new PlotDataItem
                # as set by argument `linked_curve`.
                self.tscurve_1 = HistoryChartCurve(
                    capacity=5,
                    linked_curve=self.plot_1.plot(pen=pg.mkPen('r')),
                )

                grid = QtWidgets.QGridLayout(self)
                grid.addWidget(self.gw)

        app = QtWidgets.QApplication(sys.argv)
        window = MainWindow()

        # The following line could have been executed from inside of another thread:
        window.tscurve_1.extendData([1, 2, 3, 4, 5], [10, 20, 30, 20, 10])

        # Draw the curve from out of the main thread
        window.tscurve_1.update()

        window.show()
        sys.exit(app.exec_())
"""
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/python-dvg-pyqtgraph-threadsafe"
__date__ = "10-05-2021"
__version__ = "3.1.0"

from functools import partial
from typing import Union, Tuple, List, Optional

try:
    from typing import TypedDict
except:  # pylint: disable=bare-except
    from typing_extensions import TypedDict

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets as QtWid
import pyqtgraph as pg

from dvg_ringbuffer import RingBuffer


class ThreadSafeCurve(object):
    """Provides the base class for a thread-safe plot *curve* to which
    (x, y)-data can be safely appended or set from out of any thread. It
    will wrap around the passed argument ``linked_curve`` of type
    ``pyqtgraph.PlotDataItem`` and will manage the (x, y)-data buffers
    underlying the curve.

    Intended multi-threaded operation: One or more threads push new data
    into the ``ThreadSafeCurve``-buffers. Another thread performs the GUI
    refresh by calling ``update()`` which will redraw the curve according
    to the current buffer contents.

    Args:
        capacity (``int``, optional):
            When an integer is supplied it defines the maximum number op points
            each of the x-data and y-data buffers can store. The x-data buffer
            and the y-data buffer are each a ring buffer. New readings are
            placed at the end (right-side) of the buffer, pushing out the oldest
            readings when the buffer has reached its maximum capacity (FIFO).
            Use methods ``appendData()`` and ``extendData()`` to push in new
            data.

            When ``None`` is supplied the x-data and y-data buffers are each a
            regular array buffer of undefined length. Use method ``setData()``
            to set the data.

        linked_curve (``pyqtgraph.PlotDataItem``):
            Instance of ``pyqtgraph.PlotDataItem`` to plot the buffered
            data out into.

        shift_right_x_to_zero (``bool``, optional):
            When plotting, should the x-data be shifted such that the
            right-side is always set to 0? Useful for history charts.

            Default: False

        use_ringbuffer (``bool``, deprecated):
            Deprecated since v3.1.0. Defined for backwards compatibility.
            Simply supply a value for ``capacity`` to enable use of a ring
            buffer.

    Attributes:
        x_axis_divisor (``float``):
            The x-data in the buffer will be divided by this factor when the
            plot curve is drawn. Useful to, e.g., transform the x-axis units
            from milliseconds to seconds or minutes.

            Default: 1

        y_axis_divisor (``float``):
            Same functionality as ``x_axis_divisor``.

            Default: 1
    """

    def __init__(
        self,
        capacity: Optional[int],
        linked_curve: pg.PlotDataItem,
        shift_right_x_to_zero: bool = False,
        use_ringbuffer=None,  # Deprecated arg for backwards compatibility # pylint: disable=unused-argument
    ):
        self.capacity = capacity
        self.curve = linked_curve
        self.opts = self.curve.opts  # Use for read-only

        self._shift_right_x_to_zero = shift_right_x_to_zero
        self._use_ringbuffer = capacity is not None
        self._mutex = QtCore.QMutex()  # To allow proper multithreading

        self.x_axis_divisor = 1
        self.y_axis_divisor = 1

        if self._use_ringbuffer:
            self._buffer_x = RingBuffer(capacity=capacity)
            self._buffer_y = RingBuffer(capacity=capacity)
        else:
            self._buffer_x = np.array([])
            self._buffer_y = np.array([])

        self._snapshot_x = np.array([])
        self._snapshot_y = np.array([])

    def appendData(self, x, y):
        """Append a single (x, y)-data point to the ring buffer.
        """
        if self._use_ringbuffer:
            locker = QtCore.QMutexLocker(self._mutex)
            self._buffer_x.append(x)
            self._buffer_y.append(y)
            locker.unlock()

    def extendData(self, x_list, y_list):
        """Extend the ring buffer with a list of (x, y)-data points.
        """
        if self._use_ringbuffer:
            locker = QtCore.QMutexLocker(self._mutex)
            self._buffer_x.extend(x_list)
            self._buffer_y.extend(y_list)
            locker.unlock()

    def setData(self, x_list, y_list):
        """Set the (x, y)-data of the regular array buffer.
        """
        if not self._use_ringbuffer:
            locker = QtCore.QMutexLocker(self._mutex)
            self._buffer_x = x_list
            self._buffer_y = y_list
            locker.unlock()

    def update(self, create_snapshot: bool = True):
        """Update the data behind the curve by creating a snapshot of the
        current contents of the buffer, and redraw the curve on screen.

        Args:
            create_snapshot (``bool``):
                You can suppress updating the data behind the curve by setting
                this parameter to False. The curve will then only be redrawn
                based on the old data. This is useful when the plot is paused.

                Default: True
        """

        # Create a snapshot of the currently buffered data. Fast operation.
        if create_snapshot:
            locker = QtCore.QMutexLocker(self._mutex)
            self._snapshot_x = np.copy(self._buffer_x)
            self._snapshot_y = np.copy(self._buffer_y)
            # print("numel x: %d, numel y: %d" %
            #      (self._snapshot_x.size, self._snapshot_y.size))
            locker.unlock()

        # Now update the data behind the curve and redraw it on screen.
        # Note: .setData() will internally emit a PyQt signal to redraw the
        # curve, once it has updated its data members. That's why .setData()
        # returns almost immediately, but the curve still has to get redrawn by
        # the Qt event engine, which will happen automatically, eventually.
        if len(self._snapshot_x) == 0:
            self.curve.setData([], [])
        else:
            x_0 = self._snapshot_x[-1] if self._shift_right_x_to_zero else 0
            x = (self._snapshot_x - x_0) / float(self.x_axis_divisor)
            y = self._snapshot_y / float(self.y_axis_divisor)
            # self.curve.setData(x,y)  # No! Read below.

            # PyQt5 >= 5.12.3 causes a bug in PyQtGraph where a curve won't
            # render if it contains NaNs (but only in the case when OpenGL is
            # disabled). See for more information:
            # https://github.com/pyqtgraph/pyqtgraph/pull/1287/commits/5d58ec0a1b59f402526e2533977344d043b306d8
            #
            # My approach is slightly different:
            # NaN values are allowed in the source x and y arrays, but we need
            # to filter them such that the drawn curve is displayed as
            # *fragmented* whenever NaN is encountered. The parameter `connect`
            # will help us out here.
            # NOTE: When OpenGL is used to paint the curve by setting
            #   pg.setConfigOptions(useOpenGL=True)
            #   pg.setConfigOptions(enableExperimental=True)
            # the `connect` argument will get ignored and the curve fragments
            # are connected together into a continuous curve, linearly
            # interpolating the gaps. Seems to be little I can do about that,
            # apart from modifying the pyqtgraph source-code in
            # `pyqtgraph.plotCurveItem.paintGL()`.
            #
            # UPDATE 07-08-2020:
            # Using parameter `connect` as used below will cause:
            #   ValueError: could not broadcast input array from shape ('N') into shape ('< N')
            #   --> arr[1:-1]['c'] = connect
            #   in ``pyqtgraph.functinos.arrayToQPath()``
            # This happens when ClipToView is enabled and the curve data extends
            # past the viewbox limits, when not using OpenGL.
            # We simply comment out those lines. This results in 100% working
            # code again, though the curve is no longer shown fragmented but
            # continuous (with linear interpolation) at each NaN value. That's
            # okay.

            finite = np.logical_and(np.isfinite(x), np.isfinite(y))
            # connect = np.logical_and(finite, np.roll(finite, -1))
            x_finite = x[finite]
            y_finite = y[finite]
            # connect = connect[finite]

            self.curve.setData(x_finite, y_finite)  # , connect=connect)

    @QtCore.pyqtSlot()
    def clear(self):
        """Clear the contents of the curve and redraw.
        """
        locker = QtCore.QMutexLocker(self._mutex)
        if self._use_ringbuffer:
            self._buffer_x.clear()
            self._buffer_y.clear()
        else:
            self._buffer_x = np.array([])
            self._buffer_y = np.array([])
        locker.unlock()

        self.update()

    def name(self):
        """Get the name of the curve.
        """
        return self.curve.name()

    def isVisible(self) -> bool:
        return self.curve.isVisible()

    def setVisible(self, state: bool = True):
        self.curve.setVisible(state)

    def setDownsampling(self, *args, **kwargs):
        """All arguments will be passed onto method
        ``pyqtgraph.PlotDataItem.setDownsampling()`` of the underlying curve.
        """
        self.curve.setDownsampling(*args, **kwargs)

    @property
    def size(self) -> Tuple[int, int]:
        """Number of elements currently contained in the underlying (x, y)-
        buffers of the curve. Note that this is not necessarily the number of
        elements of the currently drawn curve. Instead, it reflects the current
        sizes of the data buffers behind it that will be drawn onto screen by
        the next call to ``update()``.
        """
        # fmt: off
        locker = QtCore.QMutexLocker(self._mutex) # pylint: disable=unused-variable
        # fmt: on
        return (len(self._buffer_x), len(self._buffer_y))


# ------------------------------------------------------------------------------
#   Derived thread-safe curves
# ------------------------------------------------------------------------------


class HistoryChartCurve(ThreadSafeCurve):
    """Provides a thread-safe curve with underlying ring buffers for the
    (x, y)-data. New readings are placed at the end (right-side) of the
    buffer, pushing out the oldest readings when the buffer has reached its
    maximum capacity (FIFO). Use methods ``appendData()`` and
    ``extendData()`` to push in new data.

    The plotted x-data will be shifted such that the right-side is always
    set to 0. I.e., when ``x`` denotes time, the data is plotted backwards
    in time, hence the name *history* chart.

    See class ``ThreadSafeCurve`` for more details.
    """

    def __init__(self, capacity: int, linked_curve: pg.PlotDataItem):
        super().__init__(
            capacity=capacity,
            linked_curve=linked_curve,
            shift_right_x_to_zero=True,
        )


class BufferedPlotCurve(ThreadSafeCurve):
    """Provides a thread-safe curve with underlying ring buffers for the
    (x, y)-data. New readings are placed at the end (right-side) of the
    buffer, pushing out the oldest readings when the buffer has reached its
    maximum capacity (FIFO). Use methods ``appendData()`` and
    ``extendData()`` to push in new data.

    See class ``ThreadSafeCurve`` for more details.
    """

    def __init__(self, capacity: int, linked_curve: pg.PlotDataItem):
        super().__init__(
            capacity=capacity,
            linked_curve=linked_curve,
            shift_right_x_to_zero=False,
        )


class PlotCurve(ThreadSafeCurve):
    """Provides a thread-safe curve with underlying regular array buffers
    for the (x, y)-data. Use method ``setData()`` to set the data.

    See class ``ThreadSafeCurve`` for more details.
    """

    def __init__(self, linked_curve: pg.PlotDataItem):
        super().__init__(
            capacity=None,
            linked_curve=linked_curve,
            shift_right_x_to_zero=False,
        )


# ------------------------------------------------------------------------------
#   LegendSelect
# ------------------------------------------------------------------------------


class LegendSelect(QtCore.QObject):
    """Creates and manages a legend of all passed curves with checkboxes to
    show or hide each curve. The legend ends with a push button to show or
    hide all curves in one go. The full set of GUI elements is contained in
    attribute ``grid`` of type ``PyQt5.QtWidget.QGridLayout`` to be added to
    your GUI.

    The initial visibility, name and pen of each curve will be retrieved
    from the members within the passed curves, i.e.:

        * ``curve.isVisible()``
        * ``curve.name()``
        * ``curve.opts["pen"]``

    Example grid::

        □ Curve 1  [  /  ]
        □ Curve 2  [  /  ]
        □ Curve 3  [  /  ]
        [ Show / Hide all]

    Args:
        linked_curves (``List[Union[pyqtgraph.PlotDataItem, ThreadSafeCurve]]``):
            List of ``pyqtgraph.PlotDataItem`` or ``ThreadSafeCurve`` to be
            controlled by the legend.

        hide_toggle_button (``bool``, optional):
            Default: False

        box_bg_color (``QtGui.QColor``, optional):
            Background color of the legend boxes.

            Default: ``QtGui.QColor(0, 0, 0)``

        box_width (``int``, optional):
            Default: 40

        box_height (``int``, optional):
            Default: 23

    Attributes:
        chkbs (``List[PyQt5.QtWidgets.QCheckbox]``):
            List of checkboxes to control the visiblity of each curve.

        painted_boxes (``List[PyQt5.QtWidgets.QWidget]``):
            List of painted boxes illustrating the pen of each curve.

        qpbt_toggle (``PyQt5.QtWidgets.QPushButton``):
            Push button instance that toggles showing/hiding all curves in
            one go.

        grid (``PyQt5.QtWidgets.QGridLayout``):
            The full set of GUI elements combined into a grid to be added
            to your GUI.
    """

    def __init__(
        self,
        linked_curves: List[Union[pg.PlotDataItem, ThreadSafeCurve]],
        hide_toggle_button: bool = False,
        box_bg_color: QtGui.QColor = QtGui.QColor(0, 0, 0),
        box_width: int = 40,
        box_height: int = 23,
        parent=None,
    ):
        super().__init__(parent=parent)

        self._linked_curves = linked_curves
        self.chkbs = list()
        self.painted_boxes = list()
        self.grid = QtWid.QGridLayout(spacing=1)  # The full set of GUI elements

        for idx, curve in enumerate(self._linked_curves):
            chkb = QtWid.QCheckBox(
                text=curve.name(),
                layoutDirection=QtCore.Qt.LeftToRight,
                checked=curve.isVisible(),
            )
            self.chkbs.append(chkb)
            # fmt: off
            chkb.clicked.connect(lambda: self._updateVisibility())  # pylint: disable=unnecessary-lambda
            # fmt: on

            painted_box = self.PaintedBox(
                pen=curve.opts["pen"],
                box_bg_color=box_bg_color,
                box_width=box_width,
                box_height=box_height,
            )
            self.painted_boxes.append(painted_box)

            p = {"alignment": QtCore.Qt.AlignLeft}
            self.grid.addWidget(chkb, idx, 0, **p)
            self.grid.addWidget(painted_box, idx, 1)
            self.grid.setColumnStretch(0, 1)
            self.grid.setColumnStretch(1, 0)
            self.grid.setAlignment(QtCore.Qt.AlignTop)

        if not hide_toggle_button:
            self.qpbt_toggle = QtWid.QPushButton("Show / Hide all")
            self.grid.addItem(QtWid.QSpacerItem(0, 10), self.grid.rowCount(), 0)
            self.grid.addWidget(self.qpbt_toggle, self.grid.rowCount(), 0, 1, 3)
            self.qpbt_toggle.clicked.connect(self.toggle)

    @QtCore.pyqtSlot()
    def _updateVisibility(self):
        for idx, chkb in enumerate(self.chkbs):
            self._linked_curves[idx].setVisible(chkb.isChecked())

    @QtCore.pyqtSlot()
    def toggle(self):
        # First : If any checkbox is unchecked  --> check all
        # Second: If all checkboxes are checked --> uncheck all
        any_unchecked = False
        for chkb in self.chkbs:
            if not chkb.isChecked():
                chkb.setChecked(True)
                any_unchecked = True

        if not any_unchecked:
            for chkb in self.chkbs:
                chkb.setChecked(False)

        self._updateVisibility()

    class PaintedBox(QtWid.QWidget):
        def __init__(
            self, pen, box_bg_color, box_width, box_height, parent=None
        ):
            super().__init__(parent=parent)

            self.pen = pen
            self.box_bg_color = box_bg_color

            self.setFixedWidth(box_width)
            self.setFixedHeight(box_height)

        def paintEvent(self, _event):
            w = self.width()
            h = self.height()
            x = 8  # offset line
            y = 6  # offset line

            painter = QtGui.QPainter()
            painter.begin(self)
            painter.fillRect(0, 0, w, h, self.box_bg_color)
            painter.setPen(self.pen)
            painter.drawLine(QtCore.QLine(x, h - y, w - x, y))
            painter.end()


# ------------------------------------------------------------------------------
#   PlotManager
# ------------------------------------------------------------------------------


class PlotManager(QtCore.QObject):
    """
    Args:
        parent (``PyQt5.QtWidgets.QWidget``):
            Needs to be set to the parent ``QWidget`` for the ``QMessageBox`` as
            fired by button ``clear()`` to appear centered and modal to.

    """

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self._autorange_linked_plots = None
        self._presets_linked_plots = None
        self._presets_linked_curves = None
        self._clear_linked_curves = None

        self._presets = list()

        self.pbtn_fullrange = None
        self.pbtn_autorange_x = None
        self.pbtn_autorange_y = None
        self.pbtns_presets = list()  # Will contain list of QPushButton
        self.pbtn_clear = None

        # Will contain the full set of GUI elements:
        self.grid = QtWid.QGridLayout(spacing=1)
        self.grid.setAlignment(QtCore.Qt.AlignTop)

    class Presets(TypedDict):
        button_label: str
        x_axis_label: str
        x_axis_divisor: float
        x_axis_range: Tuple[float, float]

    # --------------------------------------------------------------------------
    #   Presets
    # --------------------------------------------------------------------------

    def add_preset_buttons(
        self,
        linked_plots: List[Union[pg.PlotItem, pg.ViewBox]],
        linked_curves: List[Union[pg.PlotDataItem, ThreadSafeCurve]],
        presets: List[Presets],
    ):
        """
        Args:
            presets (``List[TypedDict]``):
                List of dictionaries. Each dictionary should contain:
                    * button_label (``str``)
                    * x_axis_label (``str``)
                    * x_axis_divisor (``float``)
                    * x_axis_range (``Tuple(float, float)``)
        """
        if not isinstance(linked_plots, list):
            linked_plots = (linked_plots,)
        self._presets_linked_plots = linked_plots

        if not isinstance(linked_curves, list):
            linked_curves = (linked_curves,)
        self._presets_linked_curves = linked_curves

        self._presets = presets

        for idx, preset in enumerate(self._presets):
            pbtn_preset = QtWid.QPushButton(text=preset["button_label"])
            pbtn_preset.clicked.connect(partial(self.perform_preset, idx))
            self.grid.addWidget(pbtn_preset, self.grid.rowCount(), 0, 1, 2)
            self.pbtns_presets.append(pbtn_preset)

    def perform_preset(self, idx: int):
        """
        """
        if self._presets_linked_plots is not None:
            for plot in self._presets_linked_plots:
                plot.setXRange(*self._presets[idx]["x_axis_range"])
                plot.setLabel("bottom", self._presets[idx]["x_axis_label"])

        if self._presets_linked_curves is not None:
            for curve in self._presets_linked_curves:
                if isinstance(curve, ThreadSafeCurve):
                    curve.x_axis_divisor = self._presets[idx]["x_axis_divisor"]
                    curve.update(create_snapshot=False)

    # --------------------------------------------------------------------------
    #   Autorange
    # --------------------------------------------------------------------------

    def add_autorange_buttons(
        self, linked_plots: List[Union[pg.PlotItem, pg.ViewBox]],
    ):
        """
        """
        if not isinstance(linked_plots, list):
            linked_plots = (linked_plots,)
        self._autorange_linked_plots = linked_plots

        self.pbtn_fullrange = QtWid.QPushButton("Full range")
        self.pbtn_autorange_x = QtWid.QPushButton("auto x", maximumWidth=65)
        self.pbtn_autorange_y = QtWid.QPushButton("auto y", maximumWidth=65)

        self.pbtn_fullrange.clicked.connect(self.perform_fullrange)
        self.pbtn_autorange_x.clicked.connect(self.perform_autorange_x)
        self.pbtn_autorange_y.clicked.connect(self.perform_autorange_y)

        self.grid.addWidget(self.pbtn_fullrange, self.grid.rowCount(), 0, 1, 2)
        row_idx = self.grid.rowCount()
        self.grid.addWidget(self.pbtn_autorange_x, row_idx, 0)
        self.grid.addWidget(self.pbtn_autorange_y, row_idx, 1)

    def perform_fullrange(self):
        """
        """
        if self._autorange_linked_plots is not None:
            for plot in self._autorange_linked_plots:
                # Momentarily ignore `clipToView`
                clip_to_view_state = plot.clipToViewMode()
                if clip_to_view_state:
                    plot.setClipToView(False)

                plot.enableAutoRange("x", True)
                plot.enableAutoRange("x", False)
                plot.enableAutoRange("y", True)
                plot.enableAutoRange("y", False)

                # Restore `clipToView`
                if clip_to_view_state:
                    plot.setClipToView(True)

    def perform_autorange_x(self):
        """
        """
        if self._autorange_linked_plots is not None:
            for plot in self._autorange_linked_plots:
                # Momentarily ignore `clipToView`
                clip_to_view_state = plot.clipToViewMode()
                if clip_to_view_state:
                    plot.setClipToView(False)

                plot.enableAutoRange("x", True)

                # Restore `clipToView`
                if clip_to_view_state:
                    plot.setClipToView(True)

    def perform_autorange_y(self):
        """
        """
        if self._autorange_linked_plots is not None:
            for plot in self._autorange_linked_plots:
                plot.enableAutoRange("y", True)

    # --------------------------------------------------------------------------
    #   Clear
    # --------------------------------------------------------------------------

    def add_clear_button(
        self, linked_curves: List[Union[pg.PlotDataItem, ThreadSafeCurve]],
    ):
        """
        """
        if not isinstance(linked_curves, list):
            linked_curves = (linked_curves,)
        self._clear_linked_curves = linked_curves

        self.pbtn_clear = QtWid.QPushButton("Clear")
        self.pbtn_clear.clicked.connect(self.perform_clear)

        if self.grid.rowCount() > 1:
            self.grid.addItem(QtWid.QSpacerItem(0, 10), self.grid.rowCount(), 0)
        self.grid.addWidget(self.pbtn_clear, self.grid.rowCount(), 0, 1, 2)

    def perform_clear(self):
        """
        """
        str_msg = "Are you sure you want to clear the plots?"
        reply = QtWid.QMessageBox.warning(
            self.parent(),
            "Clear plots",
            str_msg,
            QtWid.QMessageBox.Yes | QtWid.QMessageBox.No,
            QtWid.QMessageBox.No,
        )

        if reply == QtWid.QMessageBox.Yes:
            if self._clear_linked_curves is not None:
                for curve in self._clear_linked_curves:
                    curve.clear()
