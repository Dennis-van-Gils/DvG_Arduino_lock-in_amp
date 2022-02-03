#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Arduino lock-in amplifier
Minimal running example for trouble-shooting library
"""
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/DvG_Arduino_lock-in_amp"
__date__ = "03-02-2022"
__version__ = "1.0.0"

import sys
import time as Time

from PyQt5 import QtCore, QtGui, QtWidgets as QtWid
from PyQt5.QtCore import QDateTime
import pyqtgraph as pg
import numpy as np

from dvg_qdeviceio import QDeviceIO, DAQ_TRIGGER
from dvg_pyqtgraph_threadsafe import HistoryChartCurve
from dvg_debug_functions import dprint

from Alia_protocol_serial import Alia, Waveform

# When True, favors more frames per second for graphing at the expense of a
# higher CPU load and possibly dropped samples.
BOOST_FPS_GRAPHING = True

try:
    import OpenGL.GL as gl  # pylint: disable=unused-import

    pg.setConfigOptions(useOpenGL=True)
    pg.setConfigOptions(enableExperimental=True)
    pg.setConfigOptions(antialias=True)
    print("OpenGL hardware acceleration enabled.")
except:  # pylint: disable=bare-except
    pg.setConfigOptions(useOpenGL=False)
    pg.setConfigOptions(enableExperimental=False)
    pg.setConfigOptions(antialias=False)
    print("WARNING: Could not enable OpenGL hardware acceleration.")
    print("Check if prerequisite 'PyOpenGL' library is installed.")

# Short-hand alias for DEBUG information
def curThreadName():
    return QtCore.QThread.currentThread().objectName()


# ------------------------------------------------------------------------------
#   MainWindow
# ------------------------------------------------------------------------------


class MainWindow(QtWid.QWidget):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        self.setGeometry(50, 50, 900, 800)
        self.setWindowTitle("Arduino lock-in amplifier")

        # -----------------------------------
        #   Frame top
        # -----------------------------------

        # Left box
        self.qlbl_update_counter = QtWid.QLabel("0")
        self.qlbl_sample_rate = QtWid.QLabel(
            "SAMPLE RATE: {:,.0f} Hz".format(alia.config.Fs)
        )
        self.qlbl_buffer_size = QtWid.QLabel(
            "BUFFER SIZE  : %i" % alia.config.BLOCK_SIZE
        )
        self.qlbl_DAQ_rate = QtWid.QLabel("Buffers/s: nan")
        self.qlbl_DAQ_rate.setMinimumWidth(100)

        vbox_left = QtWid.QVBoxLayout()
        vbox_left.addWidget(self.qlbl_sample_rate)
        vbox_left.addWidget(self.qlbl_buffer_size)
        vbox_left.addStretch(1)
        vbox_left.addWidget(self.qlbl_DAQ_rate)
        vbox_left.addWidget(self.qlbl_update_counter)

        # Middle box
        self.qlbl_title = QtWid.QLabel(
            "Arduino lock-in amplifier",
            font=QtGui.QFont("Palatino", 14, weight=QtGui.QFont.Bold),
        )
        self.qlbl_title.setAlignment(QtCore.Qt.AlignCenter)
        self.qlbl_cur_date_time = QtWid.QLabel("00-00-0000    00:00:00")
        self.qlbl_cur_date_time.setAlignment(QtCore.Qt.AlignCenter)

        vbox_middle = QtWid.QVBoxLayout()
        vbox_middle.addWidget(self.qlbl_title)
        vbox_middle.addWidget(self.qlbl_cur_date_time)

        # Right box
        self.qpbt_exit = QtWid.QPushButton("Exit")
        self.qpbt_exit.clicked.connect(self.close)
        self.qpbt_exit.setMinimumHeight(30)
        self.qpbt_ENA_lockin = QtWid.QPushButton(
            "lock-in OFF", checkable=True, minimumHeight=40
        )
        self.qpbt_ENA_lockin.clicked.connect(self.process_qpbt_ENA_lockin)

        vbox_right = QtWid.QVBoxLayout()
        vbox_right.addWidget(self.qpbt_exit)
        vbox_right.addWidget(self.qpbt_ENA_lockin)
        vbox_right.addStretch(1)

        # Round up frame
        hbox_top = QtWid.QHBoxLayout()
        hbox_top.addLayout(vbox_left)
        hbox_top.addStretch(1)
        hbox_top.addLayout(vbox_middle)
        hbox_top.addStretch(1)
        hbox_top.addLayout(vbox_right)

        # -----------------------------------
        #   Frame 'Reference and signal'
        # -----------------------------------

        # Chart 'Readings'
        self.gw_refsig = pg.GraphicsWindow()
        self.gw_refsig.setBackground([20, 20, 20])
        self.pi_refsig = self.gw_refsig.addPlot()

        p = {"color": "#BBB", "font-size": "10pt"}
        self.pi_refsig.showGrid(x=1, y=1)
        self.pi_refsig.setTitle("Readings", **p)
        self.pi_refsig.setLabel("bottom", text="time (ms)", **p)
        self.pi_refsig.setLabel("left", text="voltage (V)", **p)
        self.pi_refsig.setXRange(
            -alia.config.BLOCK_SIZE * alia.config.SAMPLING_PERIOD * 1e3,
            0,
            padding=0,
        )
        self.pi_refsig.setYRange(1.0, 2.0, padding=0.05)
        self.pi_refsig.setAutoVisible(x=True, y=True)
        self.pi_refsig.setClipToView(True)

        PEN_01 = pg.mkPen(color=[255, 0, 0], width=3)
        PEN_03 = pg.mkPen(color=[0, 255, 255], width=3)
        self.hcc_ref_X = HistoryChartCurve(
            alia.config.BLOCK_SIZE, self.pi_refsig.plot(pen=PEN_01)
        )
        self.hcc_sig_I = HistoryChartCurve(
            alia.config.BLOCK_SIZE, self.pi_refsig.plot(pen=PEN_03)
        )
        self.hcc_ref_X.x_axis_divisor = 1000  # From [us] to [ms]
        self.hcc_sig_I.x_axis_divisor = 1000  # From [us] to [ms]
        self.hccs__refsig = (self.hcc_ref_X, self.hcc_sig_I)

        hbox_refsig = QtWid.QHBoxLayout()
        hbox_refsig.addWidget(self.gw_refsig, stretch=1)

        # -----------------------------------
        #   Round up full window
        # -----------------------------------

        vbox = QtWid.QVBoxLayout(self)
        vbox.addLayout(hbox_top)
        vbox.addLayout(hbox_refsig, stretch=1)

        # -----------------------------------
        #   Create wall clock timer
        # -----------------------------------

        self.timer_wall_clock = QtCore.QTimer()
        self.timer_wall_clock.timeout.connect(self.update_wall_clock)
        self.timer_wall_clock.start(50)

    @QtCore.pyqtSlot()
    def update_graph_refsig(self):
        for hcc in self.hccs__refsig:
            hcc.update()

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    #   Handle controls
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    @QtCore.pyqtSlot()
    def update_wall_clock(self):
        cur_date_time = QDateTime.currentDateTime()
        self.qlbl_cur_date_time.setText(
            "%s    %s"
            % (
                cur_date_time.toString("dd-MM-yyyy"),
                cur_date_time.toString("HH:mm:ss"),
            )
        )

    @QtCore.pyqtSlot()
    def process_qpbt_ENA_lockin(self):
        if self.qpbt_ENA_lockin.isChecked():
            alia_qdev.turn_on()
            self.qpbt_ENA_lockin.setText("lock-in ON")
        else:
            alia_qdev.turn_off()
            self.qlbl_DAQ_rate.setText("Buffers/s: paused")
            self.qpbt_ENA_lockin.setText("lock-in OFF")


# ------------------------------------------------------------------------------
#   Arduino_pyqt
# ------------------------------------------------------------------------------


class Alia_qdev(QDeviceIO):
    def __init__(
        self,
        dev: Alia,
        DAQ_function=None,
        critical_not_alive_count=3,
        debug=False,
        **kwargs,
    ):
        super().__init__(dev, **kwargs)  # Pass kwargs onto QtCore.QObject()

        self.create_worker_DAQ(
            DAQ_trigger=DAQ_TRIGGER.CONTINUOUS,
            DAQ_function=DAQ_function,
            critical_not_alive_count=critical_not_alive_count,
            debug=debug,
        )

        self.create_worker_jobs(
            jobs_function=self.jobs_function,
            debug=debug,
        )

    def turn_on(self):
        self.send("turn_on")

    def turn_off(self):
        self.send("turn_off")

    def jobs_function(self, func, args):
        if func == "turn_on":
            if self.dev.turn_on(reset_timer=True):
                self.unpause_DAQ()

        elif func == "turn_off":
            self.pause_DAQ()
            self.dev.turn_off()

        else:
            # Default job handling
            func(*args)


# ------------------------------------------------------------------------------
#   Program termination routines
# ------------------------------------------------------------------------------


@QtCore.pyqtSlot()
def about_to_quit():
    print("\nAbout to quit")
    app.processEvents()
    alia_qdev.turn_off()
    alia_qdev.quit()
    alia.close()


# ------------------------------------------------------------------------------
#   Lock-in amplifier data-acquisition update function
# ------------------------------------------------------------------------------


def lockin_DAQ_update():
    # print(curThreadName())
    if alia.lockin_paused:  # Prevent throwings errors if just paused
        return False

    if not BOOST_FPS_GRAPHING:
        # Prevent possible concurrent pyqtgraph.GraphicsWindow() redraws and GUI
        # events when doing heavy calculations to unburden the CPU and prevent
        # dropped buffers. Dropped graphing frames are prefereable to dropped
        # data buffers.
        window.gw_refsig.setUpdatesEnabled(False)

    # tick = Time.perf_counter()
    success, _counter, time, ref_X, _ref_Y, sig_I = alia.listen_to_lockin_amp()
    # dprint("%i" % ((Time.perf_counter() - tick) * 1e3))

    if not success:
        return False

    if BOOST_FPS_GRAPHING:
        # Prevent possible concurrent pyqtgraph.GraphicsWindow() redraws and GUI
        # events when doing heavy calculations to unburden the CPU and prevent
        # dropped buffers. Dropped graphing frames are prefereable to dropped
        # data buffers.
        window.gw_refsig.setUpdatesEnabled(False)

    window.hcc_ref_X.extendData(time, ref_X)
    window.hcc_sig_I.extendData(time, sig_I)

    # Re-enable pyqtgraph.GraphicsWindow() redraws and GUI events
    window.gw_refsig.setUpdatesEnabled(True)

    return True


def update_GUI():
    # Major visual changes upcoming. Reduce CPU overhead by momentarily
    # disabling screen repaints and GUI events.
    window.setUpdatesEnabled(False)

    window.qlbl_update_counter.setText("%i" % alia_qdev.update_counter_DAQ)

    if alia_qdev.worker_DAQ._paused:  # pylint: disable=protected-access
        window.qlbl_DAQ_rate.setText("Buffers/s: paused")
    else:
        window.qlbl_DAQ_rate.setText(
            "Buffers/s: %.1f" % alia_qdev.obtained_DAQ_rate_Hz
        )
        window.update_graph_refsig()

    # Re-enable screen repaints and GUI events
    window.setUpdatesEnabled(True)


# ------------------------------------------------------------------------------
#   Main
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    QtCore.QThread.currentThread().setObjectName("MAIN")  # For DEBUG info

    # Connect to Arduino
    alia = Alia(read_timeout=1)
    alia.auto_connect()

    if not alia.is_alive:
        print("\nCheck connection and try resetting the Arduino.")
        print("Exiting...\n")
        sys.exit(0)

    alia.begin(freq=109.8, V_offset=1.5, V_ampl=0.5, waveform=Waveform.Sine)

    # Create workers and threads
    alia_qdev = Alia_qdev(
        dev=alia,
        DAQ_function=lockin_DAQ_update,
        critical_not_alive_count=np.nan,
        debug=True,
    )
    alia_qdev.signal_DAQ_updated.connect(update_GUI)

    # Create application and main window
    app = 0  # Work-around for kernel crash when using Spyder IDE
    app = QtWid.QApplication(sys.argv)
    app.aboutToQuit.connect(about_to_quit)
    window = MainWindow()

    # Start threads
    alia_qdev.start(DAQ_priority=QtCore.QThread.TimeCriticalPriority)

    # Start the main GUI event loop
    window.show()
    sys.exit(app.exec_())
