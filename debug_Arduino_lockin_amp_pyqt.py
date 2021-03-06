#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Arduino lock-in amplifier
Minimal running example for trouble-shooting library
"""
__author__      = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__         = "https://github.com/Dennis-van-Gils/DvG_Arduino_lock-in_amp"
__date__        = "24-04-2019"
__version__     = "1.0.0"

import sys

from PyQt5 import QtCore, QtGui
from PyQt5 import QtWidgets as QtWid
from PyQt5.QtCore import QDateTime
import pyqtgraph as pg
import numpy as np

import time as Time

from DvG_pyqt_ChartHistory import ChartHistory
from DvG_pyqt_controls import create_Toggle_button
from DvG_debug_functions import dprint

import DvG_dev_Base__pyqt_lib as Dev_Base_pyqt_lib
import DvG_dev_Arduino_lockin_amp__fun_serial as lockin_functions

# Monkey-patch errors in pyqtgraph v0.10
import DvG_monkeypatch_pyqtgraph as pgmp
pg.PlotCurveItem.paintGL = pgmp.PlotCurveItem_paintGL

# BOOST_FPS_GRAPHING = true:
# Favors more frames per second for graphing at the expense of a higher CPU load and possibly
# dropped samples.
BOOST_FPS_GRAPHING = True

try:
    import OpenGL.GL as gl
    pg.setConfigOptions(useOpenGL=True)
    pg.setConfigOptions(enableExperimental=True)
    pg.setConfigOptions(antialias=False)
    print("OpenGL hardware acceleration enabled.")
except:
    pg.setConfigOptions(useOpenGL=False)
    pg.setConfigOptions(enableExperimental=False)
    pg.setConfigOptions(antialias=False)
    print("WARNING: Could not enable OpenGL hardware acceleration.")
    print("Check if prerequisite 'PyOpenGL' library is installed.")

# Short-hand alias for DEBUG information
def curThreadName(): return QtCore.QThread.currentThread().objectName()
        
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
        self.qlbl_sample_rate = QtWid.QLabel("SAMPLE RATE: %.2f Hz" %
                                             (1/lockin.config.ISR_CLOCK))
        self.qlbl_buffer_size = QtWid.QLabel("BUFFER SIZE  : %i" %
                                             lockin.config.BUFFER_SIZE)
        self.qlbl_DAQ_rate = QtWid.QLabel("Buffers/s: nan")
        self.qlbl_DAQ_rate.setMinimumWidth(100)

        vbox_left = QtWid.QVBoxLayout()
        vbox_left.addWidget(self.qlbl_sample_rate)
        vbox_left.addWidget(self.qlbl_buffer_size)
        vbox_left.addStretch(1)
        vbox_left.addWidget(self.qlbl_DAQ_rate)
        vbox_left.addWidget(self.qlbl_update_counter)

         # Middle box
        self.qlbl_title = QtWid.QLabel("Arduino lock-in amplifier",
                font=QtGui.QFont("Palatino", 14, weight=QtGui.QFont.Bold))
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
        self.qpbt_ENA_lockin = create_Toggle_button("lock-in OFF")
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
        
        p = {'color': '#BBB', 'font-size': '10pt'}
        self.pi_refsig.showGrid(x=1, y=1)
        self.pi_refsig.setTitle('Readings', **p)
        self.pi_refsig.setLabel('bottom', text='time (ms)', **p)
        self.pi_refsig.setLabel('left', text='voltage (V)', **p)
        self.pi_refsig.setXRange(-lockin.config.BUFFER_SIZE * 
                                 lockin.config.ISR_CLOCK * 1e3,
                                 0, padding=0)
        self.pi_refsig.setYRange(1.0, 2.0, padding=0.05)
        self.pi_refsig.setAutoVisible(x=True, y=True)
        self.pi_refsig.setClipToView(True)

        PEN_01 = pg.mkPen(color=[255, 0  , 0  ], width=3)
        PEN_03 = pg.mkPen(color=[0  , 255, 255], width=3)
        self.CH_ref_X = ChartHistory(lockin.config.BUFFER_SIZE,
                                     self.pi_refsig.plot(pen=PEN_01))
        self.CH_sig_I = ChartHistory(lockin.config.BUFFER_SIZE,
                                     self.pi_refsig.plot(pen=PEN_03))
        self.CH_ref_X.x_axis_divisor = 1000     # From [us] to [ms]
        self.CH_sig_I.x_axis_divisor = 1000     # From [us] to [ms]
        self.CHs_refsig = [self.CH_ref_X, self.CH_sig_I]
        
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
    def update_chart_refsig(self):
        [CH.update_curve() for CH in self.CHs_refsig]

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    #   Handle controls
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
        
    @QtCore.pyqtSlot()
    def update_wall_clock(self):
        cur_date_time = QDateTime.currentDateTime()
        self.qlbl_cur_date_time.setText("%s    %s" %
                                        (cur_date_time.toString("dd-MM-yyyy"),
                                         cur_date_time.toString("HH:mm:ss")))
    
    @QtCore.pyqtSlot()
    def process_qpbt_ENA_lockin(self):
        if self.qpbt_ENA_lockin.isChecked():
            lockin_pyqt.turn_on()
            self.qpbt_ENA_lockin.setText("lock-in ON")
        else:
            lockin_pyqt.turn_off()
            self.qlbl_DAQ_rate.setText("Buffers/s: paused")
            self.qpbt_ENA_lockin.setText("lock-in OFF")

# ------------------------------------------------------------------------------
#   Arduino_pyqt
# ------------------------------------------------------------------------------

class Arduino_lockin_amp_pyqt(Dev_Base_pyqt_lib.Dev_Base_pyqt, QtCore.QObject):
    def __init__(self,
                 dev: lockin_functions.Arduino_lockin_amp,
                 DAQ_update_interval_ms=10,
                 DAQ_function_to_run_each_update=None,
                 DAQ_critical_not_alive_count=3,
                 DAQ_timer_type=QtCore.Qt.PreciseTimer,
                 DAQ_trigger_by=Dev_Base_pyqt_lib.DAQ_trigger.CONTINUOUS,
                 #DAQ_trigger_by=Dev_Base_pyqt_lib.DAQ_trigger.INTERNAL_TIMER,
                 calc_DAQ_rate_every_N_iter=25,
                 DEBUG_worker_DAQ=False,
                 DEBUG_worker_send=False,
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
        self.worker_send.queued_instruction("turn_on")
        
    def turn_off(self):
        self.worker_send.queued_instruction("turn_off")
        
    def turn_off_immediately(self):
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
    
    def alt_process_jobs_function(self, func, args):
        if func == "turn_on":
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
        
# ------------------------------------------------------------------------------
#   Program termination routines
# ------------------------------------------------------------------------------

@QtCore.pyqtSlot()
def about_to_quit():
    print("\nAbout to quit")
    app.processEvents()
    if lockin.is_alive: lockin_pyqt.turn_off_immediately()
    lockin_pyqt.close_all_threads()
    lockin.close()

# ------------------------------------------------------------------------------
#   Lock-in amplifier data-acquisition update function
# ------------------------------------------------------------------------------

def lockin_DAQ_update():
    #print(curThreadName())
    if lockin.lockin_paused:  # Prevent throwings errors if just paused
        return False
    
    if not(BOOST_FPS_GRAPHING):
        # Prevent possible concurrent pyqtgraph.GraphicsWindow() redraws and GUI
        # events when doing heavy calculations to unburden the CPU and prevent
        # dropped buffers. Dropped graphing frames are prefereable to dropped
        # data buffers.
        window.gw_refsig.setUpdatesEnabled(False)
    
    tick = Time.perf_counter()
    [success, time, ref_X, ref_Y, sig_I] = lockin.listen_to_lockin_amp()
    dprint("%i" % ((Time.perf_counter() - tick)*1e3))
    
    if not(success):
        return False
        
    if BOOST_FPS_GRAPHING:
        # Prevent possible concurrent pyqtgraph.GraphicsWindow() redraws and GUI
        # events when doing heavy calculations to unburden the CPU and prevent
        # dropped buffers. Dropped graphing frames are prefereable to dropped
        # data buffers.
        window.gw_refsig.setUpdatesEnabled(False)

    window.CH_ref_X.add_new_readings(time, ref_X)    
    window.CH_sig_I.add_new_readings(time, sig_I)
    
    # Re-enable pyqtgraph.GraphicsWindow() redraws and GUI events
    window.gw_refsig.setUpdatesEnabled(True)
    
    return True

def update_GUI():
    # Major visual changes upcoming. Reduce CPU overhead by momentarily
    # disabling screen repaints and GUI events.
    window.setUpdatesEnabled(False)
    
    window.qlbl_update_counter.setText("%i" % lockin_pyqt.DAQ_update_counter)
    
    if not lockin.lockin_paused:
        window.qlbl_DAQ_rate.setText("Buffers/s: %.1f" % 
                                     lockin_pyqt.obtained_DAQ_rate_Hz)
        window.update_chart_refsig()
        
    # Re-enable screen repaints and GUI events
    window.setUpdatesEnabled(True)

# ------------------------------------------------------------------------------
#   Main
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    QtCore.QThread.currentThread().setObjectName('MAIN')    # For DEBUG info
    
    # Connect to Arduino
    lockin = lockin_functions.Arduino_lockin_amp(baudrate=1e6, read_timeout=1)
    if not lockin.connect_at_port("COM10"):
        print("Can't connect to Arduino")
        sys.exit(0)
    lockin.begin(ref_freq=109.8, ref_V_offset=1.5, ref_V_ampl=0.5)
    
    # Create workers and threads
    lockin_pyqt = Arduino_lockin_amp_pyqt(
                            dev=lockin,
                            DAQ_function_to_run_each_update=lockin_DAQ_update,
                            DAQ_critical_not_alive_count=np.nan,
                            calc_DAQ_rate_every_N_iter=10,
                            DEBUG_worker_DAQ=False,
                            DEBUG_worker_send=False)
    lockin_pyqt.signal_DAQ_updated.connect(update_GUI)
    
    # Create application and main window
    app = 0    # Work-around for kernel crash when using Spyder IDE
    app = QtWid.QApplication(sys.argv)
    app.aboutToQuit.connect(about_to_quit)
    window = MainWindow()
    
    # Start threads
    lockin_pyqt.start_thread_worker_DAQ(QtCore.QThread.TimeCriticalPriority)
    lockin_pyqt.start_thread_worker_send()
    
    # Start the main GUI event loop
    window.show()
    sys.exit(app.exec_())