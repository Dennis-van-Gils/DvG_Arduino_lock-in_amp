#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PyQt5 user-interface elements to facilitate the FIR filter design of
`dvg_ringbuffer_fir_filter.py`.
"""
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils"
__date__ = "27-08-2021"
__version__ = "1.0.0"
# pylint: disable=invalid-name, missing-function-docstring

from typing import Union, List

import numpy as np
from PyQt5 import QtCore, QtWidgets as QtWid

from dvg_ringbuffer_fir_filter import RingBuffer_FIR_Filter

# ------------------------------------------------------------------------------
#  Filter_design_GUI
# ------------------------------------------------------------------------------


class Filter_design_GUI(QtCore.QObject):
    """PyQt5 user-interface elements to facilitate the FIR filter design of
    `dvg_ringbuffer_fir_filter.py`.

    Args:
        firf (RingBuffer_FIR_Filter or a list thereof):
            The FIR filter instances to design the filter for.

    Attributes:
        qgrp (PyQt5.QtWidgets.QGroupBox):
            The main GUI element for the design of a FIR filter.

    Important method:
        update_filter_design()

    Signals:
        signal_filter_design_updated():
            Emitted when the user has interacted with the FIR filter design
            controls, or when method `update_filter_design()` has been called.
    """

    signal_filter_design_updated = QtCore.pyqtSignal()

    def __init__(
        self,
        firf: Union[RingBuffer_FIR_Filter, List[RingBuffer_FIR_Filter]],
        hide_ACDC_coupling_controls: bool = False,
        parent=None,
    ):
        super().__init__(parent=parent)

        # Accept either a single instance of Buffered_FIR_Filter or accept
        # a list of Buffered_FIR_Filter instances.
        if not isinstance(firf, list):
            firf = [firf]

        self.firfs: List[RingBuffer_FIR_Filter] = firf
        self.qgrp = QtWid.QGroupBox("FIR filter design")
        self.qtbl_bandstop_cellChanged_lock = False  # Work-around flag
        self.create_GUI(hide_ACDC_coupling_controls=hide_ACDC_coupling_controls)

    # --------------------------------------------------------------------------
    #   create_GUI
    # --------------------------------------------------------------------------

    def create_GUI(self, hide_ACDC_coupling_controls):
        # QGROUP: Filter design controls

        # Textbox widths for fitting N 'x' characters using the current font
        e = QtWid.QLineEdit()
        ex8 = (
            8
            + 8 * e.fontMetrics().width("x")
            + e.textMargins().left()
            + e.textMargins().right()
            + e.contentsMargins().left()
            + e.contentsMargins().right()
        )
        del e
        default_font_pt = QtWid.QApplication.font().pointSize()

        self.qtbl_bandstop = QtWid.QTableWidget()
        self.qtbl_bandstop.setStyleSheet(
            "QTableWidget {font-size: %ipt;"
            "font-family: MS Shell Dlg 2}"
            "QHeaderView  {font-size: %ipt;"
            "font-family: MS Shell Dlg 2}"
            "QHeaderView:section {background-color: lightgray}"
            % (default_font_pt, default_font_pt)
        )
        self.qtbl_bandstop.setRowCount(6)
        self.qtbl_bandstop.setColumnCount(2)
        self.qtbl_bandstop.setColumnWidth(0, ex8)
        self.qtbl_bandstop.setColumnWidth(1, ex8)
        self.qtbl_bandstop.setHorizontalHeaderLabels(["from", "to"])
        self.qtbl_bandstop.horizontalHeader().setDefaultAlignment(
            QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter
        )
        self.qtbl_bandstop.horizontalHeader().setSectionResizeMode(
            QtWid.QHeaderView.Fixed
        )
        self.qtbl_bandstop.verticalHeader().setSectionResizeMode(
            QtWid.QHeaderView.Fixed
        )

        self.qtbl_bandstop.setSizePolicy(
            QtWid.QSizePolicy.Minimum, QtWid.QSizePolicy.Minimum
        )
        self.qtbl_bandstop.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOff
        )
        self.qtbl_bandstop.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOff
        )
        self.qtbl_bandstop.setFixedSize(
            self.qtbl_bandstop.horizontalHeader().length()
            + self.qtbl_bandstop.verticalHeader().width()
            + 2,
            self.qtbl_bandstop.verticalHeader().length()
            + self.qtbl_bandstop.horizontalHeader().height()
            + 2,
        )

        self.qtbl_bandstop_items = list()
        for row in range(self.qtbl_bandstop.rowCount()):
            for col in range(self.qtbl_bandstop.columnCount()):
                myItem = QtWid.QTableWidgetItem()
                myItem.setTextAlignment(
                    QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter
                )
                self.qtbl_bandstop_items.append(myItem)
                self.qtbl_bandstop.setItem(row, col, myItem)

        p = {
            "maximumWidth": ex8,
            "minimumWidth": ex8,
            "alignment": QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight,
        }
        self.qpbt_coupling = QtWid.QPushButton("AC", checkable=True)
        self.qlin_DC_cutoff = QtWid.QLineEdit(**p)
        self.qlin_window = QtWid.QLineEdit(readOnly=True)
        self.qlin_window.setText(self.firfs[0].config.firwin_window_descr)
        self.qlin_N_taps = QtWid.QLineEdit(readOnly=True, **p)
        self.qlin_T_settle_filter = QtWid.QLineEdit(readOnly=True, **p)

        # fmt: off
        i = 0
        grid = QtWid.QGridLayout(spacing=4)
        if not hide_ACDC_coupling_controls:
            grid.addWidget(QtWid.QLabel("Input coupling AC/DC:"), i, 0, 1, 3); i += 1
            grid.addWidget(self.qpbt_coupling                   , i, 0, 1, 3); i += 1
            grid.addWidget(QtWid.QLabel("Cutoff:")              , i, 0)
            grid.addWidget(self.qlin_DC_cutoff                  , i, 1)
            grid.addWidget(QtWid.QLabel("Hz")                   , i, 2)      ; i += 1
            grid.addItem(QtWid.QSpacerItem(0, 12)               , i, 0, 1, 3); i += 1
        grid.addWidget(QtWid.QLabel("Band-stop ranges [Hz]:"), i, 0, 1, 3); i += 1
        grid.addWidget(self.qtbl_bandstop                    , i, 0, 1, 3); i += 1
        grid.addItem(QtWid.QSpacerItem(0, 8)                 , i, 0, 1, 3); i += 1
        grid.addWidget(QtWid.QLabel("Window:")               , i, 0, 1, 3); i += 1
        grid.addWidget(self.qlin_window                      , i, 0, 1, 3); i += 1
        grid.addWidget(QtWid.QLabel("N_taps:")               , i, 0)
        grid.addWidget(self.qlin_N_taps                      , i, 1, 1, 2); i += 1
        grid.addItem(QtWid.QSpacerItem(0, 8)                 , i, 0, 1, 3); i += 1
        grid.addWidget(QtWid.QLabel("Settling times:")       , i, 0, 1, 3); i += 1
        grid.addWidget(QtWid.QLabel("Filter:")               , i, 0)
        grid.addWidget(self.qlin_T_settle_filter             , i, 1)
        grid.addWidget(QtWid.QLabel("s")                     , i, 2)
        grid.setAlignment(QtCore.Qt.AlignTop)
        # fmt: on

        self.qpbt_coupling.clicked.connect(self.process_coupling)
        self.qlin_DC_cutoff.editingFinished.connect(self.process_coupling)
        self.populate_design_controls()
        self.qtbl_bandstop.keyPressEvent = self.qtbl_bandstop_KeyPressEvent
        self.qtbl_bandstop.cellChanged.connect(
            self.process_qtbl_bandstop_cellChanged
        )

        # Ignore cellChanged event when locked
        self.qtbl_bandstop_cellChanged_lock = False

        self.qgrp.setLayout(grid)

    # --------------------------------------------------------------------------
    #   populate_design_controls
    # --------------------------------------------------------------------------

    def populate_design_controls(self):
        c = self.firfs[0].config  # Shorthand
        freq_list = c.firwin_cutoff

        if c.firwin_pass_zero:
            self.qpbt_coupling.setText("DC")
            self.qpbt_coupling.setChecked(True)
            self.qlin_DC_cutoff.setEnabled(False)
            self.qlin_DC_cutoff.setReadOnly(True)
        else:
            self.qpbt_coupling.setText("AC")
            self.qpbt_coupling.setChecked(False)
            self.qlin_DC_cutoff.setText("%.1f" % freq_list[0])
            self.qlin_DC_cutoff.setEnabled(True)
            self.qlin_DC_cutoff.setReadOnly(False)
            freq_list = freq_list[1:]

        self.qlin_N_taps.setText("%i" % c.firwin_numtaps)
        self.qlin_T_settle_filter.setText(
            "%.2f" % self.firfs[0].T_settle_filter
        )

        # Next line cannot be replaced by setUpdatesEnabled(False)
        self.qtbl_bandstop_cellChanged_lock = True

        for row in range(self.qtbl_bandstop.rowCount()):
            for col in range(self.qtbl_bandstop.columnCount()):
                try:
                    freq = freq_list[row * 2 + col]
                    freq_str = "%.1f" % freq
                except IndexError:
                    freq_str = ""
                self.qtbl_bandstop_items[row * 2 + col].setText(freq_str)
        self.qtbl_bandstop_cellChanged_lock = False

    # --------------------------------------------------------------------------
    #   process_coupling
    # --------------------------------------------------------------------------

    @QtCore.pyqtSlot()
    def process_coupling(self):
        c = self.firfs[0].config  # Shorthand

        DC_cutoff = self.qlin_DC_cutoff.text()
        try:
            DC_cutoff = float(DC_cutoff)
            DC_cutoff = round(DC_cutoff * 10) / 10
        except:  # pylint: disable=bare-except
            DC_cutoff = 0
        if DC_cutoff <= 0:
            DC_cutoff = 1.0
        self.qlin_DC_cutoff.setText("%.1f" % DC_cutoff)

        cutoff = c.firwin_cutoff
        cutoff_old = cutoff
        if self.qpbt_coupling.isChecked():
            pass_zero = True
            if not c.firwin_pass_zero:
                cutoff = cutoff[1:]
        else:
            pass_zero = False
            if c.firwin_pass_zero:
                cutoff = np.insert(cutoff, 0, DC_cutoff)
            else:
                cutoff[0] = DC_cutoff

        # cutoff list cannot be empty. Force at least one value in the list
        if np.size(cutoff) == 0:
            cutoff = np.append(cutoff, cutoff_old[0])
            # And refrain from displaying a QMessageBox

        for firf in self.firfs:
            firf.compute_firwin_and_freqz(
                firwin_cutoff=cutoff, firwin_pass_zero=pass_zero
            )
        self.update_filter_design()

    # --------------------------------------------------------------------------
    #   construct_cutoff_list
    # --------------------------------------------------------------------------

    def construct_cutoff_list(self):
        if self.firfs[0].config.firwin_pass_zero:
            # Input coupling: DC
            cutoff = np.array([])
        else:
            # Input coupling: AC
            cutoff = self.firfs[0].config.firwin_cutoff[0]

        for row in range(self.qtbl_bandstop.rowCount()):
            for col in range(self.qtbl_bandstop.columnCount()):
                value = self.qtbl_bandstop.item(row, col).text()
                try:
                    value = float(value)
                    value = round(value * 10) / 10
                    cutoff = np.append(cutoff, value)
                except ValueError:
                    value = None

        # cutoff list cannot be empty. Force at least one value in the list
        if np.size(cutoff) == 0:
            cutoff = np.append(cutoff, 1.0)

            msg = QtWid.QMessageBox()
            msg.setIcon(QtWid.QMessageBox.Information)
            msg.setWindowTitle("FIR filter design")
            msg.setText(
                "The list cannot be empty when input coupling is set "
                "to DC.<br>Putting 1.0 Hz back into the list.<br><br>"
                "You could set the input coupling to AC first."
            )
            msg.setStandardButtons(QtWid.QMessageBox.Ok)
            msg.exec_()

        for firf in self.firfs:
            firf.compute_firwin_and_freqz(firwin_cutoff=cutoff)

    # --------------------------------------------------------------------------
    #   qtbl_bandstop_KeyPressEvent
    # --------------------------------------------------------------------------

    def qtbl_bandstop_KeyPressEvent(self, event):
        # Handle special events
        if event.key() == QtCore.Qt.Key_Delete:
            msg = QtWid.QMessageBox()
            msg.setIcon(QtWid.QMessageBox.Information)
            msg.setWindowTitle("FIR filter design")
            msg.setText(
                "Are you sure you want to remove the<br>"
                "following frequencies from the list?"
            )
            list_freqs = np.array([])
            for item in self.qtbl_bandstop.selectedItems():
                if item.text() != "":
                    list_freqs = np.append(list_freqs, item.text())
            str_freqs = ", ".join(map(str, list_freqs))
            msg.setInformativeText("%s [Hz]" % str_freqs)
            msg.setStandardButtons(QtWid.QMessageBox.Yes | QtWid.QMessageBox.No)
            answer = msg.exec_()

            if answer == QtWid.QMessageBox.Yes:
                self.qtbl_bandstop_cellChanged_lock = True
                for item in self.qtbl_bandstop.selectedItems():
                    item.setText("")
                self.qtbl_bandstop_cellChanged_lock = False

                self.construct_cutoff_list()
                self.update_filter_design()

        # Regular event handling for QTableWidgets
        return QtWid.QTableWidget.keyPressEvent(self.qtbl_bandstop, event)

    # --------------------------------------------------------------------------
    #   process_qtbl_bandstop_cellChanged
    # --------------------------------------------------------------------------

    @QtCore.pyqtSlot(int, int)
    def process_qtbl_bandstop_cellChanged(self, _k, _l):
        if self.qtbl_bandstop_cellChanged_lock:
            return
        # print("cellChanged %i %i" % (_k, _l))

        self.construct_cutoff_list()
        self.update_filter_design()

    # --------------------------------------------------------------------------
    #   update_filter_design
    # --------------------------------------------------------------------------

    def update_filter_design(self):
        """Reflect outside changes made to the filter configuration by updating
        the GUI elements. Will emit `signal_filter_design_updated` once done.
        """
        self.populate_design_controls()
        self.signal_filter_design_updated.emit()
