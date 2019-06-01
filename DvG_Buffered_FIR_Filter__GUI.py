#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
__author__      = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__         = "https://github.com/Dennis-van-Gils"
__date__        = "01-06-2019"
__version__     = "1.0.0"

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets as QtWid
from DvG_Buffered_FIR_Filter import Buffered_FIR_Filter

class Filter_design_GUI(QtCore.QObject):
    signal_filter_design_updated = QtCore.pyqtSignal()
    
    def __init__(self, firf: Buffered_FIR_Filter, parent=None):
        super().__init__(parent=parent)
    
        # Accept either a single instance of Buffered_FIR_Filter or accept
        # a list of Buffered_FIR_Filter instances.
        if not isinstance(firf, list):
            firf = [firf]
        
        self.firfs = firf
        self.qgrp = QtWid.QGroupBox("FIR filter design")
        self.create_GUI()

    # --------------------------------------------------------------------------
    #   create_GUI
    # --------------------------------------------------------------------------

    def create_GUI(self):
        # Textbox widths for fitting N 'x' characters using the current font
        e = QtGui.QLineEdit()
        ex8  = (8 + 8 * e.fontMetrics().width('x') + 
                e.textMargins().left()     + e.textMargins().right() + 
                e.contentsMargins().left() + e.contentsMargins().right())
        del e
        
        # QGROUP: Filter design controls
        self.qtbl_bandstop = QtWid.QTableWidget()
    
        default_font_pt = QtWid.QApplication.font().pointSize()
        self.qtbl_bandstop.setStyleSheet(
                "QTableWidget {font-size: %ipt;"
                              "font-family: MS Shell Dlg 2}"
                "QHeaderView  {font-size: %ipt;"
                              "font-family: MS Shell Dlg 2}"
                "QHeaderView:section {background-color: lightgray}" %
                (default_font_pt, default_font_pt))
    
        self.qtbl_bandstop.setRowCount(6)
        self.qtbl_bandstop.setColumnCount(2)
        self.qtbl_bandstop.setColumnWidth(0, ex8)
        self.qtbl_bandstop.setColumnWidth(1, ex8)
        self.qtbl_bandstop.setHorizontalHeaderLabels (['from', 'to'])
        self.qtbl_bandstop.horizontalHeader().setDefaultAlignment(
                QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.qtbl_bandstop.horizontalHeader().setSectionResizeMode(
                QtWid.QHeaderView.Fixed)
        self.qtbl_bandstop.verticalHeader().setSectionResizeMode(
                QtWid.QHeaderView.Fixed)
        
        self.qtbl_bandstop.setSizePolicy(
                QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.qtbl_bandstop.setVerticalScrollBarPolicy(
                QtCore.Qt.ScrollBarAlwaysOff)
        self.qtbl_bandstop.setHorizontalScrollBarPolicy(
                QtCore.Qt.ScrollBarAlwaysOff)
        self.qtbl_bandstop.setFixedSize(
                self.qtbl_bandstop.horizontalHeader().length() + 
                self.qtbl_bandstop.verticalHeader().width() + 2,
                self.qtbl_bandstop.verticalHeader().length() + 
                self.qtbl_bandstop.horizontalHeader().height() + 2)
        
        self.qtbl_bandstop_items = list()
        for row in range(self.qtbl_bandstop.rowCount()):
            for col in range(self.qtbl_bandstop.columnCount()):
                myItem = QtWid.QTableWidgetItem()
                myItem.setTextAlignment(QtCore.Qt.AlignRight |
                                        QtCore.Qt.AlignVCenter)
                self.qtbl_bandstop_items.append(myItem)
                self.qtbl_bandstop.setItem(row, col, myItem)
                
        p = {'maximumWidth': ex8, 'minimumWidth': ex8,
             'alignment': QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight}
        self.qpbt_coupling = QtWid.QPushButton("AC", checkable=True)
        self.qlin_DC_cutoff = QtWid.QLineEdit(**p)
        self.qlin_window = QtWid.QLineEdit(readOnly=True)
        self.qlin_window.setText(self.firfs[0].window_description)
        self.qlin_N_taps = QtWid.QLineEdit(readOnly=True, **p)
        self.qlin_T_settle_filter = QtWid.QLineEdit(readOnly=True, **p)
        self.qlin_T_settle_deque = QtWid.QLineEdit(readOnly=True, **p)
                
        i = 0
        grid = QtWid.QGridLayout(spacing=4)
        grid.addWidget(QtWid.QLabel('Input coupling AC/DC:') , i, 0, 1, 3); i+=1
        grid.addWidget(self.qpbt_coupling                    , i, 0, 1, 3); i+=1
        grid.addWidget(QtWid.QLabel('Cutoff:')               , i, 0)
        grid.addWidget(self.qlin_DC_cutoff                   , i, 1)
        grid.addWidget(QtWid.QLabel('Hz')                    , i, 2)      ; i+=1
        grid.addItem(QtWid.QSpacerItem(0, 12)                , i, 0, 1, 3); i+=1
        grid.addWidget(QtWid.QLabel('Band-stop ranges [Hz]:'), i, 0, 1, 3); i+=1
        grid.addWidget(self.qtbl_bandstop                    , i, 0, 1, 3); i+=1
        grid.addItem(QtWid.QSpacerItem(0, 8)                 , i, 0, 1, 3); i+=1
        grid.addWidget(QtWid.QLabel('Window:')               , i, 0, 1, 3); i+=1
        grid.addWidget(self.qlin_window                      , i, 0, 1, 3); i+=1
        grid.addWidget(QtWid.QLabel('N_taps:')               , i, 0)
        grid.addWidget(self.qlin_N_taps                      , i, 1, 1, 2); i+=1
        grid.addItem(QtWid.QSpacerItem(0, 8)                 , i, 0, 1, 3); i+=1
        grid.addWidget(QtWid.QLabel('Settling times:')       , i, 0, 1, 3); i+=1
        grid.addWidget(QtWid.QLabel('Filter:')               , i, 0)
        grid.addWidget(self.qlin_T_settle_filter             , i, 1)
        grid.addWidget(QtWid.QLabel('s')                     , i, 2)      ; i+=1
        grid.addWidget(QtWid.QLabel('Buffer:')               , i, 0)
        grid.addWidget(self.qlin_T_settle_deque              , i, 1)
        grid.addWidget(QtWid.QLabel('s')                     , i, 2)      ; i+=1
        grid.setAlignment(QtCore.Qt.AlignTop)
        
        self.qpbt_coupling.clicked.connect(self.process_coupling)
        self.qlin_DC_cutoff.editingFinished.connect(self.process_coupling)        
        self.populate_design_controls()     
        self.qtbl_bandstop.keyPressEvent = self.qtbl_bandstop_KeyPressEvent
        self.qtbl_bandstop.cellChanged.connect(
                self.process_qtbl_bandstop_cellChanged)
        self.qtbl_bandstop_cellChanged_lock = False  # Ignore cellChanged event
                                                     # when locked
    
        self.qgrp.setLayout(grid)
    
    def populate_design_controls(self):
        firf = self.firfs[0]
        freq_list = firf.cutoff;
        
        if self.firfs[0].pass_zero:
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
            
        self.qlin_N_taps.setText("%i" % firf.N_taps)
        self.qlin_T_settle_filter.setText("%.2f" % firf.T_settle_filter)
        self.qlin_T_settle_deque.setText("%.2f" % firf.T_settle_deque)
        
        self.qtbl_bandstop_cellChanged_lock = True # Can not be replaced by setUpdatesEnabled(False)
        for row in range(self.qtbl_bandstop.rowCount()):
            for col in range(self.qtbl_bandstop.columnCount()):
                try:
                    freq = freq_list[row*2 + col]
                    freq_str = "%.1f" % freq
                except IndexError:
                    freq_str = ""
                self.qtbl_bandstop_items[row*2 + col].setText(freq_str)
        self.qtbl_bandstop_cellChanged_lock = False
    
    @QtCore.pyqtSlot()
    def process_coupling(self):
        DC_cutoff = self.qlin_DC_cutoff.text()
        try:
            DC_cutoff = float(DC_cutoff)
            DC_cutoff = round(DC_cutoff*10)/10;
        except:
            DC_cutoff = 0
        if DC_cutoff <= 0: DC_cutoff = 1.0
        self.qlin_DC_cutoff.setText("%.1f" % DC_cutoff)
        
        cutoff = self.firfs[0].cutoff
        if self.qpbt_coupling.isChecked():
            pass_zero = True
            if not self.firfs[0].pass_zero:
                cutoff = cutoff[1:]
        else:
            pass_zero = False
            if self.firfs[0].pass_zero:
                cutoff = np.insert(cutoff, 0, DC_cutoff)
            else:
                cutoff[0] = DC_cutoff
                
        # cutoff list cannot be empty. Force at least one value in the list
        if np.size(cutoff) == 0:
            cutoff = np.append(cutoff, 1.0)
            # And refrain from displaying a QMessageBox
        
        for firf in self.firfs:
            firf.compute_firwin(cutoff=cutoff, pass_zero=pass_zero)
        self.update_filter_design()
    
    def construct_cutoff_list(self):
        if self.firfs[0].pass_zero:
            # Input coupling: DC
            cutoff = np.array([])
        else:
            # Input coupling: AC
            cutoff = self.firfs[0].cutoff[0]

        for row in range(self.qtbl_bandstop.rowCount()):
            for col in range(self.qtbl_bandstop.columnCount()):
                value = self.qtbl_bandstop.item(row, col).text()
                try:
                    value = float(value)
                    value = round(value*10)/10
                    cutoff = np.append(cutoff, value)
                except ValueError:
                    value = None
        
        # cutoff list cannot be empty. Force at least one value in the list
        if np.size(cutoff) == 0:
            cutoff = np.append(cutoff, 1.0)
            
            msg = QtGui.QMessageBox()
            msg.setIcon(QtGui.QMessageBox.Information)
            msg.setWindowTitle("FIR filter design")
            msg.setText("The list cannot be empty when input coupling is set "
                        "to DC.<br>Putting 1.0 Hz back into the list.<br><br>"
                        "You could set the input coupling to AC first.")
            msg.setStandardButtons(QtGui.QMessageBox.Ok)
            msg.exec_()
        
        for firf in self.firfs:
            firf.compute_firwin(cutoff=cutoff)
    
    def qtbl_bandstop_KeyPressEvent(self, event):
        # Handle special events
        if event.key() == QtCore.Qt.Key_Delete:
            msg = QtGui.QMessageBox()
            msg.setIcon(QtGui.QMessageBox.Information)
            msg.setWindowTitle("FIR filter design")
            msg.setText("Are you sure you want to remove the<br>"
                        "following frequencies from the list?")
            list_freqs = np.array([])
            for item in self.qtbl_bandstop.selectedItems():
                if item.text() != '':
                    list_freqs = np.append(list_freqs, item.text())
            str_freqs = ", ".join(map(str, list_freqs))
            msg.setInformativeText("%s [Hz]" % str_freqs)
            msg.setStandardButtons(QtGui.QMessageBox.Yes |
                                   QtGui.QMessageBox.No)
            answer = msg.exec_()
            
            if answer == QtGui.QMessageBox.Yes:
                self.qtbl_bandstop_cellChanged_lock = True
                for item in self.qtbl_bandstop.selectedItems():
                    item.setText('')
                self.qtbl_bandstop_cellChanged_lock = False
    
                self.construct_cutoff_list()
                self.update_filter_design()
        
        # Regular event handling for QTableWidgets
        return QtGui.QTableWidget.keyPressEvent(self.qtbl_bandstop, event)
    
    @QtCore.pyqtSlot(int, int)
    def process_qtbl_bandstop_cellChanged(self, k, l):
        if self.qtbl_bandstop_cellChanged_lock:
            return
        #print("cellChanged %i %i" % (k, l))
        
        self.construct_cutoff_list()
        self.update_filter_design()
        
    def update_filter_design(self):
        self.populate_design_controls()
        self.signal_filter_design_updated.emit()