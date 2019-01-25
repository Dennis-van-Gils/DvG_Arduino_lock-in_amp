# DvG_Arduino_lock-in_amp
Work in progress: a basic lock-in amplifier running on an Arduino M0 Pro.

Prerequisites:
- PyQt5 (pre-installed in Anaconda)
- pyqtgraph (not installed)
- numpy (pre-installed in Anaconda)
- collections (not sure if pre-installed in Anaconda)

In Anaconda prompt:\
conda install pyqtgraph


Pin-out:\
A0: analog out, REF_X, with respect to GND\
A1: + differential in\
A2: - differential in
