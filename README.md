# DvG_Arduino_lock-in_amp
Work in progress: a basic lock-in amplifier running on an Arduino M0 Pro.

### Prerequisites
Python 3.7\
Additional packages:
'''
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/intro) (pre-installed in Anaconda)
- [numpy](http://www.numpy.org/) (pre-installed in Anaconda)
- [pyqtgraph](http://www.pyqtgraph.org/documentation/)
- [pyserial](https://pythonhosted.org/pyserial/)
'''

In Anaconda Prompt:\
'''
conda install pyqtgraph
conda install pyserial
'''

### Pin-out\
A0: analog out, REF_X, with respect to GND\
A1: + differential in\
A2: - differential in
