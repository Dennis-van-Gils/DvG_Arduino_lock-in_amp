# DvG_Arduino_lock-in_amp
Work in progress: a basic lock-in amplifier running on an Arduino M0 Pro.

Current specs:
- waveform generator and DAQ running at 10 kHz sampling rate

### Prerequisites
Python 3.7\
Additional packages:
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/intro) (pre-installed in Anaconda)
- [numpy](http://www.numpy.org/) (pre-installed in Anaconda)
- [pyqtgraph](http://www.pyqtgraph.org/documentation/)
- [pyserial](https://pythonhosted.org/pyserial/)

In Anaconda Prompt:
```
conda install pyqtgraph
conda install pyserial
```

### Pin-out
```
A0: analog out, REF_X with respect to AGND
A1: differential analog in, SIG_I+
A2: differential analog in, SIG_I-
```
