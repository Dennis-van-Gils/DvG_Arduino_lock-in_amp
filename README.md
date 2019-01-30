# DvG_Arduino_lock-in_amp
Work in progress: a lock-in amplifier running on an Arduino M0 Pro using the Atmel SAMD21 chipset.

The Arduino microcontroller board generates the reference signal REF_X and subsequently acquires the input signal SIG_I. This data is sent over USB to a PC running the main graphical user interface in Python. The Python program shows the waveform graphs of the signals in real-time, performs the heterodyne mixing and filtering of the signals similar to a lock-in amplifier, and provides logging to disk.

Current specs Arduino:
- True analog-out waveform generator (REF_X between 0 to 3 V)
- Differential analog-in data acquisition (SIG_I between -3 to 3 V)
- ADC & DAC operate at 10 kHz sampling rate
- Double-buffered binary-data transmission over USB to a PC running Python

Current specs Python:
- Separate threads for communication with the Arduino, signal processing and graphing
- Zero-phase distortion FIR filter
- Optional OpenGL hardware-accelerated graphing

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
