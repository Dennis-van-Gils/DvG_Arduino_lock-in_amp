# DvG_Arduino_lock-in_amp
_Work in progress:_ A lock-in amplifier running on an Arduino M0 Pro or M4 Metro/Feather/ItsyBitsy Express microcontroller board in combination with a PC running Python.

The Arduino microcontroller board generates the reference signal REF_X and subsequently acquires the input signal SIG_I. This data is sent over USB to a PC running the main graphical user interface in Python. The Python program shows the waveform graphs of the signals in real-time, performs the heterodyne mixing and filtering of the signals similar to a lock-in amplifier, and provides logging to disk.

Current specs Arduino:
- Support for SAMD21 or SAMD51 chipsets
- True analog-out waveform generator (REF_X between 0 to 3.3 V)
- Differential analog-in data acquisition (SIG_I between -3.3 to 3.3 V)
- ADC & DAC operate at 5 kHz sampling rate
- Double-buffered binary-data transmission over USB to a PC running Python

Current specs Python:
- Separate threads for communication with the Arduino, signal processing and graphing
- Zero-phase distortion FIR filters
- Automatic detection of the Arduino by scanning over all COM ports
- Optional OpenGL hardware-accelerated graphing (requires stencil buffer support on the GPU)

### Prerequisites
Python 3.7\
Preferred distribution: Anaconda
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/intro) (comes with Anaconda)
- [NumPy](http://www.numpy.org/) (comes with Anaconda)
- [SciPy](http://www.scipy.org/) (comes with Anaconda)
- [pyqtgraph](http://www.pyqtgraph.org/documentation/)
- [pyserial](https://pythonhosted.org/pyserial/)
- Optional [PyOpenGL](http://pyopengl.sourceforge.net/)

In Anaconda Prompt:
```
conda install -c anaconda pyserial
conda install -c anaconda pyqtgraph
conda install -c anaconda pyopengl
```

### Pin-out
```
A0: analog out, REF_X with respect to AGND
A1: differential analog in, SIG_I+
A2: differential analog in, SIG_I-
```


###### Dennis van Gils, 02-04-2019
