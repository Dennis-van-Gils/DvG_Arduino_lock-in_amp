# DvG Arduino lock-in amplifier
This project concerns a lock-in amplifier running on an Atmel SAMD21 (Arduino M0) or SAMD51 (Adafruit M4 Metro/Feather/ItsyBitsy Express) microcontroller board in combination with a PC running Python 3.8.

The Arduino microcontroller will generate the reference signal `REF_X*` and subsequently acquires the input signal `SIG_I`. This data is sent over USB to a PC running the graphical user interface in Python. The Python program shows the waveform graphs of the signals in real-time, performs the heterodyne mixing and filtering of the signals similar to a lock-in amplifier, and provides logging to disk.

![Screenshot](screenshots/tab_1.PNG)
See [here](screenshots/) for more screenshots.

### User manual
A [user manual](user_manual/DvG_ALIA_Student_user_manual.pdf) with detailed installation instructions and troubleshooting is provided. The manual is part of the lab assignments of the course 'Small Signals & Detection' of the University of Twente, Enschede, The Netherlands.

### Specifications
Microcontroller:
- Support for Atmel SAMD21 or SAMD51 chipsets
- The ADC & DAC operate at sampling rates of 10 kHz and above
- True analog-out waveform generator (`REF_X*` between 0 to 3.3 V)
- Two modes available for the analog-in data acquisition. Determined by a flag set in the firmware:
    - Single-ended (`SIG_I` between 0 to 3.3 V), default
    - Differential (`SIG_I` between -3.3 to 3.3 V)
- Double-buffered binary-data transmission over USB to a PC running Python

Python program:
- Separate threads for communication with the Arduino, signal processing and graphing
- Accelerated mathematical operations based on [pyFFTW](https://pyfftw.readthedocs.io/en/latest/), [Numba](https://numba.pydata.org/) and [SciPy](https://scipy.org/)
- OpenGL hardware-accelerated graphing
- Zero-phase distortion FIR filters
- Scans over all serial ports to automatically connect to the Arduino
- Tested under Windows 10 and Ubuntu 18.04.2 (Linux)

### Prerequisites
Python 3.8\
Preferred distribution: Anaconda full or Miniconda

Installation instructions in Anaconda Prompt:
```
conda update -n base -c defaults conda
conda create -n lia -c conda-forge  --force -y python=3.8.10
conda activate lia
pip install -r requirements.txt
```

Precompiled firmware for an Adafruit Feather M4 Express running at a 25 kHz sampling rate with single-ended analog input is available at ![CURRENT.UF2](https://github.com/Dennis-van-Gils/DvG_Arduino_lock-in_amp/raw/master/mcu_firmware/v1.0.0_VSCODE/adafruit_feather_m4__25kHz/CURRENT.UF2).
You can copy over this firmware to the M4 board by using the FEATHERBOOT mount drive.

### Pin-out
```
    - A0 : Output reference signal `ref_X*`, single-ended with respect to GND

    When `ADC_DIFFERENTIAL` = 0, default
    ------------------------------------------
    - A1 : Input signal `sig_I`, single-ended with respect to GND
    - A2 : Not used

    When `ADC_DIFFERENTIAL` = 1
    ------------------------------------------
    - A1 : Input signal `sig_I`, differential(+)
    - A2 : Input signal `sig_I`, differential(-)

    - D12: A digital trigger-out signal that is in sync with every full period
           of `ref_X*`, useful for connecting up to an oscilloscope.
```

###### Dennis van Gils, 03-11-2021
