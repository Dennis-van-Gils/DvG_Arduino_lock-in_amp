# DvG_Arduino_lock-in_amp
A lock-in amplifier running on a SAMD21 (Arduino M0) or SAMD51 (Adafruit M4 Metro/Feather/ItsyBitsy Express) microcontroller board in combination with a PC running Python 3.8, confirmed to work in both Windows 10 and Ubuntu 18.04.2 (Linux).

The Arduino microcontroller board generates the reference signal `REF_X*` and subsequently acquires the input signal `SIG_I`. This data is sent over USB to a PC running the main graphical user interface in Python. The Python program shows the waveform graphs of the signals in real-time, performs the heterodyne mixing and filtering of the signals similar to a lock-in amplifier, and provides logging to disk.

![Screenshot](screenshots/tab_1.PNG)
See [here](screenshots/) for more screenshots.

Current specs microcontroller firmware:
- Support for SAMD21 or SAMD51 chipsets
- True analog-out waveform generator (`REF_X*` between 0 to 3.3 V)
- Two modes available for the analog-in data-acquisition. Determined by a flag set in the firmware:
    - Single-ended (`SIG_I` between 0 to 3.3 V)
    - Differential (`SIG_I` between -3.3 to 3.3 V)
- ADC & DAC operate at sampling rates of 10 kHz and above
- Double-buffered binary-data transmission over USB to a PC running Python

Current specs Python:
- Separate threads for communication with the Arduino, signal processing and graphing
- Zero-phase distortion FIR filters
- Automatic detection of the Arduino by scanning over all COM ports
- OpenGL hardware-accelerated graphing
- Tested under Windows 10 and Ubuntu 18.04.2 (Linux)

### Prerequisites
Python 3.8\
Preferred distribution: Anaconda full or Miniconda

In Anaconda Prompt:
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

    When `ADC_DIFFERENTIAL` = 0
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

###### Dennis van Gils, 23-07-2021
