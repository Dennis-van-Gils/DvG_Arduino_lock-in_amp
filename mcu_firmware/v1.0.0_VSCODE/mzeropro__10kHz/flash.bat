@echo off
echo;
echo ---------------------------------------------------------
echo   Connect to the programming port of the M0.
echo ---------------------------------------------------------
echo;

edbg -b -t samd41 -pv -o 0x4000 -f firmware.bin


REM Alternative flash
REM Connect to the native port of the M0.
REM Port should be set to the port visible when running the
REM M0 in boot mode.

REM set port=COM13
REM set tool=%UserProfile%\.platformio\packages\tool-avrdude\avrdude.exe
REM set conf=%UserProfile%\.platformio\packages\tool-avrdude\avrdude.conf
REM set firmware=firmware.hex

REM %tool% -v -p atmega2560 -C %conf% -c stk500v2 -P %port% -b 57600 -u -U flash:w:%firmware%:i
