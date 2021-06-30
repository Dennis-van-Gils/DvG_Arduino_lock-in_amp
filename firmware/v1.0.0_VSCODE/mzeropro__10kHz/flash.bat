@echo off
echo;
echo ---------------------------------------------------------
echo   Connect to the native port of the M0.
echo   Port should be set to the port visible when running the
echo   M0 in boot mode.
echo ---------------------------------------------------------
echo;


set port=COM13
set tool=%UserProfile%\.platformio\packages\tool-avrdude\avrdude.exe
set conf=%UserProfile%\.platformio\packages\tool-avrdude\avrdude.conf
set firmware=firmware.hex

%tool% -v -p atmega2560 -C %conf% -c stk500v2 -P %port% -b 57600 -u -U flash:w:%firmware%:i
