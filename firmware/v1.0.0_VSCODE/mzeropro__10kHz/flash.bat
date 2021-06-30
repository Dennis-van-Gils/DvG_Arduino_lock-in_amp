@echo off

REM Port should be set to the native port visible when (re)booting the M0
set port=COM13

set tool=%UserProfile%\.platformio\packages\tool-avrdude\avrdude.exe
set conf=%UserProfile%\.platformio\packages\tool-avrdude\avrdude.conf
set firmware=firmware.hex

%tool% -v -p atmega2560 -C %conf% -c stk500v2 -P %port% -b 57600 -u -U flash:w:%firmware%:i
