@echo off
echo;
echo ---------------------------------------------------------
echo   Connect to the programming port of the M0.
echo ---------------------------------------------------------
echo;

edbg -b -t samd21 -pv -f firmware.bin

REM Alternative flash
REM Before flashing make sure to erase the chip and to set
REM the NVMCTRL_BOOTPROT fuse to 0 bytes (EDBG tool).

REM set tool=%UserProfile%\.platformio\packages\tool-openocd\bin\openocd.exe
REM set scripts=%UserProfile%\.platformio\packages\tool-openocd\scripts
REM set firmware=firmware.bin

REM %tool% -d2 -s %scripts% -f interface/cmsis-dap.cfg -c "set CHIPNAME at91samd21g18" -f target/at91samdXX.cfg -c "program {%firmware%} 0x0000 verify reset; shutdown;"