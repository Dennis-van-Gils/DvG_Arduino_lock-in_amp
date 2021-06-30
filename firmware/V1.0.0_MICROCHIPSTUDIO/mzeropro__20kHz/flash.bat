@echo off
echo;
echo ---------------------------------------------------------
echo   Connect to the programming port of the M0
echo   Before flashing make sure to erase the chip and to set
echo   the NVMCTRL_BOOTPROT fuse to 0 bytes (EDBG tool)
echo ---------------------------------------------------------
echo;


set tool=%UserProfile%\.platformio\packages\tool-openocd\bin\openocd.exe
set scripts=%UserProfile%\.platformio\packages\tool-openocd\scripts
set firmware=LIA_SAMD21.bin

%tool% -d2 -s %scripts% -f interface/cmsis-dap.cfg -c "set CHIPNAME at91samd21g18" -f target/at91samdXX.cfg -c "program {%firmware%} 0x0000 verify reset; shutdown;"