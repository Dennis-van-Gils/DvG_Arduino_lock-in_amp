@echo off
echo;
echo ---------------------------------------------------------
echo   Port should be set to the port visible when running the
echo   M4 in boot mode.
echo ---------------------------------------------------------
echo;


set port=COM3
set tool=%UserProfile%\.platformio\packages\tool-bossac\bossac.exe
set firmware=bootloader-feather_m4-v3.13.0.bin

REM %tool% -p %port% -e -w -v -R --offset=0x4000 %firmware%
%tool% --info --debug --port %port% --write --verify --reset -U --offset 0x4000 %firmware%