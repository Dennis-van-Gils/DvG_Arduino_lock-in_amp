@echo off

REM Port should be set to the port visible when running M4 in boot mode
set port=COM7

set tool=%UserProfile%\.platformio\packages\tool-bossac\bossac.exe
set firmware=firmware.bin

%tool% --info --debug --port %port% --write --verify --reset -U --offset 0x4000 %firmware%