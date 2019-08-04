@echo off
if exist %USERPROFILE%\Anaconda3\Scripts\activate.bat (
  %windir%\System32\cmd.exe "/K" %USERPROFILE%\Anaconda3\Scripts\activate.bat %USERPROFILE%\Anaconda3\envs\lia
)
if exist %PROGRAMDATA%\Anaconda3\Scripts\activate.bat (
  %windir%\System32\cmd.exe "/K" %PROGRAMDATA%\Anaconda3\Scripts\activate.bat %PROGRAMDATA%\Anaconda3\envs\lia
)