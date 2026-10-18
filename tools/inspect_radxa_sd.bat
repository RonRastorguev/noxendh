@echo off
rem Read-only: prints the Radxa SD card's filesystem UUID and where /boot/armbianEnv.txt's data sits (USB reader = PhysicalDrive1).
rem Right-click this file -> "Run as administrator".
py -3.9 "%~dp0ext4_extract.py" \\.\PhysicalDrive1 --offset-mb 16 --out "%~dp0..\data\radxa_sd_inspect" --inspect /boot/armbianEnv.txt --inspect /boot/boot.scr
echo.
pause
