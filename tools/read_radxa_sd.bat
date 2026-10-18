@echo off
rem Read-only: copies the Radxa's Wi-Fi settings and logs off its SD card (USB reader = PhysicalDrive1) into data\radxa_sd.
rem Right-click this file -> "Run as administrator".
py -3.9 "%~dp0ext4_extract.py" \\.\PhysicalDrive1 --offset-mb 16 --out "%~dp0..\data\radxa_sd" /etc/NetworkManager /etc/netplan /var/lib/NetworkManager /boot/armbianEnv.txt /etc/hostname /etc/modprobe.d /etc/modules-load.d /var/log.hdd /var/log
echo.
pause
