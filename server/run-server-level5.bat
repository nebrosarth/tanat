@echo off
REM Tanat server -- spawn hunt avatars at level 5 (ult reachable immediately).
setlocal
cd /d "%~dp0"
set "TANAT_HUNT_START_LEVEL=5"
set "TANAT_WTF_MODE="
echo ================================================
echo  Tanat server: START LEVEL 5
echo ================================================
ctrlserver.exe %*
