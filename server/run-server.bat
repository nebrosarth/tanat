@echo off
REM Tanat server -- normal mode (no debug flags).
setlocal
cd /d "%~dp0"
set "TANAT_HUNT_START_LEVEL="
set "TANAT_WTF_MODE="
echo ================================================
echo  Tanat server: NORMAL (start level 1, mana/CD on)
echo ================================================
ctrlserver.exe %*
