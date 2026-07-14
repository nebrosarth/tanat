@echo off
REM Tanat server -- WTF MODE: avatar skills cost no mana and never go on cooldown.
setlocal
cd /d "%~dp0"
set "TANAT_HUNT_START_LEVEL="
set "TANAT_WTF_MODE=1"
echo ================================================
echo  Tanat server: WTF MODE (no mana, no cooldown)
echo ================================================
ctrlserver.exe %*
