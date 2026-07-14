@echo off
REM Tanat server -- WTF MODE + spawn at level 5 (the full "test everything" combo).
setlocal
cd /d "%~dp0"
set "TANAT_HUNT_START_LEVEL=5"
set "TANAT_WTF_MODE=1"
echo ================================================
echo  Tanat server: WTF MODE + START LEVEL 5
echo ================================================
ctrlserver.exe %*
