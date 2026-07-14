@echo off
REM Tanat server -- interactive: pick the debug flags at launch.
setlocal
cd /d "%~dp0"

set /p "LVL=Start level (blank = 1): "
set /p "WTF=WTF mode - no mana/cooldown? (y/N): "

if defined LVL ( set "TANAT_HUNT_START_LEVEL=%LVL%" ) else ( set "TANAT_HUNT_START_LEVEL=" )

set "TANAT_WTF_MODE="
if /i "%WTF%"=="y"   set "TANAT_WTF_MODE=1"
if /i "%WTF%"=="yes" set "TANAT_WTF_MODE=1"
if "%WTF%"=="1"      set "TANAT_WTF_MODE=1"

echo ================================================
echo  Tanat server: level=[%TANAT_HUNT_START_LEVEL%] wtf=[%TANAT_WTF_MODE%]
echo ================================================
ctrlserver.exe %*
