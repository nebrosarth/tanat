@echo off
REM Launch a bot-filled Assault match with independent team AI profiles.
setlocal
cd /d "%~dp0"

set "TANAT_DOTA_BOTS=10"
set "TANAT_DOTA_BOT_AI_TEAM1=AI-0"
set "TANAT_DOTA_BOT_AI_TEAM2=AI-20"

echo ================================================
echo  Tanat server: AI-0 (team 1) vs AI-20 (team 2)
echo  Assault bots: %TANAT_DOTA_BOTS% total participants
echo ================================================
ctrlserver.exe %*
