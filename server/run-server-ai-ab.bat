@echo off
REM Launch a bot-filled Assault match with AI-30 on both teams.
setlocal
cd /d "%~dp0"

set "TANAT_DOTA_BOTS=10"
set "TANAT_DOTA_BOT_AI_TEAM1=AI-30"
set "TANAT_DOTA_BOT_AI_TEAM2=AI-30"

echo ================================================
echo  Tanat server: AI-30 (team 1) vs AI-30 (team 2)
echo  Assault bots: %TANAT_DOTA_BOTS% total participants
echo ================================================
ctrlserver.exe %*
