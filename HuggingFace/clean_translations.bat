@echo off
REM Clean English translations from answer files
cd /d %~dp0
py clean_translations.py
pause

