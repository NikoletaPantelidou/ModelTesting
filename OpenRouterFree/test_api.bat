@echo off
REM Batch file to test OpenRouter API connection

echo ============================================================
echo OpenRouter API Connection Test
echo ============================================================
echo.

REM Check if OPENROUTER_API_KEY is set
if "%OPENROUTER_API_KEY%"=="" (
    echo [WARNING] OPENROUTER_API_KEY environment variable is not set!
    echo.
    echo Please run set_api_key.bat first or set it manually:
    echo set OPENROUTER_API_KEY=your_api_key_here
    echo.
)

python test_api.py

pause

