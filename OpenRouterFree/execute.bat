@echo off
REM Batch file to execute the OpenRouter Free script

echo ============================================================
echo OpenRouter Free Models - Execution Script
echo ============================================================
echo.

REM Check if OPENROUTER_API_KEY is set
if "%OPENROUTER_API_KEY%"=="" (
    echo [ERROR] OPENROUTER_API_KEY environment variable is not set!
    echo Please set it using: set OPENROUTER_API_KEY=your_api_key_here
    echo.
    echo You can get a free API key from: https://openrouter.ai/
    echo.
    pause
    exit /b 1
)

echo [INFO] API Key detected: %OPENROUTER_API_KEY:~0,10%...
echo.

REM Activate virtual environment if needed (uncomment if you use venv)
REM call ..\venv\Scripts\activate.bat

REM Execute the script
echo [INFO] Starting script execution...
echo.
python script.py

echo.
echo ============================================================
echo [INFO] Execution completed!
echo ============================================================
pause

