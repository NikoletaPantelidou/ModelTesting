@echo off
echo ========================================
echo Executing Ollama API Script
echo ========================================
echo.

echo Checking internet connection...
ping -n 1 ollama.com >nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: Cannot reach ollama.com
    echo Please check your internet connection
    echo.
    echo The script may fail if the API is not accessible
    echo.
    pause
)

echo Internet connection OK
echo.

REM Activate virtual environment if it exists
if exist .venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
    echo.
)

echo Starting script execution...
echo.
python script.py

if errorlevel 1 (
    echo.
    echo ========================================
    echo Script execution failed!
    echo ========================================
    pause
    exit /b 1
) else (
    echo.
    echo ========================================
    echo Script execution completed successfully!
    echo ========================================
)

pause

