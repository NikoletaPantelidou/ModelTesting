@echo off
echo ========================================
echo Executing Ollama Local Script
echo ========================================
echo.

REM Check if Ollama is running
echo Checking if Ollama is running...
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
if errorlevel 1 (
    echo ERROR: Ollama is not running!
    echo Please start Ollama first:
    echo   ollama serve
    echo   ollama serve
    echo.
    echo Or if Ollama is already running in background,
    echo make sure it's accessible at http://localhost:11434
    echo make sure it's accessible at http://localhost:11434
    echo.
    pause
    exit /b 1
)
echo Ollama is running!
echo Ollama is running!
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

