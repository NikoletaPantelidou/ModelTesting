@echo off
echo ========================================
echo Executing Ollama Cloud API Script
echo ========================================
echo.

REM Check if API key is set
if "%OLLAMA_API_KEY%"=="" (
    echo ERROR: OLLAMA_API_KEY environment variable is not set!
    echo.
    echo Please set your Ollama API key:
    echo   set OLLAMA_API_KEY=your_api_key_here
    echo.
    echo Or permanently:
    echo   setx OLLAMA_API_KEY "your_api_key_here"
    echo.
    echo Get your API key from: https://ollama.com/
    echo.
    pause
    exit /b 1
)

echo API Key: [SET]
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

