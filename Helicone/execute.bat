@echo off
REM Execute Helicone script for processing prompts
REM Make sure OPENAI_API_KEY and HELICONE_API_KEY are set as environment variables

echo ========================================
echo Helicone API Script - Starting
echo ========================================

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install requirements if needed
if not exist "venv\Lib\site-packages\pandas" (
    echo Installing requirements...
    pip install -r ..\requirements.txt
)

REM Run the script
python script.py

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo ========================================
    echo Error occurred during execution
    echo ========================================
    pause
)

REM Deactivate virtual environment
deactivate

