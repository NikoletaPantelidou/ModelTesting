@echo off
echo ========================================
echo Cleaning Ollama Cache
echo ========================================
echo.

REM Clean Python cache
if exist __pycache__ (
    echo Removing __pycache__...
    rmdir /s /q __pycache__
)

REM Clean log files (optional - uncomment if you want to delete logs)
REM if exist logs (
REM     echo Removing logs...
REM     rmdir /s /q logs
REM )

REM Clean answer files (optional - uncomment if you want to delete answers)
REM if exist answers (
REM     echo Removing answers...
REM     rmdir /s /q answers
REM )

echo.
echo ========================================
echo Cache cleaned successfully!
echo ========================================
pause

