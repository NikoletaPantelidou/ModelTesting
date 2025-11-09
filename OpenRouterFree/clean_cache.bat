@echo off
REM Batch file to clear cache files

echo ============================================================
echo OpenRouter Free - Clear Cache
echo ============================================================
echo.

echo [INFO] Cleaning Python cache files...
if exist __pycache__ (
    rmdir /S /Q __pycache__
    echo [OK] __pycache__ directory removed
) else (
    echo [INFO] No __pycache__ directory found
)

if exist *.pyc (
    del /Q *.pyc
    echo [OK] .pyc files removed
) else (
    echo [INFO] No .pyc files found
)

echo.
echo [OK] Cache cleanup completed!
pause

