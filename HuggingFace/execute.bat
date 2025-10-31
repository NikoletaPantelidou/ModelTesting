@echo off
:: Verifica si se está ejecutando como administrador
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Solicitando permisos de administrador...
    powershell -Command "Start-Process -FilePath '%~f0' -Verb runAs"
    exit /b
)

:: Cambiar al directorio donde está este .bat (funciona entre unidades)
cd /d "%~dp0"

:: Ejecuta tu script de Python (siempre desde el directorio del .bat)
py script.py

pause
