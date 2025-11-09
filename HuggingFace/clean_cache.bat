@echo off
echo ========================================
echo Limpiando cache de HuggingFace
echo ========================================
echo.

set CACHE_DIR=%USERPROFILE%\.cache\huggingface

if exist "%CACHE_DIR%" (
    echo Cache encontrado en: %CACHE_DIR%
    echo.
    echo ADVERTENCIA: Esto eliminara todos los modelos descargados.
    echo Tendras que descargarlos nuevamente la proxima vez.
    echo.
    choice /C SN /M "¿Estas seguro de que quieres continuar?"

    if errorlevel 2 (
        echo.
        echo Operacion cancelada.
        pause
        exit /b
    )

    echo.
    echo Eliminando cache...
    rd /s /q "%CACHE_DIR%"
    echo ✓ Cache eliminado exitosamente
) else (
    echo No se encontro cache en: %CACHE_DIR%
)

echo.
pause

