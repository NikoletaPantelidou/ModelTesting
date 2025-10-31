@echo off
echo ========================================
echo Comparador de Respuestas de Modelos
echo ========================================
echo.

REM Check if pandas is installed
python -c "import pandas" 2>nul
if errorlevel 1 (
    echo Installing pandas...
    pip install pandas
)

REM Run comparison
if "%1"=="" (
    echo Buscando directorio 'answers' automaticamente...
    python compare_models.py
) else (
    echo Usando directorio: %1
    python compare_models.py %1
)

echo.
echo ========================================
echo Comparacion completada!
echo ========================================
echo.
echo Archivos generados:
echo   - comparison.csv         (Comparacion completa)
echo   - summary_report.txt     (Reporte resumido)
echo.
echo Puedes abrir estos archivos con Excel o cualquier editor.
echo.
pause

