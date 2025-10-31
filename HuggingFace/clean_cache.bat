@echo off
REM Script para eliminar la cache de Hugging Face

SET HF_CACHE=%USERPROFILE%\.cache\huggingface

echo Eliminando la cache de Hugging Face en:
echo %HF_CACHE%

rmdir /s /q "%HF_CACHE%"

echo Cache eliminada.
pause
