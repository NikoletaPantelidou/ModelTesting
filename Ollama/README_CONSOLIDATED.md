# Formato de Respuestas Consolidadas - Ollama

## Descripción

Los scripts de Ollama ahora guardan las respuestas en **dos formatos**:

### 1. Archivos Individuales por Modelo (Compatibilidad)
- **Ubicación**: `answers/[nombre-modelo]_answers.csv`
- **Formato**: Columnas `test_item`, `language`, `prompt`, `answer`
- **Uso**: Mantiene compatibilidad con versiones anteriores

### 2. Archivo Consolidado (NUEVO)
- **Ubicación**: `answers/consolidated_answers.csv`
- **Formato**: Columnas base (`test_item`, `language`, `prompt`) + una columna `answer_[modelo]` por cada modelo
- **Ventaja**: Permite comparar fácilmente las respuestas de todos los modelos en un solo archivo

## Ejemplo de Estructura

### Archivo Individual: `glm-4.6-cloud_answers.csv`
```csv
test_item,language,prompt,answer
B1T,English,"Here's the context...",Yes, it is true...
B1F,English,"Here's the context...",No, it is not true...
```

### Archivo Consolidado: `consolidated_answers.csv`
```csv
test_item,language,prompt,answer_glm-4.6-cloud,answer_llama3.2-latest,answer_qwen3-vl-235b-cloud
B1T,English,"Here's the context...",Yes, it is true...,True...,Affirmative...
B1F,English,"Here's the context...",No, it is not true...,False...,Negative...
```

## Ventajas del Formato Consolidado

1. **Comparación Visual**: Puedes ver las respuestas de todos los modelos lado a lado
2. **Análisis Fácil**: Identifica rápidamente qué test_items tienen respuestas y cuáles no
3. **Seguimiento de Progreso**: Ve el estado de cada modelo en una sola vista
4. **Compatible con Excel/Google Sheets**: Fácil de abrir y analizar

## Scripts Disponibles

### 1. `script.py` - Script Principal
- Procesa prompts usando modelos de Ollama
- Guarda automáticamente en ambos formatos
- Resume desde donde se quedó si se interrumpe

### 2. `check_status.py` - Verificación de Estado
- Muestra el progreso de cada modelo
- Identifica qué test_items faltan
- Genera reportes detallados
- **Uso**: `python check_status.py`

### 3. `check_status.bat` - Ejecutar Verificación
- Ejecuta el script de verificación con un doble clic
- **Uso**: Doble clic en `check_status.bat`

## Comparación con HuggingFace

El formato consolidado de Ollama es **similar** al de HuggingFace:
- HuggingFace: Cada modelo tiene su archivo individual con columna `answer`
- Ollama: Mantiene archivos individuales + archivo consolidado con todas las respuestas

## Cómo Usar

### Ejecutar el Script Principal
```bash
python script.py
```

### Ver Estado de Todas las Respuestas
```bash
python check_status.py
```

O simplemente ejecuta `check_status.bat`

### Abrir el Archivo Consolidado
El archivo `answers/consolidated_answers.csv` puede abrirse con:
- Excel
- Google Sheets
- LibreOffice Calc
- Cualquier editor de CSV

## Reporte de Estado

El script `check_status.py` genera:
- **Consola**: Tabla resumen con estadísticas por modelo
- **Archivo**: `answers/consolidated_status_report.txt` con detalles completos

### Información Incluida
- Total de prompts
- Respuestas completadas por modelo
- Respuestas exitosas vs errores
- Lista de test_items pendientes
- Lista de test_items con errores

## Notas

- Los archivos se actualizan automáticamente después de cada respuesta
- Si se interrumpe la ejecución, se resume desde donde se quedó
- El archivo consolidado siempre tiene la información más reciente
- Los archivos individuales se mantienen para compatibilidad

