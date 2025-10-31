# Estructura del Proyecto

```
PyCharmMiscProject/
│
├── COMPARISON.md                      # Comparación HuggingFace vs Ollama
│
├── huggingFace/                       # 🖥️ IMPLEMENTACIÓN LOCAL
│   ├── script.py                      # Script principal (GPU)
│   ├── script_models_config.py        # Config de modelos HF
│   ├── execute.bat                    # Ejecutar script
│   ├── clean_cache.bat               # Limpiar caché
│   ├── check_gpu.py                  # Verificar GPU
│   ├── requirements.txt              # torch, transformers, etc.
│   │
│   ├── prompts/
│   │   └── example.csv               # Entrada de prompts
│   │
│   ├── answers/
│   │   ├── distilbert-*_answers.csv
│   │   ├── mistral-*_answers.csv
│   │   └── ...                       # Respuestas por modelo
│   │
│   └── logs/
│       └── execution_*.log           # Logs de ejecución
│
└── Ollama/                           # ☁️ IMPLEMENTACIÓN CLOUD
    ├── script.py                     # Script principal (API)
    ├── script_models_config.py       # Config de modelos Ollama
    ├── test_connection.py            # Test de conexión API
    ├── execute.bat                   # Ejecutar script
    ├── setup.bat                     # Configuración inicial
    ├── clean_cache.bat              # Limpiar caché
    ├── requirements.txt             # requests, pandas
    ├── README.md                    # Documentación completa
    ├── QUICKSTART.md                # Inicio rápido
    ├── .gitignore                   # Archivos a ignorar
    │
    ├── prompts/
    │   └── example.csv              # Entrada de prompts
    │
    ├── answers/
    │   ├── llama3.2-latest_answers.csv
    │   ├── mistral-latest_answers.csv
    │   └── ...                      # Respuestas por modelo
    │
    └── logs/
        └── execution_*.log          # Logs de ejecución
```

## 📁 Descripción de Archivos

### Archivos Principales

| Archivo | Descripción | Ubicación |
|---------|-------------|-----------|
| `script.py` | Motor principal de procesamiento | HF y Ollama |
| `script_models_config.py` | Configuración de modelos a usar | HF y Ollama |
| `execute.bat` | Script de ejecución rápida | HF y Ollama |
| `requirements.txt` | Dependencias Python | HF y Ollama |

### Archivos de Configuración

| Archivo | Descripción | Ubicación |
|---------|-------------|-----------|
| `setup.bat` | Setup inicial + instalación | Solo Ollama |
| `check_gpu.py` | Verificar GPU disponible | Solo HF |
| `test_connection.py` | Test API connection | Solo Ollama |
| `clean_cache.bat` | Limpiar archivos cache | HF y Ollama |

### Documentación

| Archivo | Descripción | Ubicación |
|---------|-------------|-----------|
| `README.md` | Documentación completa | Solo Ollama |
| `QUICKSTART.md` | Guía de inicio rápido | Solo Ollama |
| `COMPARISON.md` | Comparación HF vs Ollama | Raíz |

### Directorios de Datos

| Directorio | Contenido | Propósito |
|------------|-----------|-----------|
| `prompts/` | Archivos CSV con prompts | Entrada del sistema |
| `answers/` | Archivos CSV con respuestas | Salida del sistema |
| `logs/` | Logs de ejecución | Debug y monitoreo |
| `__pycache__/` | Cache de Python | Temporal (auto-generado) |
| `.venv/` | Entorno virtual | Aislamiento de deps |

## 🔄 Flujo de Datos

```
prompts/example.csv
        ↓
  [script.py]
   ├─ Model 1 → answers/model1_answers.csv
   ├─ Model 2 → answers/model2_answers.csv
   └─ Model N → answers/modelN_answers.csv
        ↓
   logs/execution_[timestamp].log
```

## 📊 Formato de Archivos

### Input (prompts/example.csv)
```csv
prompt;test_item
"Context text, question?";test_1
"Another context, question?";test_2
```

### Output (answers/model_answers.csv)
```csv
prompt;test_item;answer
"Context text, question?";test_1;"The answer is..."
"Another context, question?";test_2;"Another answer..."
```

### Logs (logs/execution_*.log)
```
2025-10-31 19:27:00 - INFO - [INFO] Loading model: llama3.2:latest
2025-10-31 19:27:05 - INFO - [INFO] Processing row 1/10
2025-10-31 19:27:10 - INFO - [OK] Row 1 completed
```

## 🚀 Comandos Rápidos

### HuggingFace (Local)
```bash
cd huggingFace
execute.bat
```

### Ollama (Cloud)
```bash
cd Ollama
set OLLAMA_API_KEY=your_key_here
execute.bat
```

## 📝 Variables de Entorno

### HuggingFace
- `HF_TOKEN` - Token de HuggingFace (en script.py)

### Ollama
- `OLLAMA_API_KEY` - API key de Ollama Cloud (requerida)

## 🔧 Archivos de Configuración Clave

### HuggingFace: script.py (líneas 50-56)
```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEMPERATURE = 0.0
OUTPUT_DIR = "answers"
MAX_WORKERS = 4
```

### Ollama: script.py (líneas 21-29)
```python
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")
OLLAMA_API_BASE = "https://api.ollama.com/v1"
MAX_WORKERS = 2
TEMPERATURE = 0.0
MAX_TOKENS = 300
```

## 📦 Dependencias

### HuggingFace
- torch (PyTorch con CUDA)
- transformers
- pandas
- huggingface_hub

### Ollama
- requests
- pandas

## 🎯 Cuándo Usar Cada Carpeta

| Escenario | Usa | Razón |
|-----------|-----|-------|
| Tengo GPU potente | `huggingFace/` | Mejor rendimiento local |
| No tengo GPU | `Ollama/` | No requiere GPU |
| Datos sensibles | `huggingFace/` | Todo queda local |
| Prototipo rápido | `Ollama/` | Setup en minutos |
| Alto volumen | `huggingFace/` | Sin costos de API |
| Bajo volumen | `Ollama/` | Pago por uso |

## 🔐 .gitignore

Ambas carpetas ignoran:
- `__pycache__/`
- `.venv/`
- `logs/` (opcionales)
- `answers/` (opcional)

## 📌 Notas Importantes

1. **Los archivos CSV son compatibles** entre HF y Ollama
2. **Los modelos NO son intercambiables** (diferente naming)
3. **Ambos soportan paralelización** con ThreadPoolExecutor
4. **Ambos soportan reinicio automático** (skip completados)
5. **Los logs tienen el mismo formato** para facilitar debugging

## 🆘 Ayuda Rápida

```bash
# HuggingFace
cd huggingFace
python check_gpu.py          # Verificar GPU

# Ollama
cd Ollama
python test_connection.py   # Verificar API
python script.py --help     # Ver opciones
```

## 📚 Más Información

- Ver `Ollama/README.md` para detalles de Ollama Cloud
- Ver `Ollama/QUICKSTART.md` para inicio rápido
- Ver `COMPARISON.md` para comparación detallada

