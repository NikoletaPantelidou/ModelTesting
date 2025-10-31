# 🚀 Guía de Instalación Completa

## Opción 1: Ollama Cloud (Recomendado para Empezar)

### ✅ Ventajas
- ⚡ Setup en 5 minutos
- 💻 No requiere GPU
- 🎯 Sin problemas técnicos

### 📋 Requisitos
- Python 3.8+
- Conexión a Internet
- API Key de Ollama Cloud

### 🔧 Pasos de Instalación

#### 1. Navegar a la carpeta
```bash
cd C:\Users\manel\PyCharmMiscProject\Ollama
```

#### 2. Ejecutar setup automático
```bash
setup.bat
```

Esto hará:
- Crear entorno virtual
- Instalar pandas y requests
- Mostrar instrucciones para API key

#### 3. Obtener API Key
1. Visita: https://ollama.com/cloud
2. Crea cuenta/inicia sesión
3. Genera tu API key

#### 4. Configurar API Key

**Opción A - Temporal (actual sesión):**
```bash
set OLLAMA_API_KEY=tu_api_key_aqui
```

**Opción B - Permanente:**
```bash
setx OLLAMA_API_KEY "tu_api_key_aqui"
```

#### 5. Verificar instalación
```bash
python test_connection.py
```

Deberías ver:
```
✅ SUCCESS! API is working correctly!
✓ Connection test passed!
```

#### 6. ¡Ejecutar!
```bash
execute.bat
```

---

## Opción 2: HuggingFace Local (Para Usuarios Avanzados)

### ✅ Ventajas
- 🔒 Datos permanecen locales
- 💰 Sin costos de API (después de inversión inicial)
- 🎛️ Control total

### 📋 Requisitos Mínimos
- Python 3.8+
- GPU NVIDIA con 8GB+ VRAM (recomendado)
- CUDA 11.8+ instalado
- 16GB+ RAM
- 50GB+ espacio libre en disco

### 📋 Requisitos Recomendados
- GPU NVIDIA con 12GB+ VRAM (ej: RTX 3060 12GB, RTX 4070, A4000)
- CUDA 12.1+
- 32GB RAM
- 100GB+ espacio libre (SSD)

### 🔧 Pasos de Instalación

#### 1. Verificar/Instalar NVIDIA Drivers

**Verificar instalación:**
```bash
nvidia-smi
```

Deberías ver información de tu GPU.

**Si no funciona:**
1. Descarga drivers desde: https://www.nvidia.com/Download/index.aspx
2. Selecciona tu GPU
3. Instala los drivers
4. Reinicia el PC

#### 2. Instalar CUDA Toolkit

**Verificar si ya está instalado:**
```bash
nvcc --version
```

**Si no está instalado:**
1. Descarga CUDA desde: https://developer.nvidia.com/cuda-downloads
2. Selecciona tu sistema operativo
3. Instala CUDA Toolkit
4. Reinicia el PC

#### 3. Navegar a la carpeta
```bash
cd C:\Users\manel\PyCharmMiscProject\huggingFace
```

#### 4. Crear entorno virtual
```bash
python -m venv .venv
.venv\Scripts\activate
```

#### 5. Actualizar pip
```bash
python -m pip install --upgrade pip
```

#### 6. Instalar PyTorch con CUDA

**Para CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Para CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Para CPU solamente (no recomendado):**
```bash
pip install torch torchvision torchaudio
```

#### 7. Instalar otras dependencias
```bash
pip install transformers pandas huggingface_hub accelerate
```

#### 8. Verificar instalación GPU
```bash
python check_gpu.py
```

Deberías ver:
```
CUDA available: True
CUDA device count: 1
Current CUDA device: 0
CUDA device name: NVIDIA GeForce RTX 3060
```

#### 9. Configurar HuggingFace Token (opcional)

Algunos modelos requieren autenticación:

1. Visita: https://huggingface.co/settings/tokens
2. Crea un token (Read access)
3. En `script.py`, actualiza línea 19:
```python
HF_TOKEN = "tu_token_aqui"
```

#### 10. ¡Ejecutar!
```bash
execute.bat
```

---

## ⚠️ Solución de Problemas

### HuggingFace Local

#### Problema: "CUDA not available"

**Solución 1:** Reinstalar PyTorch con CUDA
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Solución 2:** Verificar drivers NVIDIA
```bash
nvidia-smi
```

**Solución 3:** Verificar variable CUDA_PATH
```bash
echo %CUDA_PATH%
```
Debería mostrar: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1`

#### Problema: "CUDA out of memory"

**Solución 1:** Reducir batch size / workers
En `script.py`:
```python
MAX_WORKERS = 1  # Reducir a 1
```

**Solución 2:** Usar modelos más pequeños
En `script_models_config.py`, comenta modelos grandes:
```python
# {"name": "microsoft/phi-4", ...},  # Comentar modelos grandes
```

**Solución 3:** Usar FP16 (ya configurado)
Ya está usando `torch.float16` para GPU.

#### Problema: "Model download too slow"

**Solución:** Usar proxy HF o mirror:
```bash
set HF_ENDPOINT=https://hf-mirror.com
```

### Ollama Cloud

#### Problema: "OLLAMA_API_KEY not set"

**Solución:**
```bash
set OLLAMA_API_KEY=tu_api_key_aqui
```

Para hacerlo permanente:
```bash
setx OLLAMA_API_KEY "tu_api_key_aqui"
```
Luego reinicia el cmd/PowerShell.

#### Problema: "Invalid API key"

**Solución:**
1. Verifica que la key esté correcta
2. Verifica que la cuenta esté activa
3. Genera una nueva key en https://ollama.com/cloud

#### Problema: "Connection timeout"

**Solución 1:** Verificar internet
```bash
ping api.ollama.com
```

**Solución 2:** Reducir workers
En `script.py`:
```python
MAX_WORKERS = 1
```

**Solución 3:** Aumentar timeout
En `script.py`, línea 95:
```python
response = requests.post(url, json=payload, headers=headers, timeout=120)  # Aumentar a 120
```

#### Problema: "Rate limit exceeded"

**Solución:** Reducir paralelismo
```python
MAX_WORKERS = 1
```

Y añadir delay entre requests en `script.py`, función `call_ollama_api`:
```python
time.sleep(1)  # Esperar 1 segundo entre llamadas
```

---

## 🧪 Verificar Instalación

### HuggingFace
```bash
cd huggingFace
python check_gpu.py
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### Ollama
```bash
cd Ollama
python test_connection.py
python -c "import requests; print(f'Requests: {requests.__version__}')"
```

---

## 📊 Comparación de Setup

| Aspecto | HuggingFace | Ollama |
|---------|-------------|---------|
| Tiempo de instalación | 1-2 horas | 5-10 minutos |
| Dificultad | Alta | Baja |
| Requisitos de hardware | GPU potente | Cualquier PC |
| Tamaño descarga inicial | 5-10GB (PyTorch + CUDA) | < 100MB |
| Requiere conocimientos técnicos | Sí (CUDA, drivers) | No |

---

## 🎯 ¿Cuál Instalar Primero?

### Para Principiantes
**→ Ollama Cloud**
- Rápido de configurar
- Sin problemas técnicos
- Puedes probar el sistema inmediatamente

### Para Usuarios Avanzados
**→ Ambos**
1. Instala Ollama primero (para probar)
2. Luego configura HuggingFace (para producción)

### Para Producción
**Depende de tu caso:**
- **Alto volumen + datos sensibles** → HuggingFace
- **Bajo volumen + necesitas flexibilidad** → Ollama
- **Ambos casos** → Instala ambos y elige según necesidad

---

## 📚 Próximos Pasos

Después de la instalación:

1. ✅ Lee el quickstart correspondiente:
   - Ollama: `Ollama/QUICKSTART.md`
   - HuggingFace: Similar a Ollama pero sin API key

2. ✅ Ejecuta con datos de ejemplo

3. ✅ Revisa resultados en `answers/`

4. ✅ Lee `COMPARISON.md` para entender diferencias

5. ✅ Ajusta configuración según tus necesidades

---

## 🆘 ¿Necesitas Ayuda?

### Documentación
- `PROJECT_STRUCTURE.md` - Estructura completa
- `COMPARISON.md` - Comparación detallada
- `Ollama/README.md` - Docs de Ollama
- `Ollama/QUICKSTART.md` - Inicio rápido

### Recursos Online
- **PyTorch + CUDA**: https://pytorch.org/get-started/locally/
- **Ollama Cloud**: https://docs.ollama.com/cloud
- **HuggingFace**: https://huggingface.co/docs

### Verificación Rápida

```bash
# HuggingFace
python check_gpu.py

# Ollama
python test_connection.py
```

---

¡Buena suerte con la instalación! 🚀

