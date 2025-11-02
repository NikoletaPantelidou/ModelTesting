# Problema con las respuestas vacías de Qwen

## Resumen del Problema

El modelo **qwen3-vl:235b-cloud** devuelve respuestas **vacías** en el archivo CSV de respuestas.

## Causa del Problema

**qwen3-vl** es un modelo **Vision-Language (VL)**, lo que significa que está diseñado para procesar:
- Imágenes + Texto (multimodal)
- NO solo texto

Cuando se le envía un prompt de solo texto (sin imagen), el modelo:
- Recibe la solicitud exitosamente (código HTTP 200)
- Devuelve una respuesta vacía `""`
- El script guarda esta respuesta vacía en el CSV

## Evidencia

En los logs se puede ver:

```
2025-11-02 15:53:37,903 - INFO - [ANSWER] 
2025-11-02 15:53:37,904 - INFO - [OK] Row 33 [F3F] completed successfully
```

La línea `[ANSWER]` está vacía, sin contenido después.

En el archivo CSV `qwen3-vl-235b-cloud_answers.csv`:

```csv
test_item,language,prompt,answer
B1T,English,"Here's the context: ...",
B1F,English,"Here's the context: ...",
```

La columna `answer` existe pero está vacía para todas las filas.

## Solución

### Opción 1: Usar un modelo Qwen de solo texto (RECOMENDADO)

He actualizado `script_models_config.py` para usar **qwen2.5:72b-cloud** en lugar de qwen3-vl:

```python
MODELS = [
    {"name": "deepseek-v3.1:671b-cloud"},
    {"name": "kimi-k2:1t-cloud"},
    {"name": "qwen2.5:72b-cloud"},  # ✓ Modelo de texto puro
    {"name": "glm-4.6:cloud"},
]
```

Otras alternativas de Qwen para texto puro:
- `qwen2.5:72b-cloud` - Más grande y potente (RECOMENDADO)
- `qwen2.5:32b-cloud` - Mediano
- `qwen2.5:14b-cloud` - Más pequeño y rápido

### Opción 2: Eliminar Qwen de la lista

Si no necesitas un modelo Qwen, simplemente comenta o elimina la línea en `script_models_config.py`.

## Cambios Realizados

### 1. Mejora en el código (`script.py`)

El código ahora detecta respuestas vacías y las marca claramente:

```python
if not answer:
    logger.warning(f"[WARNING] Model {model_name} returned an empty response")
    logger.warning(f"[WARNING] This may indicate the model is incompatible with text-only prompts")
    if "vl" in model_name.lower():
        logger.warning(f"[WARNING] Model appears to be a Vision-Language model - it may require image input")
    return "[EMPTY RESPONSE]"
```

Ahora en lugar de guardar `""` (vacío), guardará `"[EMPTY RESPONSE]"` para que sea visible en el CSV.

### 2. Actualización de configuración (`script_models_config.py`)

- Comentado el modelo problemático `qwen3-vl:235b-cloud`
- Agregado `qwen2.5:72b-cloud` como reemplazo
- Agregados comentarios explicativos

## Próximos Pasos

### Para limpiar y volver a procesar:

1. **Eliminar el archivo de respuestas vacías:**
   ```cmd
   del C:\Users\manel\PyCharmMiscProject\Ollama\answers\qwen3-vl-235b-cloud_answers.csv
   ```

2. **Ejecutar el script nuevamente:**
   ```cmd
   cd C:\Users\manel\PyCharmMiscProject\Ollama
   execute.bat
   ```

3. **Verificar los resultados:**
   - El nuevo modelo `qwen2.5:72b-cloud` debería generar respuestas completas
   - Revisa el archivo `qwen2.5-72b-cloud_answers.csv`

## Modelos Recomendados para Solo Texto

| Modelo | Tipo | Tamaño | Uso |
|--------|------|--------|-----|
| `qwen2.5:72b-cloud` | Texto | Grande | Mejor calidad |
| `qwen2.5:32b-cloud` | Texto | Mediano | Balance |
| `qwen2.5:14b-cloud` | Texto | Pequeño | Más rápido |
| `qwen3-vl:235b-cloud` | **Vision-Language** | Muy grande | ❌ **NO usar para texto puro** |

## Verificación

Para verificar si un modelo es de visión, busca estas palabras clave en el nombre:
- **vl** - Vision-Language
- **vision** - Modelos de visión
- **multimodal** - Soporta múltiples modalidades (imagen + texto)

Estos modelos requieren entrada de imágenes y no funcionarán correctamente con solo texto.

