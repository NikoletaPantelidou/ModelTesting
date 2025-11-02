# Error 429: Too Many Requests - Helicone

## Problema Detectado

```
Error code: 429 - {'success': False, 'error': {'code': 'request_failed', 
'message': 'Insufficient credits', 
'details': [{'type': 'insufficient_credit_limit', 
'message': 'Insufficient balance for escrow. Available: 0 cents, needed: 1.0290304 cents', 
'statusCode': 429}]}}
```

## Causa del Problema

El error **429 Too Many Requests** en tu caso se debe a **CRÉDITOS INSUFICIENTES** en tu cuenta de Helicone, NO a exceso de peticiones.

- **Disponible**: 0 cents
- **Necesario**: ~1.03 cents por petición

## Solución

### Opción 1: Agregar Créditos a Helicone (RECOMENDADO)

1. Ve a https://www.helicone.ai/
2. Inicia sesión en tu cuenta
3. Ve a **Billing** o **Credits**
4. Agrega créditos a tu cuenta
   - Mínimo recomendado: $5 USD para comenzar
   - Para 72 preguntas × 3 modelos = 216 peticiones ≈ $2-3 USD

### Opción 2: Usar Modelos Más Baratos

En `script_models_config.py`, cambia a modelos más económicos:

```python
MODELS = [
    {"name": "gpt-3.5-turbo"},  # Más barato que gpt-4o-mini
]
```

### Opción 3: Usar un Proveedor Diferente

Si no quieres agregar créditos, considera usar:
- **Ollama** (carpeta `Ollama/`) - Tiene su propia API key y pricing
- **HuggingFace** (carpeta `HuggingFace/`) - Algunos modelos son gratuitos

## Cambios Realizados en el Código

He mejorado el script para manejar mejor este error:

### 1. Detección Clara del Error de Créditos

```python
if "Insufficient credits" in error_msg or "Insufficient balance" in error_msg:
    logger.error(f"[ERROR] ⚠️  INSUFFICIENT CREDITS in Helicone account")
    logger.error(f"[ERROR] Cannot continue - please add credits at https://www.helicone.ai/")
    raise Exception("Insufficient credits in Helicone account. Please add credits to continue.")
```

### 2. Delay Entre Peticiones

Agregado `DELAY_BETWEEN_REQUESTS = 2.0` segundos de espera entre cada petición para evitar rate limits reales:

```python
DELAY_BETWEEN_REQUESTS = 2.0  # Ajusta este valor si tienes límites de rate
```

Puedes aumentar este valor en `script.py` si sigues teniendo problemas de rate limit después de agregar créditos.

### 3. Reintentos con Espera Exponencial

Para rate limits reales (no por créditos), el código ahora espera:
- 1er intento: 5 segundos
- 2do intento: 10 segundos  
- 3er intento: 20 segundos

## Costos Aproximados

Para tu caso de uso (72 prompts × 3 modelos = 216 peticiones):

| Modelo | Costo Aprox. por 1000 tokens | Costo Total Estimado |
|--------|------------------------------|----------------------|
| gpt-4o-mini | $0.15 input / $0.60 output | $1-2 USD |
| gpt-4o | $5 input / $15 output | $10-20 USD |
| gpt-3.5-turbo | $0.50 input / $1.50 output | $0.50-1 USD |

*Estimaciones basadas en ~50 tokens de input y ~200 tokens de output por petición*

## Verificar el Estado de tu Cuenta

Después de agregar créditos, verifica:

1. **Balance disponible**: https://www.helicone.ai/billing
2. **Límites de rate**: Verifica si tienes límites por minuto/hora
3. **Historial de uso**: Revisa cuántos créditos has usado

## Ejecutar el Script Después de Agregar Créditos

Una vez que hayas agregado créditos:

```cmd
cd C:\Users\manel\PyCharmMiscProject\Helicone
python script.py
```

El script ahora:
- ✓ Detectará si tienes créditos insuficientes y te avisará claramente
- ✓ Esperará 2 segundos entre cada petición para evitar rate limits
- ✓ Si hay un rate limit real, esperará más tiempo antes de reintentar
- ✓ Guardará cada respuesta inmediatamente (no perderás progreso)

## Alternativa: Probar con Ollama

Si prefieres no agregar créditos, Ollama ya está configurado en tu proyecto:

```cmd
cd C:\Users\manel\PyCharmMiscProject\Ollama
python script.py
```

Ollama usa diferentes modelos y pricing. Revisa `Ollama/README.md` para más detalles.

## Ajustar el Delay Entre Peticiones

Si después de agregar créditos sigues teniendo problemas de rate limit, aumenta el delay:

En `script.py`, línea 23:
```python
DELAY_BETWEEN_REQUESTS = 5.0  # Cambia de 2.0 a 5.0 segundos
```

## Contacto de Soporte

Si el problema persiste después de agregar créditos:
- Soporte de Helicone: https://docs.helicone.ai/
- Discord: Busca el servidor de Helicone
- Email: Revisa en su página de contacto

