# INICIO RÁPIDO - OpenRouter Free

## Pasos para empezar:

### 1. Obtener API Key (5 minutos)
   - Ve a: https://openrouter.ai/
   - Regístrate (es gratis)
   - Copia tu API key del dashboard

### 2. Configurar API Key
   
   **Opción A - Temporal (para esta sesión):**
   ```bash
   set OPENROUTER_API_KEY=tu_api_key_aqui
   ```

   **Opción B - Usar el script:**
   1. Edita `set_api_key.bat`
   2. Reemplaza `YOUR_API_KEY_HERE` con tu API key real
   3. Ejecuta `set_api_key.bat`

   **Opción C - Permanente:**
   1. Panel de Control > Sistema > Configuración avanzada
   2. Variables de entorno > Nueva
   3. Nombre: `OPENROUTER_API_KEY`
   4. Valor: tu API key

### 3. Probar la Conexión
   ```bash
   test_api.bat
   ```
   
   Si ves "[SUCCESS] API connection test passed!" todo está listo.

### 4. Preparar tus Prompts
   - Edita `prompts/prompts_all.csv`
   - Formato mínimo requerido:
     ```csv
     prompt;language
     "Context: The sky is blue. Question: What color is the sky?";english
     ```

### 5. Ejecutar el Script Principal
   ```bash
   execute.bat
   ```

### 6. Ver Resultados
   - Los resultados estarán en `answers/[modelo]_answers.csv`
   - Los logs en `logs/execution_[timestamp].log`

## Modelos Gratuitos Disponibles

Por defecto, se usan estos modelos gratuitos:
- `meta-llama/llama-3.3-8b-instruct:free`
- `google/gemini-2.0-flash-exp:free`
- `mistralai/mistral-7b-instruct:free`
- `nousresearch/hermes-3-llama-3.1-405b:free`
- `qwen/qwen-2.5-7b-instruct:free`

Puedes añadir más editando `script_models_config.py`.
Ver modelos disponibles: https://openrouter.ai/models?q=free

## Solución Rápida de Problemas

### "OPENROUTER_API_KEY not found"
→ No configuraste la API key. Ve al paso 2.

### "401 Unauthorized"
→ API key inválida. Verifica que copiaste bien la key.

### "429 Rate Limit"
→ Demasiadas peticiones. Espera un minuto o aumenta `REQUEST_DELAY` en script.py.

### Script muy lento
→ Ajusta `MAX_WORKERS` y `REQUEST_DELAY` en script.py para tu caso.

## Archivos Importantes

- `script.py` - Script principal
- `script_models_config.py` - Configuración de modelos
- `prompts/prompts_all.csv` - TUS prompts aquí
- `answers/` - Resultados aquí
- `logs/` - Logs de ejecución aquí

## Soporte

- Documentación completa: Ver `README.md`
- OpenRouter Docs: https://openrouter.ai/docs
- OpenRouter Discord: https://discord.gg/openrouter

## ¿Listo?

1. ✅ Configura tu API key
2. ✅ Ejecuta `test_api.bat`
3. ✅ Prepara tus prompts
4. ✅ Ejecuta `execute.bat`
5. ✅ ¡Disfruta! 🚀

