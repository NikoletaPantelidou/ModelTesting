"""
Configuration file for Ollama models.
See full list at: https://ollama.com/library
"""

MODELS = [
    # Modelos disponibles en Ollama Cloud
    # Usa los nombres exactos de https://ollama.com/library

    {"name": "gpt-oss:20b-cloud"},              # Llama 3.2 - Recomendado
    {"name": "mistral"},               # Mistral 7B - Rápido y eficiente
    {"name": "phi3"},                  # Phi-3 - Pequeño pero potente
    {"name": "gemma2"},                # Gemma 2 - Google
    {"name": "qwen2.5"},               # Qwen 2.5 - Alibaba

    # Más modelos disponibles (descomenta para usar):
    # {"name": "llama3.1"},            # Llama 3.1
    # {"name": "deepseek-r1"},         # DeepSeek R1 - Razonamiento
    # {"name": "codellama"},           # Code Llama - Para código
    # {"name": "mixtral"},             # Mixtral - MoE
    # {"name": "neural-chat"},         # Neural Chat
]

