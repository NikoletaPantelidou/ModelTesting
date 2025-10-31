"""
Configuration file for Ollama Cloud models.
Each model configuration includes the model name.

Available models on Ollama Cloud - Comprehensive List
See full list at: https://ollama.com/library
Last updated: October 2025
"""

# ============================================================================
# MODELS CONFIGURATION
# ============================================================================
# Uncomment the models you want to use
# Format: {"name": "model-name:version"}

MODELS = [
    # ========== POPULAR/RECOMMENDED MODELS ==========
    {"name": "llama3.2:latest"},           # Meta's Llama 3.2 - Latest (3B/1B)
    {"name": "llama3.1:latest"},           # Meta's Llama 3.1 (8B/70B/405B)
    {"name": "mistral:latest"},            # Mistral 7B - Fast and efficient
    {"name": "phi3:latest"},               # Microsoft Phi-3 - Small but powerful
    {"name": "gemma2:latest"},             # Google Gemma 2 (2B/9B/27B)

    # ========== META LLAMA FAMILY ==========
    # {"name": "llama3.2:1b"},             # Llama 3.2 1B - Smallest, fastest
    # {"name": "llama3.2:3b"},             # Llama 3.2 3B - Balanced
    # {"name": "llama3.1:8b"},             # Llama 3.1 8B
    # {"name": "llama3.1:70b"},            # Llama 3.1 70B - High quality
    # {"name": "llama3.1:405b"},           # Llama 3.1 405B - Largest
    # {"name": "llama3:latest"},           # Llama 3 (previous version)
    # {"name": "llama2:latest"},           # Llama 2 (older)
    # {"name": "llama2:13b"},              # Llama 2 13B
    # {"name": "llama2:70b"},              # Llama 2 70B
    # {"name": "codellama:latest"},        # Code Llama - Specialized for coding
    # {"name": "codellama:7b"},            # Code Llama 7B
    # {"name": "codellama:13b"},           # Code Llama 13B
    # {"name": "codellama:34b"},           # Code Llama 34B

    # ========== MISTRAL FAMILY ==========
    # {"name": "mistral:7b"},              # Mistral 7B base
    # {"name": "mistral-nemo:latest"},     # Mistral Nemo 12B
    # {"name": "mistral-large:latest"},    # Mistral Large - Most capable
    # {"name": "mixtral:latest"},          # Mixtral 8x7B - MoE model
    # {"name": "mixtral:8x7b"},            # Mixtral 8x7B explicit
    # {"name": "mixtral:8x22b"},           # Mixtral 8x22B - Larger MoE

    # ========== GOOGLE GEMMA FAMILY ==========
    # {"name": "gemma:latest"},            # Gemma (previous version)
    # {"name": "gemma:2b"},                # Gemma 2B
    # {"name": "gemma:7b"},                # Gemma 7B
    # {"name": "gemma2:2b"},               # Gemma 2 2B - Latest version
    # {"name": "gemma2:9b"},               # Gemma 2 9B
    # {"name": "gemma2:27b"},              # Gemma 2 27B - High quality

    # ========== MICROSOFT PHI FAMILY ==========
    # {"name": "phi3:mini"},               # Phi-3 Mini 3.8B
    # {"name": "phi3:medium"},             # Phi-3 Medium 14B
    # {"name": "phi3.5:latest"},           # Phi-3.5 - Latest version

    # ========== QWEN FAMILY (Alibaba) ==========
    # {"name": "qwen2.5:latest"},          # Qwen 2.5 Latest
    # {"name": "qwen2.5:0.5b"},            # Qwen 2.5 0.5B - Tiny
    # {"name": "qwen2.5:1.5b"},            # Qwen 2.5 1.5B
    # {"name": "qwen2.5:3b"},              # Qwen 2.5 3B
    # {"name": "qwen2.5:7b"},              # Qwen 2.5 7B
    # {"name": "qwen2.5:14b"},             # Qwen 2.5 14B
    # {"name": "qwen2.5:32b"},             # Qwen 2.5 32B
    # {"name": "qwen2.5:72b"},             # Qwen 2.5 72B - High quality
    # {"name": "qwen2:latest"},            # Qwen 2 (previous)
    # {"name": "qwen:latest"},             # Qwen (original)
    # {"name": "qwen2.5-coder:latest"},    # Qwen 2.5 Coder - For coding

    # ========== DEEPSEEK FAMILY ==========
    # {"name": "deepseek-r1:latest"},      # DeepSeek-R1 - Reasoning model
    # {"name": "deepseek-r1:1.5b"},        # DeepSeek-R1 1.5B
    # {"name": "deepseek-r1:7b"},          # DeepSeek-R1 7B
    # {"name": "deepseek-r1:8b"},          # DeepSeek-R1 8B
    # {"name": "deepseek-r1:14b"},         # DeepSeek-R1 14B
    # {"name": "deepseek-r1:32b"},         # DeepSeek-R1 32B
    # {"name": "deepseek-r1:70b"},         # DeepSeek-R1 70B
    # {"name": "deepseek-r1:671b"},        # DeepSeek-R1 671B - Massive
    # {"name": "deepseek-coder:latest"},   # DeepSeek Coder
    # {"name": "deepseek-coder:6.7b"},     # DeepSeek Coder 6.7B
    # {"name": "deepseek-coder:33b"},      # DeepSeek Coder 33B

    # ========== COMMAND R (Cohere) ==========
    # {"name": "command-r:latest"},        # Command R 35B
    # {"name": "command-r:35b"},           # Command R 35B explicit
    # {"name": "command-r-plus:latest"},   # Command R+ 104B - More capable

    # ========== YI FAMILY (01.AI) ==========
    # {"name": "yi:latest"},               # Yi Latest
    # {"name": "yi:6b"},                   # Yi 6B
    # {"name": "yi:9b"},                   # Yi 9B
    # {"name": "yi:34b"},                  # Yi 34B

    # ========== SPECIALIZED MODELS ==========
    # {"name": "vicuna:latest"},           # Vicuna - Chat optimized
    # {"name": "orca-mini:latest"},        # Orca Mini - Efficient
    # {"name": "neural-chat:latest"},      # Intel Neural Chat
    # {"name": "starling-lm:latest"},      # Starling-LM - RLHF trained
    # {"name": "openchat:latest"},         # OpenChat
    # {"name": "nous-hermes2:latest"},     # Nous Hermes 2
    # {"name": "dolphin-mixtral:latest"},  # Dolphin Mixtral - Uncensored
    # {"name": "solar:latest"},            # Solar 10.7B
    # {"name": "zephyr:latest"},           # Zephyr - Helpful assistant

    # ========== CODING SPECIALIZED ==========
    # {"name": "codegemma:latest"},        # CodeGemma - Google's code model
    # {"name": "starcoder2:latest"},       # StarCoder2 - Code generation
    # {"name": "codeqwen:latest"},         # CodeQwen - Qwen for coding
    # {"name": "deepseek-coder-v2:latest"}, # DeepSeek Coder V2

    # ========== MULTILINGUAL ==========
    # {"name": "aya:latest"},              # Aya - 101 languages
    # {"name": "aya:8b"},                  # Aya 8B
    # {"name": "aya:35b"},                 # Aya 35B

    # ========== VISION MODELS (Multimodal) ==========
    # {"name": "llava:latest"},            # LLaVA - Vision + Language
    # {"name": "llava:7b"},                # LLaVA 7B
    # {"name": "llava:13b"},               # LLaVA 13B
    # {"name": "llava:34b"},               # LLaVA 34B
    # {"name": "bakllava:latest"},         # BakLLaVA - Vision model

    # ========== MATH/REASONING ==========
    # {"name": "wizardmath:latest"},       # WizardMath - Math specialist
    # {"name": "wizardcoder:latest"},      # WizardCoder - Coding specialist
    # {"name": "mathstral:latest"},        # Mathstral - Mistral math

    # ========== SMALLER/EFFICIENT MODELS ==========
    # {"name": "tinyllama:latest"},        # TinyLlama 1.1B - Very fast
    # {"name": "stablelm2:latest"},        # StableLM 2
    # {"name": "orca2:latest"},            # Orca 2 - Microsoft
    # {"name": "falcon:latest"},           # Falcon - TII
    # {"name": "falcon:7b"},               # Falcon 7B
    # {"name": "falcon:40b"},              # Falcon 40B

    # ========== EXPERIMENTAL/RESEARCH ==========
    # {"name": "granite-code:latest"},     # IBM Granite Code
    # {"name": "samantha-mistral:latest"}, # Samantha - Companion AI
    # {"name": "wizardlm2:latest"},        # WizardLM 2
    # {"name": "openhermes:latest"},       # OpenHermes
    # {"name": "xwinlm:latest"},           # Xwin-LM
]

# ============================================================================
# SIZE RECOMMENDATIONS
# ============================================================================
# Small & Fast (< 5B parameters):
#   - llama3.2:1b, llama3.2:3b
#   - qwen2.5:0.5b, qwen2.5:1.5b, qwen2.5:3b
#   - gemma2:2b
#   - tinyllama:latest
#   - phi3:mini
#
# Balanced (5-15B parameters):
#   - mistral:7b, mistral-nemo:12b
#   - llama3.1:8b
#   - gemma2:9b
#   - qwen2.5:7b, qwen2.5:14b
#   - phi3:medium
#
# High Quality (15-70B parameters):
#   - llama3.1:70b
#   - qwen2.5:32b, qwen2.5:72b
#   - gemma2:27b
#   - mixtral:8x22b
#   - command-r-plus:latest
#
# Specialized:
#   - Coding: codellama, deepseek-coder, qwen2.5-coder, starcoder2
#   - Math: wizardmath, mathstral
#   - Multilingual: aya
#   - Vision: llava, bakllava
#   - Reasoning: deepseek-r1
#
# ============================================================================
# NOTES
# ============================================================================
# - Larger models = better quality but slower and more expensive
# - :latest tag gets the most recent version
# - Specific versions (e.g., :7b, :70b) give you control over size
# - Test with small models first, then scale up if needed
# - For cloud usage, consider cost per token for larger models
# ============================================================================

