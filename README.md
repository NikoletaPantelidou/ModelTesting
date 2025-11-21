# ModelTesting

ModelTesting is a Python-based framework for evaluating 25 Large Language Models (LLMs) across four languages — English, Spanish, Catalan, and Italian — with a focus on semantic–pragmatic behavior.
The goal is to analyze how meaning-related tasks are handled by models from different providers and architectures.

Models are tested through:

Hugging Face (locally downloaded models)

Ollama (API-based execution)

OpenRouter (API-based execution)

This repository provides a unified interface to run the same semantic–pragmatic prompts across all models and compare performance.

## Features

Multilingual evaluation (EN, ES, CA, IT)

25 LLMs tested from multiple sources and architectures

Semantic–pragmatic prompt suite targeting:

contextual interpretation

different types of conditionals

## Project Structure

Each folder in the project follows the **same internal logic**, regardless of the model provider (Hugging Face, Ollama, or OpenRouter):

- **script.py** — main code for running the model tests  
- **answers/** — contains the outputs of each model in a CSV file, automatically generated after execution
- **script_models_config/** — configuration files that define parameters for each tested model  
- **logs/** — contains the console log messages for each run  
- **execute.bat** — Windows shortcut for running `script.py`  
- **clean_cache.bat** — Windows shortcut for clearing the Hugging Face model cache (only relevant for HF-based tests)

## Requirements / Libraries

The project uses different libraries depending on the provider: **Ollama**, **Hugging Face**, or **OpenRouter**.  
Common imports used across all modules are listed once, followed by provider-specific requirements.

---

### **Common Libraries (used in all providers)**

```python
import os
import sys
import logging
import pandas as pd
from datetime import datetime
from script_models_config import MODELS
```

---

### **Ollama – Additional Requirements**

```python
import requests
import time
```

Used for sending API requests to the local Ollama server.

---

### **Hugging Face – Additional Requirements**

```python
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
```

Required for local model execution using Hugging Face Transformers and for authenticating with Hugging Face Hub.

---

### **OpenRouter – Additional Requirements**

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import time
import threading
```
**Important:**  
➡️ *Replace the API key placeholder with your own private key. Never share your key publicly.*

## Contributing

Contributions are welcome.  
Feel free to extend:

- Model coverage  
- Languages  
- Prompt categories  
- Evaluation metrics  
- Visualization tools  
