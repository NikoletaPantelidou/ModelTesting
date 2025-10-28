# 🧠 Multi-LLM Linguistic Test Framework

This project provides a **testing pipeline** for collecting responses from multiple large language models (LLMs) using a set of linguistic prompts.  
This project provides automated testing to measure each model’s linguistic reasoning capabilities.

---

## 📚 Table of Contents
- [About](#about)
- [Features](#features)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Setup & Installation](#setup--installation)
- [License](#license)

---

## 💡 About

This repository contains scripts and utilities for **benchmarking multiple LLMs** on a controlled **linguistic evaluation task**.  
It systematically measures accuracy, consistency, and reasoning quality across different model families (OpenAI, Anthropic, Mistral, etc.).

The main objectives are to:
- Evaluate LLMs’ performance on fine-grained linguistic phenomena  
- Compare how different kinds of models reason and comprehend language in more complex prompts

---

## ✨ Features
- 🧩 Tests **25 Large Language Models (LLMs)**  
- 🔍 Benchmarks **linguistic comprehension and reasoning**   
- 🧠 Supports custom prompt templates and datasets  

---

## 📁 Project Structure

```bash
llm-linguistic-eval/
├── script.py            # Main Python script that runs 25 LLMs 
├── prompts.csv          # Input file containing the linguistic prompts 
├── answers.csv          # Output file with model-generated answers 
├── requirements.txt     # Dependencies for the project
└── README.md            # Project documentation
```

---

## 🛠️ Tech Stack

**Language:** Python 3.13.5  
**Environment:** PyCharm IDE  

**Core Libraries:**
- `pandas` — data manipulation and analysis  
- `torch` — model interaction and computation  
- `matplotlib` / `seaborn` — visualization  
- `requests` / `openai` / `transformers` — model APIs and interfaces  
- `numpy`, `tqdm`, `json`, `os`, etc. — utilities


  ## ⚙️ Setup & Installation

### Prerequisites
- Python **3.13.5**
- PyCharm (recommended)
- A valid Hugging Face token and acceptance of Gemma’s policy

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/llm-linguistic-eval.git
   cd llm-linguistic-eval


## 📄 License
This project’s source code is licensed under the **MIT License** — see the [`LICENSE`](LICENSE) file for details.

⚠️ **Note on model usage:**
This repository interfaces with third-party large language models (LLMs) such as Gemma and others via official APIs or Hugging Face endpoints.  
Each model is governed by its own terms of service and licensing agreements.  
Users must:
- Obtain their own API tokens (e.g., from Hugging Face or OpenAI)
- Accept and comply with each model’s usage policy (e.g., [Gemma Model Policy](https://ai.google.dev/gemma/terms))
- Avoid redistributing model weights or outputs in violation of those terms

The authors of this repository do **not** claim ownership of or rights to any external model used in testing.
