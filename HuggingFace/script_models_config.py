"""
Configuration file for models used in prompt processing.
Each model has its own configuration including whether to use QA pipeline
and whether to trust remote code.
"""

MODELS = [
    {"name": "distilbert-base-cased-distilled-squad", "use_qa_pipeline": True, "trust_remote_code": False},
    {"name": "mistralai/Mistral-7B-Instruct-v0.2", "use_qa_pipeline": False, "trust_remote_code": False},
    {"name": "tiiuae/falcon-7b-instruct", "use_qa_pipeline": False, "trust_remote_code": False},  # Falcon is already integrated in transformers
    {"name": "microsoft/phi-4", "use_qa_pipeline": False, "trust_remote_code": False}, 
    {"name": "arcee-ai/Virtuoso-Lite", "use_qa_pipeline": False, "trust_remote_code": False},
   #{"name": "cerebras/GLM-4.6-REAP-218B-A32B-FP8", "use_qa_pipeline": False, "trust_remote_code": False},
   #{"name": "inclusionAI/Ling-1T-FP8", "use_qa_pipeline": False, "trust_remote_code": True},
   #{"name": "arcee-ai/Arcee-Blitz", "use_qa_pipeline": False, "trust_remote_code": False},
   #{"name": "moonshotai/Kimi-K2-Instruct-0905", "use_qa_pipeline": False, "trust_remote_code": True},
    {"name": "deepseek-ai/DeepSeek-V3.2-Exp", "use_qa_pipeline": False, "trust_remote_code": False},
    {"name": "google/gemma-3-4b-it", "use_qa_pipeline": False, "trust_remote_code": False},  
    {"name": "mlfoundations-dev/oh-dcft-v3.1-gpt-4o-mini", "use_qa_pipeline": False, "trust_remote_code": False},  
]

