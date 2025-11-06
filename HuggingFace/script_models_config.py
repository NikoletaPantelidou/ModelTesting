"""
Configuration file for models used in prompt processing.
Each model has its own configuration including whether to use QA pipeline
and whether to trust remote code.
"""

MODELS = [
  #  {"name": "distilbert-base-cased-distilled-squad", "use_qa_pipeline": True, "trust_remote_code": False}, # DONE
    {"name": "mistralai/Mistral-7B-Instruct-v0.2", "use_qa_pipeline": False, "trust_remote_code": False},
    {"name": "tiiuae/falcon-7b-instruct", "use_qa_pipeline": False, "trust_remote_code": False},
    {"name": "microsoft/phi-4", "use_qa_pipeline": False, "trust_remote_code": False},
    {"name": "arcee-ai/Virtuoso-Lite", "use_qa_pipeline": False, "trust_remote_code": False},
    {"name": "mlfoundations-dev/oh-dcft-v3.1-gpt-4o-mini", "use_qa_pipeline": False, "trust_remote_code": False},
    {"name": "allenai/OLMo-7B", "use_qa_pipeline": False, "trust_remote_code": True}, #ALL languages
    {"name": "google/gemma-3-4b-it", "use_qa_pipeline": False, "trust_remote_code": False}, #ALL languages
    {"name": "arcee-ai/Arcee-Blitz", "use_qa_pipeline": False, "trust_remote_code": False}, #ALL languages

]

