# Models configuration for OpenRouter free tests
# Follow the same simple structure as other scripts: a list of dicts with at least the "name" key

MODELS = [
    {
        # Example free models on OpenRouter; update as needed
        "name": "meta-llama/llama-3.3-70b-instruct:free",
        "use_qa_pipeline": False,
        "trust_remote_code": False
    },
    {
    "name": "openai/gpt-5",
    "use_qa_pipeline": False,
    "trust_remote_code": False,
    },

    {
    "name": "perplexity/sonar-reasoning-pro",
    "use_qa_pipeline": False,
    "trust_remote_code": False,
    },
    {
        "name": "meta-llama/llama-4-scout:free",
        "use_qa_pipeline": False,
        "trust_remote_code": False
    },
    {
        "name": "google/gemma-3-4b-it:free",
        "use_qa_pipeline": False,
        "trust_remote_code": False
    },
    {
        "name": "openai/chatgpt-4o-latest",
        "use_qa_pipeline": False,
        "trust_remote_code": False
    },

]

