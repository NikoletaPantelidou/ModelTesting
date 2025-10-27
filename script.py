"""
This script processes a series of text prompts using various AI language models.
It loads different models from HuggingFace Hub, processes each prompt, and saves the results.
The script handles both question-answering and text generation tasks, with full logging and error handling.

Key features:
- Supports multiple AI models (Mistral, Phi, Falcon, etc.)
- Processes prompts from CSV files
- Saves results incrementally
- Includes detailed logging
- Handles errors gracefully
"""

# ============================================================================
# IMPORTS - Required Python packages and modules
# ============================================================================
import os              # Operating system interface - for file and path operations
import sys            # System-specific parameters and functions
import logging        # Logging facility for tracking script execution
from datetime import datetime  # Date and time utilities for log naming
import pandas as pd   # Data manipulation and CSV file handling
import torch         # PyTorch deep learning framework for model operations
from transformers import (
    pipeline,         # High-level interface for model tasks
    AutoTokenizer,    # Automatic tokenizer loading
    AutoModelForCausalLM  # Automatic model loading for text generation
)

# Disable HuggingFace Hub warnings
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = '1'

# HuggingFace authentication token (for gated models)
HF_TOKEN = "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"

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
os.makedirs('logs', exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f'logs/execution_{timestamp}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION - Main settings and model configurations
# ============================================================================
# List of AI models to process, each with specific settings:
#   - name: HuggingFace model identifier
#   - use_qa_pipeline: True for question-answering, False for text generation
#   - trust_remote_code: Whether to allow model-specific code from HuggingFace

#All models are added to the MODELS list so only one run is needed.
MODELS = [
    # {"name": "mistralai/Mistral-7B-Instruct-v0.2", "use_qa_pipeline": False, "trust_remote_code": False},
    {"name": "microsoft/phi-4", "use_qa_pipeline": False, "trust_remote_code": False},
    {"name": "tiiuae/falcon-7b-instruct", "use_qa_pipeline": False, "trust_remote_code": False},
    {"name": "arcee-ai/Arcee-Blitz", "use_qa_pipeline": False, "trust_remote_code": False},
    {"name": "arcee-ai/Virtuoso-Lite", "use_qa_pipeline": False, "trust_remote_code": False},
    {"name": "inclusionAI/Ling-1T-FP8", "use_qa_pipeline": False, "trust_remote_code": True},
    {"name": "moonshotai/Kimi-K2-Instruct-0905", "use_qa_pipeline": False, "trust_remote_code": True},
    {"name": "distilbert-base-cased-distilled-squad", "use_qa_pipeline": True, "trust_remote_code": False},
    # Models that require authentication:
    {"name": "google/gemma-3-1b-it", "use_qa_pipeline": False, "trust_remote_code": False},  # Requires gated access
]

INPUT_FILE = "prompts/example.csv"  # Input CSV file path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEMPERATURE = 0.0  # Temperature for generation (0.0 = deterministic)
OUTPUT_DIR = "answers"  # Directory for output files
CSV_SEPARATOR = ";"



# ============================================================================
# FUNCTIONS
# ============================================================================

def load_model(model_name, use_qa_pipeline=False, trust_remote_code=False):
    """
    Load an AI model from HuggingFace Hub with appropriate configuration.

    Args:
        model_name (str): The name/path of the model on HuggingFace Hub
        use_qa_pipeline (bool): If True, loads as Q&A pipeline; if False, as text generation
        trust_remote_code (bool): Whether to allow custom code from model repository

    Returns:
        dict: Contains model type ('qa' or 'generative') and loaded model components

    This function handles two types of models:
    1. Question-Answering (QA) models - Uses HuggingFace pipeline
    2. Text Generation models - Loads tokenizer and model separately
    """
    logger.info(f"[INFO] Loading model: {model_name} (device: {DEVICE})")

    try:
        if use_qa_pipeline:
            qa_model = pipeline(
                "question-answering",
                model=model_name,
                tokenizer=model_name,
                device=0 if DEVICE == "cuda" else -1,
                trust_remote_code=trust_remote_code,
                token=HF_TOKEN
            )
            logger.info(f"[OK] QA pipeline loaded successfully")
            return {'type': 'qa', 'model': qa_model}
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                token=HF_TOKEN
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                token=HF_TOKEN
            ).to(DEVICE)
            model.config.pad_token_id = tokenizer.pad_token_id
            logger.info(f"[OK] Model and tokenizer loaded successfully")
            return {'type': 'generative', 'model': model, 'tokenizer': tokenizer}
    except Exception as e:
        logger.error(f"[ERROR] Error loading model: {str(e)}")
        raise


def load_input_file():
    """
    Load and validate the input CSV file containing prompts.

    The function performs several checks:
    1. Verifies file exists at INPUT_FILE path
    2. Loads CSV with specified separator
    3. Cleans column names (strips whitespace and BOM)
    4. Validates presence of required 'prompt' column

    Returns:
        pandas.DataFrame: Loaded and validated dataframe with prompts

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If required 'prompt' column is missing
    """
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Could not find {INPUT_FILE}")

    try:
        df = pd.read_csv(INPUT_FILE, sep=CSV_SEPARATOR)
        df.columns = df.columns.str.strip().str.replace('\ufeff', '')

        logger.info(f"[OK] File loaded: {len(df)} rows, columns: {df.columns.tolist()}")

        if "prompt" not in df.columns:
            raise ValueError("CSV must contain a 'prompt' column")

        return df
    except Exception as e:
        logger.error(f"[ERROR] Error reading CSV: {str(e)}")
        raise


def parse_prompt(raw_prompt):
    """
    Parse a raw text prompt to separate it into context and question parts.

    The function handles two formats:
    1. Text with comma: Everything before last comma is context, after is question
    2. Text without comma: Last sentence is question, rest is context

    Args:
        raw_prompt (str): The complete text prompt to parse

    Returns:
        tuple: (context, question) where question always ends with '?'
    """
    sentences = raw_prompt.split(".")
    joined_text = ".".join(sentences)
    last_comma_idx = joined_text.rfind(",")

    if last_comma_idx != -1:
        question = joined_text[last_comma_idx + 1:].strip()
        question = question[0].upper() + question[1:] if question else question
        context = joined_text[:last_comma_idx].strip()
    else:
        context = ".".join(sentences[:-1]).strip() + "."
        question = sentences[-1].strip()

    if not question.endswith("?"):
        question += "?"

    return context, question


def generate_answer(context, question, model_dict):
    """
    Generate an answer using either QA pipeline or text generation model.

    This function handles two types of answer generation:
    1. Question-Answering (QA) pipeline: Directly extracts answer from context
    2. Text Generation: Formats prompt with context and question, generates response

    Args:
        context (str): The background information or context
        question (str): The specific question to answer
        model_dict (dict): Dictionary containing model type and components

    Returns:
        str: The generated answer, with any prompt text removed for generation models
    """
    if model_dict['type'] == 'qa':
        result = model_dict['model'](question=question, context=context)
        return result[0]["answer"] if isinstance(result, list) else result["answer"]
    else:
        full_prompt = f"Context: {context}\nQuestion: {question}\n\nAnswer:"
        tokenizer = model_dict['tokenizer']
        inputs = tokenizer(full_prompt, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = model_dict['model'].generate(
                **inputs,
                max_new_tokens=50,
                temperature=TEMPERATURE if TEMPERATURE > 0 else None,
                do_sample=TEMPERATURE > 0,
                pad_token_id=tokenizer.pad_token_id
            )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.replace(full_prompt, "").strip()


def process_row(idx, row, total_rows, model_dict):
    """Process a single row and return the answer."""
    try:
        logger.info(f"[INFO] Processing row {idx + 1}/{total_rows}")

        raw_prompt = str(row.prompt)
        context, question = parse_prompt(raw_prompt)
        answer = generate_answer(context, question, model_dict)

        logger.info(f"[OK] Row {idx + 1} completed")
        return idx, answer

    except Exception as e:
        logger.error(f"[ERROR] Error processing row {idx + 1}: {str(e)}")
        return idx, f"ERROR: {str(e)}"


def process_prompts_sequential(df, model_dict, model_name):
    """Process all prompts."""
    total_rows = len(df)

    logger.info(f"[INFO] Starting sequential processing for {model_name}")

    # Load existing answers if they exist
    answers, completed_count = load_existing_answers(df, model_name)

    if completed_count > 0:
        logger.info(f"[INFO] Resuming from row {completed_count + 1}")

    for idx, row in enumerate(df.itertuples(index=False)):
        # Skip if a valid answer already exists (not empty and not an error)
        if pd.notna(answers[idx]) and not str(answers[idx]).startswith("ERROR:"):
            logger.info(f"[SKIP] Row {idx + 1}/{total_rows} already completed, skipping")
            continue

        idx_result, answer = process_row(idx, row, total_rows, model_dict)
        answers[idx] = answer

        # Save immediately after generating each answer
        save_single_answer(df, answers, model_name, idx)

        logger.info(f"[PROGRESS] {model_name} - {idx + 1}/{total_rows} rows completed")

    logger.info(f"[OK] All rows processed for {model_name}")

def get_output_file_path(model_name):
    """Get the output file path for a model."""
    safe_model_name = model_name.replace("/", "-")
    return os.path.join(OUTPUT_DIR, f"{safe_model_name}_answers.csv")


def load_existing_answers(df, model_name):
    """Load existing answers from a previous execution if available."""
    output_file = get_output_file_path(model_name)

    if os.path.exists(output_file):
        try:
            existing_df = pd.read_csv(output_file)
            if 'answer' in existing_df.columns and len(existing_df) == len(df):
                logger.info(f"[INFO] Found existing answers file: {output_file}")
                # Count completed answers (not empty and not errors)
                completed = existing_df['answer'].notna().sum()
                logger.info(f"[INFO] Existing progress: {completed}/{len(df)} answers completed")
                return existing_df['answer'].tolist(), completed
            else:
                logger.info(f"[INFO] Existing file found but incomplete or size mismatch, starting fresh")
        except Exception as e:
            logger.warning(f"[WARNING] Could not load existing answers: {str(e)}")

    # Return list of None if no previous answers
    return [None] * len(df), 0


def save_single_answer(df, answers, model_name, current_idx):
    """Save answers incrementally to CSV file."""
    try:
        output_file = get_output_file_path(model_name)
        df_temp = df.assign(answer=answers)
        df_temp.to_csv(output_file, index=False)
        logger.info(f"[OK] Answer {current_idx + 1} saved to {output_file}")
    except Exception as e:
        logger.error(f"[ERROR] Error saving answer {current_idx + 1}: {str(e)}")
        raise


def process_single_model(model_config, df):
    """Process prompts with a single model."""
    model_name = model_config["name"]
    use_qa_pipeline = model_config.get("use_qa_pipeline", False)
    trust_remote_code = model_config.get("trust_remote_code", False)

    try:
        logger.info(f"[INFO] ===== Starting processing for model: {model_name} =====")

        model_dict = load_model(model_name, use_qa_pipeline, trust_remote_code)
        process_prompts_sequential(df, model_dict, model_name)


        logger.info(f"[OK] ===== Completed processing for model: {model_name} =====\n")

    except Exception as e:
        logger.error(f"[ERROR] Error processing model {model_name}: {str(e)}")
        raise


def process_all_models(df):
    """Process all models sequentially."""
    total_models = len(MODELS)
    successful_models = []
    failed_models = []

    for idx, model_config in enumerate(MODELS, 1):
        model_name = model_config["name"]
        try:
            logger.info(f"[INFO] Processing model {idx}/{total_models}: {model_name}")

            process_single_model(model_config, df)
            
            successful_models.append(model_name)
            logger.info(f"[OK] Model {model_name} completed successfully ({len(successful_models)}/{total_models})")

        except Exception as e:
            failed_models.append((model_name, str(e)))
            logger.error(f"[ERROR] Model {model_name} failed: {str(e)}")
            logger.info(f"[INFO] Continuing with next model...")

    # Final summary
    if successful_models:
        logger.info(f"[OK] Successfully processed {len(successful_models)} models:")
        for model in successful_models:
            logger.info(f"  ✓ {model}")

    if failed_models:
        logger.warning(f"[WARNING] {len(failed_models)} models failed:")
        for model, error in failed_models:
            logger.warning(f"  ✗ {model}")
            error_summary = error.split('\n')[0][:100]
            logger.warning(f"    Error: {error_summary}...")

    return successful_models, failed_models


def main():
    """
    Main execution function that orchestrates the entire process.

    This function coordinates the following steps:
    1. Loads input prompts from CSV file
    2. Creates output directory if needed
    3. Iterates through configured models:
        - Loads each model
        - Processes all prompts
        - Saves answers to model-specific CSV file
    4. Handles errors and logs progress throughout

    The function uses incremental saving to preserve progress
    in case of interruption or errors with specific models.
    """
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df = load_input_file()
        successful_models, failed_models = process_all_models(df)

        logger.info(f"[OK] Processing Summary:")
        logger.info(f"  Total models: {len(MODELS)}")
        logger.info(f"  Successful: {len(successful_models)}")
        logger.info(f"  Failed: {len(failed_models)}")

    except Exception as e:
        logger.error(f"[FATAL] Fatal error in main execution: {str(e)}")
        raise

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    main()
