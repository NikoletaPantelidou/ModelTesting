"""
Script for sequential processing of prompts using transformer models.
Includes complete logging and error handling.
"""

# ============================================================================
# IMPORTS
# ============================================================================
import os
import sys
import logging
import pandas as pd
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from script_models_config import MODELS
from datetime import datetime
from huggingface_hub import login

login(token="YOUR_TOKEN")


# HuggingFace authentication token (for gated models)
HF_TOKEN = None  # Set to None to skip authentication
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

# CONFIGURATION
# ============================================================================
# List of models to process (each model has its own configuration)

INPUT_FILE = "prompts/prompts_all.csv"  # Input CSV file path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEMPERATURE = 0.0  # Temperature for generation (0.0 = deterministic)
OUTPUT_DIR = "answers"  # Directory for output files
CSV_SEPARATOR = ";"
MAX_WORKERS = 4  # Number of parallel workers for processing prompts

# ============================================================================
# FUNCTIONS
# ============================================================================

def load_model(model_name, use_qa_pipeline=False, trust_remote_code=False):
    """Load the AI model (QA pipeline or generative model)."""
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
                token=HF_TOKEN,
            )
            # Configure padding for decoder-only models
            tokenizer.padding_side = 'left'
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load model with GPU support
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                token=HF_TOKEN,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                device_map="auto" if DEVICE == "cuda" else None,
            )

            # Only manually move to device if device_map is not used
            if DEVICE != "cuda":
                model = model.to(DEVICE)

            model.config.pad_token_id = tokenizer.pad_token_id
            logger.info(f"[OK] Model and tokenizer loaded successfully on {DEVICE}")
            return {'type': 'generative', 'model': model, 'tokenizer': tokenizer}
    except Exception as e:
        logger.error(f"[ERROR] Error loading model: {str(e)}")
        raise

def unload_model(model_dict, model_name):
    """Unload model from memory and free up resources."""
    logger.info(f"[INFO] Unloading model: {model_name}")

    try:
        if model_dict['type'] == 'generative':
            # Move model to CPU before deleting (helps with CUDA memory)
            if hasattr(model_dict['model'], 'to'):
                model_dict['model'].to('cpu')
            del model_dict['model']
            del model_dict['tokenizer']
        else:
            del model_dict['model']

        # Clear CUDA cache if using GPU
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        logger.info(f"[OK] Model {model_name} unloaded successfully")
    except Exception as e:
        logger.warning(f"[WARNING] Error unloading model: {str(e)}")

def load_input_file():
    """Load and validate the input CSV file."""
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
    """Parse the raw prompt to extract context and question."""
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
    """Generate an answer using the loaded model."""
    if model_dict['type'] == 'qa':
        result = model_dict['model'](question=question, context=context)
        return result[0]["answer"] if isinstance(result, list) else result["answer"]
    else:
        full_prompt = f"Context: {context}\nQuestion: {question}\n\nAnswer:"
        tokenizer = model_dict['tokenizer']
        inputs = tokenizer(full_prompt, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            # Preparar parámetros de generación
            gen_kwargs = {
                "max_new_tokens": 300,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id
            }

            # Solo añadir temperature y do_sample si temperature > 0
            if TEMPERATURE > 0:
                gen_kwargs["temperature"] = TEMPERATURE
                gen_kwargs["do_sample"] = True

            outputs = model_dict['model'].generate(**inputs, **gen_kwargs)

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.replace(full_prompt, "").strip()

def process_row(idx, row, total_rows, model_dict):
    """Process a single row and return the answer."""
    try:
        raw_prompt = str(row.prompt)

        # Log which prompt is being processed
        test_item = getattr(row, 'test_item', None)
        if test_item:
            logger.info(f"[INFO] Processing row {idx + 1}/{total_rows} - test_item: {test_item}")
        else:
            logger.info(f"[INFO] Processing row {idx + 1}/{total_rows}")

        logger.info(f"[PROMPT] {raw_prompt[:100]}..." if len(raw_prompt) > 100 else f"[PROMPT] {raw_prompt}")

        context, question = parse_prompt(raw_prompt)
        answer = generate_answer(context, question, model_dict)

        # Log the model's response
        logger.info(f"[RESPONSE] {answer[:200]}..." if len(answer) > 200 else f"[RESPONSE] {answer}")

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

def process_prompts_parallel(df, model_dict, model_name):
    """Process all prompts in parallel using ThreadPoolExecutor."""
    total_rows = len(df)

    logger.info(f"[INFO] Starting parallel processing for {model_name} with {MAX_WORKERS} workers")

    # Load existing answers if they exist
    answers, completed_count = load_existing_answers(df, model_name)

    if completed_count > 0:
        logger.info(f"[INFO] Resuming from row {completed_count + 1}")

    # Create a list of pending tasks (indices that need processing)
    pending_tasks = []
    for idx, row in enumerate(df.itertuples(index=False)):
        # Skip if a valid answer already exists (not empty and not an error)
        if pd.notna(answers[idx]) and not str(answers[idx]).startswith("ERROR:"):
            logger.info(f"[SKIP] Row {idx + 1}/{total_rows} already completed, skipping")
            continue
        pending_tasks.append((idx, row))

    if not pending_tasks:
        logger.info(f"[OK] All rows already processed for {model_name}")
        return

    logger.info(f"[INFO] Processing {len(pending_tasks)} pending prompts...")

    # Process tasks in parallel
    completed_tasks = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all pending tasks
        future_to_idx = {
            executor.submit(process_row, idx, row, total_rows, model_dict): idx
            for idx, row in pending_tasks
        }

        # Process completed tasks as they finish
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                idx_result, answer = future.result()
                answers[idx_result] = answer

                # Save immediately after generating each answer
                save_single_answer(df, answers, model_name, idx_result)

                completed_tasks += 1
                logger.info(f"[PROGRESS] {model_name} - {completed_tasks}/{len(pending_tasks)} pending prompts completed ({completed_tasks + completed_count}/{total_rows} total)")

            except Exception as e:
                logger.error(f"[ERROR] Task for row {idx + 1} failed: {str(e)}")
                answers[idx] = f"ERROR: {str(e)}"

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

def check_pending_answers(df, model_name):
    """Check if there are pending answers to complete for a model."""
    answers, completed_count = load_existing_answers(df, model_name)
    total_rows = len(df)

    # Check how many valid answers exist (not None, not empty, not errors)
    pending_count = 0
    for answer in answers:
        if pd.isna(answer) or str(answer).startswith("ERROR:"):
            pending_count += 1

    return pending_count, answers

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

    logger.info(f"[INFO] ===== Starting processing for model: {model_name} =====")

    # Check if there are pending answers before loading the model
    pending_count, _ = check_pending_answers(df, model_name)

    if pending_count == 0:
        logger.info(f"[SKIP] Model {model_name} has all answers completed, skipping model load")
        return

    logger.info(f"[INFO] Model {model_name} has {pending_count} pending answers, loading model...")

    model_dict = load_model(model_name, use_qa_pipeline, trust_remote_code)

    try:
        # Use parallel processing for better performance
        process_prompts_parallel(df, model_dict, model_name)
    finally:
        # Always unload the model, even if processing fails
        unload_model(model_dict, model_name)

    logger.info(f"[OK] ===== Completed processing for model: {model_name} =====\n")

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
    """Main execution function."""
    try:
        # Log GPU information
        logger.info(f"[INFO] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"[INFO] CUDA device count: {torch.cuda.device_count()}")
            logger.info(f"[INFO] Current CUDA device: {torch.cuda.current_device()}")
            logger.info(f"[INFO] CUDA device name: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning(f"[WARNING] No CUDA device detected. Running on CPU.")

        logger.info(f"[INFO] Selected device: {DEVICE}")
        logger.info(f"[INFO] Max workers for parallel processing: {MAX_WORKERS}")

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
