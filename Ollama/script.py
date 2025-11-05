"""
Script for sequential processing of prompts using Ollama Cloud API.
"""

import os
import sys
import logging
import pandas as pd
import requests
import time
from script_models_config import MODELS
from datetime import datetime

# CONFIGURATION
OLLAMA_API_BASE = "https://ollama.com/api"
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "9fe5315cf7ce4326bd5249373221cfc7.STrBgk70DS0dH888JwANFA3K")
INPUT_FILE = "prompts/prompts_all.csv"
OUTPUT_DIR = "answers"
CSV_SEPARATOR = ";"
TEMPERATURE = 0.0


# LOGGING
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

# API FUNCTIONS
def check_ollama_api():
    if not OLLAMA_API_KEY:
        logger.error("[ERROR] OLLAMA_API_KEY not set. Set it as environment variable.")
        return False
    try:
        headers = {"Authorization": f"Bearer {OLLAMA_API_KEY}"}
        response = requests.get(f"{OLLAMA_API_BASE}/tags", headers=headers, timeout=10)
        if response.status_code == 200:
            logger.info("[OK] Ollama API is accessible")
            return True
        logger.error(f"[ERROR] API returned status {response.status_code}")
        return False
    except Exception as e:
        logger.error(f"[ERROR] Cannot connect to API: {str(e)}")
        return False

def call_ollama_api(model_name, prompt, max_retries=3):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OLLAMA_API_KEY}"
    }
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": TEMPERATURE}
    }
    url = f"{OLLAMA_API_BASE}/generate"

    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=120)
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            elif response.status_code == 404:
                raise ValueError(f"Model {model_name} not found")
            else:
                logger.warning(f"[WARNING] API status {response.status_code}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise Exception(f"API error: {response.status_code}")
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise Exception("Request timeout")
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise Exception("Connection error")
    raise Exception("Failed after all retries")

# FILE FUNCTIONS
def load_input_file():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Could not find {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, sep=CSV_SEPARATOR)
    df.columns = df.columns.str.strip().str.replace('\ufeff', '')
    logger.info(f"[OK] File loaded: {len(df)} rows")
    if "prompt" not in df.columns:
        raise ValueError("CSV must contain a 'prompt' column")

    # Log test_items if available
    if 'test_item' in df.columns:
        test_items = df['test_item'].tolist()
        logger.info(f"[INFO] Test items: {', '.join(map(str, test_items[:10]))}{'...' if len(test_items) > 10 else ''}")

    return df

def parse_prompt(raw_prompt):
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

def generate_answer(context, question, model_name):
    full_prompt = f"Context: {context}\nQuestion: {question}\n\nAnswer:"
    return call_ollama_api(model_name, full_prompt)

def process_row(idx, row, total_rows, model_name):
    try:
        raw_prompt = str(row.prompt)
        test_item = getattr(row, 'test_item', None)

        # Always show test_item in the log for better tracking
        item_str = f" [{test_item}]" if test_item else ""
        logger.info(f"[INFO] Processing Row {idx + 1}/{total_rows}{item_str}")
        logger.info(f"[PROMPT] {raw_prompt[:100]}..." if len(raw_prompt) > 100 else f"[PROMPT] {raw_prompt}")

        context, question = parse_prompt(raw_prompt)
        answer = generate_answer(context, question, model_name)

        logger.info(f"[ANSWER] {answer[:100]}..." if len(answer) > 100 else f"[ANSWER] {answer}")
        logger.info(f"[OK] Row {idx + 1}{item_str} completed successfully")
        return idx, answer
    except Exception as e:
        logger.error(f"[ERROR] Row {idx + 1}: {str(e)}")
        return idx, f"ERROR: {str(e)}"

# ANSWER MANAGEMENT
def get_output_file_path(model_name):
    safe_model_name = model_name.replace("/", "-").replace(":", "-")
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
                errors = sum(1 for a in existing_df['answer'] if pd.notna(a) and str(a).startswith("ERROR:"))
                pending = len(df) - completed

                logger.info(f"[INFO] Existing progress: {completed}/{len(df)} answers completed, {errors} errors, {pending} pending")

                # Show which test_items are pending if available
                if 'test_item' in df.columns and pending > 0:
                    pending_items = []
                    for idx, ans in enumerate(existing_df['answer']):
                        if pd.isna(ans) or str(ans).startswith("ERROR:"):
                            test_item = df.iloc[idx].get('test_item', f'Row {idx+1}')
                            pending_items.append(str(test_item))
                    if pending_items:
                        logger.info(f"[INFO] Pending items: {', '.join(pending_items[:20])}{'...' if len(pending_items) > 20 else ''}")

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


# PROCESSING
def process_prompts_sequential(df, model_name):
    total_rows = len(df)
    logger.info(f"[INFO] Starting sequential processing for {model_name}")
    answers, completed_count = load_existing_answers(df, model_name)

    for idx, row in enumerate(df.itertuples(index=False)):
        if pd.notna(answers[idx]) and not str(answers[idx]).startswith("ERROR:"):
            test_item = getattr(row, 'test_item', None)
            item_str = f" [{test_item}]" if test_item else ""
            logger.info(f"[SKIP] Row {idx + 1}/{total_rows}{item_str} already completed")
            continue
        idx_result, answer = process_row(idx, row, total_rows, model_name)
        answers[idx_result] = answer
        save_single_answer(df, answers, model_name, idx_result)

        # Show overall progress
        completed_so_far = sum(1 for a in answers if pd.notna(a) and not str(a).startswith("ERROR:"))
        errors_so_far = sum(1 for a in answers if pd.notna(a) and str(a).startswith("ERROR:"))
        logger.info(f"[PROGRESS] {model_name} - {idx + 1}/{total_rows} processed | {completed_so_far} successful | {errors_so_far} errors")

    logger.info(f"[OK] All rows processed for {model_name}")

def process_single_model(model_config, df):
    model_name = model_config["name"]
    logger.info(f"[INFO] ===== Starting: {model_name} =====")

    answers, completed_count = load_existing_answers(df, model_name)
    pending_count = sum(1 for a in answers if pd.isna(a) or str(a).startswith("ERROR:"))

    if pending_count == 0:
        logger.info(f"[SKIP] {model_name} already completed")
        return

    logger.info(f"[INFO] {model_name} has {pending_count} pending answers")
    process_prompts_sequential(df, model_name)


    logger.info(f"[OK] ===== Completed: {model_name} =====\n")

def process_all_models(df):
    total_models = len(MODELS)
    successful_models = []
    failed_models = []

    for idx, model_config in enumerate(MODELS, 1):
        model_name = model_config["name"]
        try:
            logger.info(f"[INFO] Processing model {idx}/{total_models}: {model_name}")
            process_single_model(model_config, df)
            successful_models.append(model_name)
        except Exception as e:
            failed_models.append((model_name, str(e)))
            logger.error(f"[ERROR] {model_name} failed: {str(e)}")

    if successful_models:
        logger.info(f"[OK] Successfully processed {len(successful_models)} models")
    if failed_models:
        logger.warning(f"[WARNING] {len(failed_models)} models failed")

    return successful_models, failed_models

# MAIN
def main():
    try:
        logger.info("[INFO] ===== Ollama Cloud API Processing Started =====")
        logger.info(f"[INFO] API: {OLLAMA_API_BASE}")
        logger.info(f"[INFO] Mode: Sequential")

        if not check_ollama_api():
            logger.error("[FATAL] Cannot access Ollama API")
            raise RuntimeError("Ollama API not accessible")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df = load_input_file()
        successful_models, failed_models = process_all_models(df)

        logger.info(f"[OK] Summary: {len(successful_models)} successful, {len(failed_models)} failed")
        
        # Show output files location
        if successful_models:
            logger.info(f"[INFO] Answer files saved in: {OUTPUT_DIR}")
            logger.info(f"[INFO] Each model has its own file: <model-name>_answers.csv")

        logger.info("[INFO] ===== Completed =====")
    except Exception as e:
        logger.error(f"[FATAL] Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main()

