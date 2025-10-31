"""
Script for sequential processing of prompts using Ollama API.
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
INPUT_FILE = "prompts/example.csv"
OUTPUT_DIR = "answers"
CSV_SEPARATOR = ";"
TEMPERATURE = 0.0
MAX_TOKENS = 300

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
    try:
        response = requests.get(f"{OLLAMA_API_BASE}/tags", timeout=10)
        if response.status_code == 200:
            logger.info("[OK] Ollama API is accessible")
            return True
        logger.error(f"[ERROR] API returned status {response.status_code}")
        return False
    except Exception as e:
        logger.error(f"[ERROR] Cannot connect to API: {str(e)}")
        return False

def call_ollama_api(model_name, prompt, max_retries=3):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS}
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
        if test_item:
            logger.info(f"[INFO] Row {idx + 1}/{total_rows} - test_item: {test_item}")
        else:
            logger.info(f"[INFO] Row {idx + 1}/{total_rows}")
        logger.info(f"[PROMPT] {raw_prompt[:100]}..." if len(raw_prompt) > 100 else f"[PROMPT] {raw_prompt}")
        context, question = parse_prompt(raw_prompt)
        answer = generate_answer(context, question, model_name)
        logger.info(f"[OK] Row {idx + 1} completed")
        return idx, answer
    except Exception as e:
        logger.error(f"[ERROR] Row {idx + 1}: {str(e)}")
        return idx, f"ERROR: {str(e)}"

# ANSWER MANAGEMENT
def get_output_file_path(model_name):
    safe_model_name = model_name.replace("/", "-").replace(":", "-")
    return os.path.join(OUTPUT_DIR, f"{safe_model_name}_answers.csv")

def load_existing_answers(df, model_name):
    output_file = get_output_file_path(model_name)
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_csv(output_file)
            if 'answer' in existing_df.columns and len(existing_df) == len(df):
                completed = existing_df['answer'].notna().sum()
                logger.info(f"[INFO] Found existing: {completed}/{len(df)} completed")
                return existing_df['answer'].tolist(), completed
        except Exception as e:
            logger.warning(f"[WARNING] Could not load existing: {str(e)}")
    return [None] * len(df), 0

def save_single_answer(df, answers, model_name, current_idx):
    output_file = get_output_file_path(model_name)
    df_temp = df.assign(answer=answers)
    df_temp.to_csv(output_file, index=False)
    logger.info(f"[OK] Answer {current_idx + 1} saved")

# PROCESSING
def process_prompts_sequential(df, model_name):
    total_rows = len(df)
    logger.info(f"[INFO] Starting sequential processing for {model_name}")
    answers, completed_count = load_existing_answers(df, model_name)

    for idx, row in enumerate(df.itertuples(index=False)):
        if pd.notna(answers[idx]) and not str(answers[idx]).startswith("ERROR:"):
            logger.info(f"[SKIP] Row {idx + 1}/{total_rows} already completed")
            continue
        idx_result, answer = process_row(idx, row, total_rows, model_name)
        answers[idx_result] = answer
        save_single_answer(df, answers, model_name, idx_result)
        logger.info(f"[PROGRESS] {model_name} - {idx + 1}/{total_rows} completed")

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
        logger.info("[INFO] ===== Ollama API Processing Started =====")
        logger.info(f"[INFO] API: {OLLAMA_API_BASE}")
        logger.info(f"[INFO] Mode: Sequential")

        if not check_ollama_api():
            logger.error("[FATAL] Cannot access Ollama API")
            raise RuntimeError("Ollama API not accessible")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df = load_input_file()
        successful_models, failed_models = process_all_models(df)

        logger.info(f"[OK] Summary: {len(successful_models)} successful, {len(failed_models)} failed")
        logger.info("[INFO] ===== Completed =====")
    except Exception as e:
        logger.error(f"[FATAL] Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main()

