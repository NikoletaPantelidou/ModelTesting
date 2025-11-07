"""
Script to process prompts using OpenRouter (free models) via OpenAI-compatible client.
This mirrors the Ollama script logic but calls the OpenRouter API.
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Use requests to call OpenRouter's API
import requests
import time
import threading

from script_models_config import MODELS

# Make file paths relative to this script's directory so it can be run from any CWD
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuration
INPUT_FILE = os.path.join(BASE_DIR, "prompts", "prompts_all.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "answers")
CSV_SEPARATOR = ","
MAX_WORKERS = 4
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
BASE_URL = "https://openrouter.ai/api/v1"
# Optional metadata for rankings on openrouter.ai
OPENROUTER_REFERER = os.getenv("OPENROUTER_REFERER")
OPENROUTER_SITE_TITLE = os.getenv("OPENROUTER_SITE_TITLE")

# Rate limiting: minimal delay between consecutive HTTP requests (across threads)
# Configure via environment variable OPENROUTER_REQUEST_DELAY (seconds). Default 30s (0.5 minute).
REQUEST_DELAY_SECONDS = float(os.getenv("OPENROUTER_REQUEST_DELAY", "30"))
# A lock and timestamp to serialize requests so we don't burst the API
_rate_limit_lock = threading.Lock()
_last_request_time = 0.0

# Logging
os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = os.path.join(BASE_DIR, f'logs/openrouter_execution_{timestamp}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Utilities

def load_input_file():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Could not find {INPUT_FILE}")

    # Try reading with the configured separator first. If it fails, try to auto-detect.
    try:
        if CSV_SEPARATOR:
            df = pd.read_csv(INPUT_FILE, sep=CSV_SEPARATOR)
        else:
            # Let pandas sniff the separator (requires python engine)
            df = pd.read_csv(INPUT_FILE, sep=None, engine='python')
    except Exception as e:
        logger.warning(f"Initial CSV read failed with {e}; trying to auto-detect separator")
        # Try to auto-detect delimiter using python engine which supports regex/seps
        try:
            df = pd.read_csv(INPUT_FILE, sep=None, engine='python')
        except Exception as e2:
            # As a last resort, try semicolon which is common in exported CSVs
            try:
                df = pd.read_csv(INPUT_FILE, sep=';')
            except Exception:
                raise RuntimeError(f"Failed to parse CSV file: {e2}")

    # Normalize column names and handle BOM
    df.columns = df.columns.str.strip().str.replace('\ufeff', '')

    # Ensure there's a 'prompt' column; if not, try to guess that the last column is the prompt
    if 'prompt' not in df.columns:
        if df.shape[1] >= 1:
            # If there are at least 3 columns (test_item, language, prompt), take the last as prompt
            df = df.rename(columns={df.columns[-1]: 'prompt'})
            logger.info("Renamed last column to 'prompt' as fallback")
        else:
            raise ValueError("CSV must contain a 'prompt' column")

    return df


def get_output_file_path(model_name):
    safe_name = model_name.replace('/', '-').replace(':', '-')
    return os.path.join(OUTPUT_DIR, f"{safe_name}_answers.csv")


def load_existing_answers(df, model_name):
    output_file = get_output_file_path(model_name)
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        if 'answer' in existing_df.columns and len(existing_df) == len(df):
            return existing_df['answer'].tolist(), existing_df['answer'].notna().sum()
    return [None] * len(df), 0


def save_single_answer(df, answers, model_name, current_idx):
    output_file = get_output_file_path(model_name)
    df_temp = df.assign(answer=answers)
    df_temp.to_csv(output_file, index=False)
    logger.info(f"Saved answer {current_idx + 1} to {output_file}")


def parse_prompt(raw_prompt):
    sentences = raw_prompt.split('.')
    joined_text = '.'.join(sentences)
    last_comma_idx = joined_text.rfind(',')

    if last_comma_idx != -1:
        question = joined_text[last_comma_idx + 1:].strip()
        question = question[0].upper() + question[1:] if question else question
        context = joined_text[:last_comma_idx].strip()
    else:
        context = '.'.join(sentences[:-1]).strip() + '.'
        question = sentences[-1].strip()

    if not question.endswith('?'):
        question += '?'

    return context, question


def call_openrouter(model_name, context, question):
    """Call OpenRouter via HTTP POST using requests. Returns the text content of the first choice."""
    # Enforce a global minimal delay between requests to avoid spamming the API
    if REQUEST_DELAY_SECONDS and REQUEST_DELAY_SECONDS > 0:
        global _last_request_time
        with _rate_limit_lock:
            now = time.monotonic()
            elapsed = now - _last_request_time
            if elapsed < REQUEST_DELAY_SECONDS:
                to_sleep = REQUEST_DELAY_SECONDS - elapsed
                logger.debug(f"Rate limiter sleeping for {to_sleep:.3f}s")
                time.sleep(to_sleep)
            # update last request time to now (after sleep)
            _last_request_time = time.monotonic()

    full_message = f"Context: {context}\nQuestion: {question}\n\nAnswer:"

    url = f"{BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    # Optional headers for openrouter ranking
    if OPENROUTER_REFERER:
        headers["HTTP-Referer"] = OPENROUTER_REFERER
    if OPENROUTER_SITE_TITLE:
        headers["X-Title"] = OPENROUTER_SITE_TITLE

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": full_message}]
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
    except requests.RequestException as e:
        raise RuntimeError(f"Network error calling OpenRouter: {e}")

    if resp.status_code != 200:
        # Try to include response body for easier debugging
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        raise RuntimeError(f"OpenRouter returned status {resp.status_code}: {body}")

    try:
        data = resp.json()
        # Typical shape: { 'choices': [ { 'message': { 'content': '...' } } ] }
        content = data['choices'][0]['message']['content']
        return content
    except Exception as e:
        raise RuntimeError(f"Failed to parse OpenRouter response: {e} -- raw: {resp.text}")


def process_row(idx, row, total_rows, client_model_name):
    try:
        raw_prompt = str(row.prompt)
        logger.info(f"Processing {idx + 1}/{total_rows}")
        context, question = parse_prompt(raw_prompt)
        answer = call_openrouter(client_model_name, context, question)
        logger.info(f"Row {idx + 1} completed")
        return idx, answer
    except Exception as e:
        logger.error(f"Error processing row {idx + 1}: {e}")
        return idx, f"ERROR: {e}"


def process_prompts_parallel(df, model_name):
    total_rows = len(df)
    answers, completed_count = load_existing_answers(df, model_name)

    pending = []
    for idx, row in enumerate(df.itertuples(index=False)):
        if pd.notna(answers[idx]) and not str(answers[idx]).startswith('ERROR:'):
            continue
        pending.append((idx, row))

    if not pending:
        logger.info('No pending prompts')
        return

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        fut_to_idx = {executor.submit(process_row, idx, row, total_rows, model_name): idx for idx, row in pending}
        completed = 0
        for fut in as_completed(fut_to_idx):
            idx = fut_to_idx[fut]
            try:
                idx_result, answer = fut.result()
                answers[idx_result] = answer
                save_single_answer(df, answers, model_name, idx_result)
                completed += 1
                logger.info(f"Progress: {completed}/{len(pending)} pending completed")
            except Exception as e:
                logger.error(f"Task failed for row {idx + 1}: {e}")
                answers[idx] = f"ERROR: {e}"


def process_single_model(model_config, df):
    model_name = model_config['name']
    pending_count, _ = load_existing_answers(df, model_name)
    logger.info(f"Processing model {model_name}")
    process_prompts_parallel(df, model_name)


def process_all_models(df):
    successes = []
    fails = []
    for model in MODELS:
        try:
            process_single_model(model, df)
            successes.append(model['name'])
        except Exception as e:
            fails.append((model['name'], str(e)))
    return successes, fails


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_input_file()
    successes, fails = process_all_models(df)
    logger.info(f"Done. Successes: {len(successes)}, Fails: {len(fails)}")


if __name__ == '__main__':
    main()
