"""
Script for sequential processing of prompts using Helicone AI Gateway.
Helicone provides monitoring and logging for LLM API calls.
"""

import os
import sys
import logging
import pandas as pd
import time
from openai import OpenAI
from script_models_config import MODELS
from datetime import datetime

# CONFIGURATION
HELICONE_API_BASE = "https://ai-gateway.helicone.ai"
HELICONE_API_KEY = os.getenv("HELICONE_API_KEY", "sk-helicone-4ml2yby-dr4u7hq-rmzewgi-k2svq4q")
INPUT_FILE = "prompts/example.csv"
OUTPUT_DIR = "answers"
CSV_SEPARATOR = ";"
TEMPERATURE = 0.0
MAX_TOKENS = 200
DELAY_BETWEEN_REQUESTS = 2.0  # Seconds to wait between API calls to avoid rate limits

# Initialize OpenAI client with Helicone gateway
client = OpenAI(
    base_url=HELICONE_API_BASE,
    api_key=HELICONE_API_KEY
)

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
def check_helicone_api():
    if not HELICONE_API_KEY:
        logger.error("[ERROR] HELICONE_API_KEY not set. Set it as environment variable.")
        return False

    logger.info("[INFO] Using Helicone API Key for authentication")

    try:
        # Test with a simple API call
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        if response and response.choices:
            logger.info("[OK] Helicone API is accessible")
            return True
        logger.error("[ERROR] API returned unexpected response")
        return False
    except Exception as e:
        error_msg = str(e)

        # Check for insufficient credits error
        if "429" in error_msg or "Too Many Requests" in error_msg:
            if "Insufficient credits" in error_msg or "Insufficient balance" in error_msg:
                logger.error("[ERROR] ⚠️  INSUFFICIENT CREDITS in Helicone account")
                logger.error("[ERROR] You need to add credits to your Helicone account")
                logger.error("[ERROR] Visit: https://www.helicone.ai/ -> Billing -> Add Credits")
            else:
                logger.error("[ERROR] Rate limit exceeded - too many requests")
                logger.error("[ERROR] Consider increasing DELAY_BETWEEN_REQUESTS in script.py")

        logger.error(f"[ERROR] Cannot connect to API: {error_msg}")
        return False

def call_helicone_api(model_name, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )

            if response and response.choices:
                answer = response.choices[0].message.content.strip()

                # Check for empty responses
                if not answer:
                    logger.warning(f"[WARNING] Model {model_name} returned an empty response")
                    return "[EMPTY RESPONSE]"

                return answer
            else:
                raise Exception("No response from API")

        except Exception as e:
            error_msg = str(e)

            # Check for insufficient credits
            if "429" in error_msg or "Too Many Requests" in error_msg:
                if "Insufficient credits" in error_msg or "Insufficient balance" in error_msg:
                    logger.error(f"[ERROR] ⚠️  INSUFFICIENT CREDITS in Helicone account")
                    logger.error(f"[ERROR] Cannot continue - please add credits at https://www.helicone.ai/")
                    raise Exception("Insufficient credits in Helicone account. Please add credits to continue.")
                else:
                    # Rate limit - wait longer before retrying
                    wait_time = (2 ** attempt) * 5  # 5s, 10s, 20s
                    logger.warning(f"[WARNING] Rate limit hit. Waiting {wait_time}s before retry...")
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
                        continue

            # Check for model not found
            if "model" in error_msg.lower() and "not found" in error_msg.lower():
                raise ValueError(f"Model {model_name} not found")

            logger.warning(f"[WARNING] Attempt {attempt + 1}/{max_retries} failed: {error_msg}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise Exception(f"API error after {max_retries} attempts: {error_msg}")

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
    return call_helicone_api(model_name, full_prompt)

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

        # Add delay between requests to avoid rate limits
        if idx < total_rows - 1:  # Don't wait after the last request
            logger.info(f"[INFO] Waiting {DELAY_BETWEEN_REQUESTS}s before next request...")
            time.sleep(DELAY_BETWEEN_REQUESTS)

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
        logger.info("[INFO] ===== Helicone API Processing Started =====")
        logger.info(f"[INFO] API: {HELICONE_API_BASE}")
        logger.info(f"[INFO] Mode: Sequential")

        if not check_helicone_api():
            logger.error("[FATAL] Cannot access Helicone API")
            raise RuntimeError("Helicone API not accessible")

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

