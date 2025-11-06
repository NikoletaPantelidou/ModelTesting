"""
Script for processing prompts using OpenRouter API.
Includes complete logging and error handling.
"""

# ============================================================================
# IMPORTS
# ============================================================================
import os
import sys
import logging
import pandas as pd
import time
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from script_models_config import MODELS
from datetime import datetime
from clean_translations import clean_answer_file

# ============================================================================
# CONFIGURATION
# ============================================================================

# OpenRouter API Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")  # Obtener de variable de entorno
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
YOUR_SITE_URL = "http://localhost"  # Optional. Site URL for rankings on openrouter.ai
YOUR_SITE_NAME = "PyCharmMiscProject"  # Optional. Site title for rankings on openrouter.ai

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

INPUT_FILE = "prompts/prompts_all.csv"  # Input CSV file path
TEMPERATURE = 0  # Temperature for generation (0 = deterministic)
OUTPUT_DIR = "answers"  # Directory for output files
CSV_SEPARATOR = ";"
MAX_WORKERS = 4  # Number of parallel workers for processing prompts
REQUEST_DELAY = 2  # Delay in seconds between requests to avoid rate limiting

# ============================================================================
# FUNCTIONS
# ============================================================================

def check_api_key():
    """Check if API key is configured."""
    if not OPENROUTER_API_KEY:
        logger.error("[ERROR] OPENROUTER_API_KEY not found. Please set it as environment variable.")
        logger.error("[INFO] You can set it by running: set OPENROUTER_API_KEY=your_api_key")
        raise ValueError("OPENROUTER_API_KEY not configured")
    logger.info(f"[OK] API Key configured: {OPENROUTER_API_KEY[:10]}...")

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

def clean_multilingual_answer(answer, target_language):
    """Clean answers that may contain multiple languages, keeping only the target language."""
    import re

    if not answer or not target_language:
        return answer

    lang_lower = target_language.lower()

    if lang_lower != "english":
        # Remove English translations in parentheses at the end
        answer = re.sub(r'\s*\([^)]*[A-Z][^)]*\)\s*$', '', answer)

        # Remove English translations in parentheses anywhere in the text
        answer = re.sub(r'\s*\([^)]*(?:Yes|No|If|It|The|There|According|Otherwise)[^)]*\)', '', answer, flags=re.IGNORECASE)

        # Patterns to detect English translations added by the model
        english_patterns = [
            r'\n\n\s*(?:Translation|In English|English version|English answer)[\s:]+.*',
            r'\n\s*English[\s:]+.*',
            r'\n\n\s*[A-Z][^.]*\.\s*(?:[A-Z][^.]*\.)+',
            r'\n\n\s*La traducción directa.*',
            r'\n\n\s*La traducción.*',
            r'\n\s*La traducción directa.*',
        ]

        for pattern in english_patterns:
            answer = re.sub(pattern, '', answer, flags=re.IGNORECASE | re.DOTALL)

        # If the answer has two distinct paragraphs and the second looks like English, remove it
        paragraphs = answer.split('\n\n')
        if len(paragraphs) > 1:
            first = paragraphs[0].strip()
            if first and (first.endswith('.') or first.endswith('?') or first.endswith('!')):
                second = paragraphs[1].strip()
                if second and not any(char in second.lower() for char in ['à', 'è', 'é', 'ò', 'í', 'ñ', 'ç', 'ü', 'ú']):
                    answer = first

    return answer.strip()

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

def generate_answer_via_api(context, question, model_name, language=None):
    """Generate an answer using OpenRouter API."""

    # Map language names to native language names and strict instructions
    language_map = {
        "english": {
            "name": "English",
            "system": "You are a helpful assistant that answers ONLY in English.",
            "prompt_template": "Context: {context}\n\nQuestion: {question}\n\nProvide a direct answer in English only:"
        },
        "catalan": {
            "name": "català",
            "system": "Ets un assistent útil que respon NOMÉS en català.",
            "prompt_template": "Context: {context}\n\nPregunta: {question}\n\nRespon NOMÉS en català. NO proporcionis traduccions a l'anglès:"
        },
        "spanish": {
            "name": "español",
            "system": "Eres un asistente útil que responde SOLO en español.",
            "prompt_template": "Contexto: {context}\n\nPregunta: {question}\n\nResponde SOLO en español. NO proporciones traducciones al inglés:"
        },
        "italian": {
            "name": "italiano",
            "system": "Sei un assistente utile che risponde SOLO in italiano.",
            "prompt_template": "Contesto: {context}\n\nDomanda: {question}\n\nRispondi SOLO in italiano. NON fornire traduzioni in inglese:"
        }
    }

    # Get language instruction
    language_info = None
    if language:
        lang_lower = str(language).strip().lower()
        language_info = language_map.get(lang_lower)

    # Build the user message
    if language_info:
        user_message = language_info['prompt_template'].format(context=context, question=question)
        system_message = language_info['system']
    else:
        user_message = f"Context: {context}\nQuestion: {question}\n\nAnswer:"
        system_message = "You are a helpful assistant that answers questions based on the given context."

    try:
        # Prepare API request
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": YOUR_SITE_URL,
            "X-Title": YOUR_SITE_NAME,
        }

        data = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ]
        }

        # Add temperature if > 0
        if TEMPERATURE > 0:
            data["temperature"] = TEMPERATURE

        # Make API request
        response = requests.post(
            url=OPENROUTER_API_URL,
            headers=headers,
            data=json.dumps(data),
            timeout=60
        )

        response.raise_for_status()

        result = response.json()

        # Extract answer from response
        if 'choices' in result and len(result['choices']) > 0:
            answer = result['choices'][0]['message']['content'].strip()

            # Clean up multilingual responses
            if language:
                answer = clean_multilingual_answer(answer, language)

            return answer
        else:
            logger.error(f"[ERROR] Unexpected API response format: {result}")
            return f"ERROR: Unexpected response format"

    except requests.exceptions.Timeout:
        logger.error(f"[ERROR] Request timeout for model {model_name}")
        return "ERROR: Request timeout"
    except requests.exceptions.RequestException as e:
        logger.error(f"[ERROR] API request failed: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                logger.error(f"[ERROR] API error details: {error_detail}")
            except:
                logger.error(f"[ERROR] Response status code: {e.response.status_code}")
        return f"ERROR: {str(e)}"
    except Exception as e:
        logger.error(f"[ERROR] Unexpected error: {str(e)}")
        return f"ERROR: {str(e)}"

def process_row(idx, row, total_rows, model_name):
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

        # Get language from the row
        language = None
        if hasattr(row, 'language'):
            language = getattr(row, 'language')
            logger.info(f"[LANGUAGE] Detected language: {language}")
        elif hasattr(row, 'lang'):
            language = getattr(row, 'lang')
            logger.info(f"[LANGUAGE] Detected language: {language}")

        context, question = parse_prompt(raw_prompt)
        answer = generate_answer_via_api(context, question, model_name, language)

        # Add delay to avoid rate limiting
        time.sleep(REQUEST_DELAY)

        logger.info(f"[OK] Row {idx + 1} completed")
        return idx, answer

    except Exception as e:
        logger.error(f"[ERROR] Error processing row {idx + 1}: {str(e)}")
        return idx, f"ERROR: {str(e)}"

def process_prompts_sequential(df, model_name):
    """Process all prompts sequentially."""
    total_rows = len(df)

    logger.info(f"[INFO] Starting sequential processing for {model_name}")

    # Load existing answers if they exist
    answers, completed_count = load_existing_answers(df, model_name)

    if completed_count > 0:
        logger.info(f"[INFO] Resuming from row {completed_count + 1}")

    for idx, row in enumerate(df.itertuples(index=False)):
        # Skip if a valid answer already exists
        if pd.notna(answers[idx]) and not str(answers[idx]).startswith("ERROR:"):
            logger.info(f"[SKIP] Row {idx + 1}/{total_rows} already completed, skipping")
            continue

        idx_result, answer = process_row(idx, row, total_rows, model_name)
        answers[idx] = answer

        # Save immediately after generating each answer
        save_single_answer(df, answers, model_name, idx)

        logger.info(f"[PROGRESS] {model_name} - {idx + 1}/{total_rows} rows completed")

        # Add delay between requests
        if idx < total_rows - 1:
            logger.info(f"[INFO] Waiting {REQUEST_DELAY} seconds before next request...")
            time.sleep(REQUEST_DELAY)

    logger.info(f"[OK] All rows processed for {model_name}")

def process_prompts_parallel(df, model_name):
    """Process all prompts in parallel using ThreadPoolExecutor."""
    total_rows = len(df)

    logger.info(f"[INFO] Starting parallel processing for {model_name} with {MAX_WORKERS} workers")

    # Load existing answers if they exist
    answers, completed_count = load_existing_answers(df, model_name)

    if completed_count > 0:
        logger.info(f"[INFO] Resuming from row {completed_count + 1}")

    # Create a list of pending tasks
    pending_tasks = []
    for idx, row in enumerate(df.itertuples(index=False)):
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
        future_to_idx = {
            executor.submit(process_row, idx, row, total_rows, model_name): idx
            for idx, row in pending_tasks
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                idx_result, answer = future.result()
                answers[idx_result] = answer

                save_single_answer(df, answers, model_name, idx_result)

                completed_tasks += 1
                logger.info(f"[PROGRESS] {model_name} - {completed_tasks}/{len(pending_tasks)} pending prompts completed ({completed_tasks + completed_count}/{total_rows} total)")

            except Exception as e:
                logger.error(f"[ERROR] Task for row {idx + 1} failed: {str(e)}")
                answers[idx] = f"ERROR: {str(e)}"

    logger.info(f"[OK] All rows processed for {model_name}")

def get_output_file_path(model_name):
    """Get the output file path for a model."""
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
                completed = existing_df['answer'].notna().sum()
                logger.info(f"[INFO] Existing progress: {completed}/{len(df)} answers completed")
                return existing_df['answer'].tolist(), completed
            else:
                logger.info(f"[INFO] Existing file found but incomplete or size mismatch, starting fresh")
        except Exception as e:
            logger.warning(f"[WARNING] Could not load existing answers: {str(e)}")

    return [None] * len(df), 0

def check_pending_answers(df, model_name):
    """Check if there are pending answers to complete for a model."""
    answers, completed_count = load_existing_answers(df, model_name)

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

    logger.info(f"[INFO] ===== Starting processing for model: {model_name} =====")

    # Check if there are pending answers
    pending_count, _ = check_pending_answers(df, model_name)

    if pending_count == 0:
        logger.info(f"[SKIP] Model {model_name} has all answers completed, skipping")
        return

    logger.info(f"[INFO] Model {model_name} has {pending_count} pending answers")

    # Use parallel processing for better performance
    process_prompts_parallel(df, model_name)

    logger.info(f"[OK] ===== Completed processing for model: {model_name} =====")

    # Automatically clean translations from the answer file
    logger.info(f"[INFO] Cleaning translations from answer file for {model_name}...")
    output_file = get_output_file_path(model_name)
    try:
        clean_answer_file(output_file)
        logger.info(f"[OK] Translations cleaned successfully for {model_name}")
    except Exception as e:
        logger.warning(f"[WARNING] Error cleaning translations for {model_name}: {str(e)}")

    logger.info("")

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
        logger.info(f"[INFO] OpenRouter Free Models Script Starting...")
        logger.info(f"[INFO] Max workers for parallel processing: {MAX_WORKERS}")
        logger.info(f"[INFO] Request delay: {REQUEST_DELAY} seconds")

        # Check API key
        check_api_key()

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

