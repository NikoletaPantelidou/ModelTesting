"""
Script para procesamiento paralelo de prompts usando modelos de transformers.
Incluye logging completo, manejo de errores y procesamiento multi-thread.
"""

# ============================================================================
# IMPORTS
# ============================================================================
import os
import sys
import logging
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
# Configurar handlers manualmente para que INFO/OK salgan en stdout (blanco)
file_handler = logging.FileHandler('script_execution.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)

# StreamHandler usando stdout explícitamente para evitar texto rojo
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Formato del log
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Configurar logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)


# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
USE_QA_PIPELINE = False  # Set False if your model doesn't do QA
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEMPERATURE = 0.0  # Temperature for generation (0.0 = deterministic)
MAX_WORKERS = 5  # Number of threads for parallel processing
INPUT_FILE = "example.csv"
OUTPUT_FILE = "answers.csv"
CSV_SEPARATOR = ";"


# ============================================================================
# GLOBAL VARIABLES
# ============================================================================
qa_model = None
tokenizer = None
model = None
answers_lock = Lock()


# ============================================================================
# FUNCTIONS
# ============================================================================

def load_model():
    """Load the AI model (QA pipeline or generative model)."""
    global qa_model, tokenizer, model

    logger.info(f"[INFO] Loading model: {MODEL_NAME}")
    logger.info(f"[INFO] Using device: {DEVICE}")

    try:
        if USE_QA_PIPELINE:
            qa_model = pipeline(
                "question-answering",
                model=MODEL_NAME,
                tokenizer=MODEL_NAME,
                device=0 if DEVICE == "cuda" else -1
            )
            logger.info("[OK] QA pipeline loaded successfully")
        else:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            # Configurar pad_token para evitar warnings
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id

            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
            # Configurar pad_token_id en el modelo también
            model.config.pad_token_id = tokenizer.pad_token_id
            logger.info("[OK] Model and tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"[ERROR] Error loading model: {str(e)}")
        raise


def load_input_file():
    """Load and validate the input CSV file."""
    logger.info(f"[INFO] Reading input file: {INPUT_FILE}")

    if not os.path.exists(INPUT_FILE):
        logger.error(f"[ERROR] Could not find {INPUT_FILE}")
        raise FileNotFoundError(
            f"[ERROR] Could not find {INPUT_FILE}. "
            "Please place it in the same folder as this script."
        )

    try:
        df = pd.read_csv(INPUT_FILE, sep=CSV_SEPARATOR)
        df.columns = df.columns.str.strip().str.replace('\ufeff', '')

        logger.info(f"[OK] File loaded successfully. Rows: {len(df)}")
        logger.info(f"[INFO] Columns detected: {df.columns.tolist()}")
        print("[OK] Columns detected:", df.columns.tolist())

        if "prompt" not in df.columns:
            logger.error("[ERROR] CSV must contain a 'prompt' column")
            raise ValueError("[ERROR] CSV must contain a 'prompt' column.")

        return df
    except Exception as e:
        logger.error(f"[ERROR] Error reading CSV file: {str(e)}")
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


def generate_answer(context, question):
    """Generate an answer using the loaded model."""
    full_prompt = f"Context: {context}\nQuestion: {question}\n\nAnswer:"

    if USE_QA_PIPELINE:
        result = qa_model(question=question, context=context)
        if isinstance(result, list):
            answer = result[0]["answer"]
        else:
            answer = result["answer"]
    else:
        inputs = tokenizer(full_prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=TEMPERATURE if TEMPERATURE > 0 else None,
                do_sample=TEMPERATURE > 0,
                pad_token_id=tokenizer.pad_token_id  # Evitar warning de padding
            )
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = answer.replace(full_prompt, "").strip()

    return answer


def process_row(idx, row, total_rows):
    """Process a single row and return the answer."""
    try:
        logger.info(f"[INFO] Processing row {idx + 1}/{total_rows}")

        raw_prompt = str(row["prompt"])
        context, question = parse_prompt(raw_prompt)

        logger.debug(f"[DEBUG] Row {idx + 1} - Context: {context[:50]}...")
        logger.debug(f"[DEBUG] Row {idx + 1} - Question: {question}")

        answer = generate_answer(context, question)

        logger.info(f"[OK] Row {idx + 1} completed")
        return idx, answer

    except Exception as e:
        logger.error(f"[ERROR] Error processing row {idx + 1}: {str(e)}")
        return idx, f"ERROR: {str(e)}"


def process_prompts_parallel(df):
    """Process all prompts in parallel using ThreadPoolExecutor."""
    answers = [None] * len(df)
    total_rows = len(df)

    logger.info(f"[INFO] Starting parallel processing with {MAX_WORKERS} workers")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_row, idx, row, total_rows): idx
            for idx, row in df.iterrows()
        }

        completed_count = 0
        for future in as_completed(futures):
            try:
                idx, answer = future.result()
                with answers_lock:
                    answers[idx] = answer
                    completed_count += 1
                    logger.info(f"[PROGRESS] {completed_count}/{total_rows} rows completed")
            except Exception as e:
                logger.error(f"[ERROR] Unexpected error in thread execution: {str(e)}")

    logger.info("[OK] All rows processed")
    return answers


def save_output(df, answers):
    """Save the results to a CSV file."""
    try:
        df["answer"] = answers
        df.to_csv(OUTPUT_FILE, index=False)

        logger.info(f"[OK] Done! Answers saved to {OUTPUT_FILE}")
        print(f"\n[OK] Done! Answers saved to {OUTPUT_FILE}")
        print("\n[INFO] Preview of results:")
        print(df.head())
    except Exception as e:
        logger.error(f"[ERROR] Error saving output file: {str(e)}")
        raise


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    try:
        # Step 1: Load model
        load_model()

        # Step 2: Load input file
        df = load_input_file()

        # Step 3: Process prompts in parallel
        answers = process_prompts_parallel(df)

        # Step 4: Save output
        save_output(df, answers)

    except Exception as e:
        logger.error(f"[FATAL] Fatal error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
