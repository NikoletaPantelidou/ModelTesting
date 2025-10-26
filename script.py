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
from datetime import datetime

# Desactivar warnings de HuggingFace Hub
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = '1'

import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
# Configurar handlers manualmente para que INFO/OK salgan en stdout (blanco)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f'logs/execution_{timestamp}.log'

os.makedirs('logs', exist_ok=True)

file_handler = logging.FileHandler(log_filename, encoding='utf-8')
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
# Lista de modelos a procesar (cada modelo tiene su configuración)
MODELS = [
    {"name": "mistralai/Mistral-7B-Instruct-v0.2", "use_qa_pipeline": False, "trust_remote_code": False},
    {"name": "microsoft/phi-4", "use_qa_pipeline": False, "trust_remote_code": False},
    {"name": "tiiuae/falcon-7b-instruct", "use_qa_pipeline": False, "trust_remote_code": False},  # Falcon ya está integrado en transformers
    {"name": "arcee-ai/Arcee-Blitz", "use_qa_pipeline": False, "trust_remote_code": False},
    {"name": "arcee-ai/Virtuoso-Lite", "use_qa_pipeline": False, "trust_remote_code": False},
    {"name": "inclusionAI/Ling-1T-FP8", "use_qa_pipeline": False, "trust_remote_code": True},
    {"name": "moonshotai/Kimi-K2-Instruct-0905", "use_qa_pipeline": False, "trust_remote_code": True},
    {"name": "distilbert-base-cased-distilled-squad", "use_qa_pipeline": True, "trust_remote_code": False},

    # Modelos que requieren autenticación (necesitas hacer login con huggingface-cli):
    # {"name": "google/gemma-3-1b-it", "use_qa_pipeline": False, "trust_remote_code": False},  # Requiere acceso gated
]

INPUT_FILE = "prompts/example.csv"  # Input CSV file path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEMPERATURE = 0.0  # Temperature for generation (0.0 = deterministic)
MAX_WORKERS = 4  # Number of threads for parallel processing per model
MAX_MODEL_WORKERS = 1  # Number of models to process in parallel (max 2 to avoid overload)
OUTPUT_DIR = "answers"  # Directory for output files
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

def load_model(model_name, use_qa_pipeline=False, trust_remote_code=False):
    """Load the AI model (QA pipeline or generative model)."""
    logger.info(f"[INFO] Loading model: {model_name}")
    logger.info(f"[INFO] Using device: {DEVICE}")
    logger.info(f"[INFO] Use QA pipeline: {use_qa_pipeline}")
    logger.info(f"[INFO] Trust remote code: {trust_remote_code}")

    try:
        if use_qa_pipeline:
            qa_model = pipeline(
                "question-answering",
                model=model_name,
                tokenizer=model_name,
                device=0 if DEVICE == "cuda" else -1,
                trust_remote_code=trust_remote_code
            )
            logger.info(f"[OK] QA pipeline loaded successfully for {model_name}")
            return {'type': 'qa', 'model': qa_model, 'tokenizer': None}
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code
            )
            # Configurar pad_token para evitar warnings
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code
            ).to(DEVICE)
            # Configurar pad_token_id en el modelo también
            model.config.pad_token_id = tokenizer.pad_token_id
            logger.info(f"[OK] Model and tokenizer loaded successfully for {model_name}")
            return {'type': 'generative', 'model': model, 'tokenizer': tokenizer}
    except Exception as e:
        logger.error(f"[ERROR] Error loading model {model_name}: {str(e)}")
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


def generate_answer(context, question, model_dict):
    """Generate an answer using the loaded model."""
    full_prompt = f"Context: {context}\nQuestion: {question}\n\nAnswer:"

    if model_dict['type'] == 'qa':
        qa_model = model_dict['model']
        result = qa_model(question=question, context=context)
        if isinstance(result, list):
            answer = result[0]["answer"]
        else:
            answer = result["answer"]
    else:
        model = model_dict['model']
        tokenizer = model_dict['tokenizer']
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


def process_row(idx, row, total_rows, model_dict):
    """Process a single row and return the answer."""
    try:
        logger.info(f"[INFO] Processing row {idx + 1}/{total_rows}")

        raw_prompt = str(row["prompt"])
        context, question = parse_prompt(raw_prompt)

        logger.debug(f"[DEBUG] Row {idx + 1} - Context: {context[:50]}...")
        logger.debug(f"[DEBUG] Row {idx + 1} - Question: {question}")

        answer = generate_answer(context, question, model_dict)

        logger.info(f"[OK] Row {idx + 1} completed")
        return idx, answer

    except Exception as e:
        logger.error(f"[ERROR] Error processing row {idx + 1}: {str(e)}")
        return idx, f"ERROR: {str(e)}"


def process_prompts_parallel(df, model_dict, model_name):
    """Process all prompts in parallel using ThreadPoolExecutor."""
    answers = [None] * len(df)
    total_rows = len(df)

    logger.info(f"[INFO] Starting parallel processing for {model_name} with {MAX_WORKERS} workers")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_row, idx, row, total_rows, model_dict): idx
            for idx, row in df.iterrows()
        }

        completed_count = 0
        for future in as_completed(futures):
            try:
                idx, answer = future.result()
                with answers_lock:
                    answers[idx] = answer
                    completed_count += 1
                    logger.info(f"[PROGRESS] {model_name} - {completed_count}/{total_rows} rows completed")
            except Exception as e:
                logger.error(f"[ERROR] Unexpected error in thread execution: {str(e)}")

    logger.info(f"[OK] All rows processed for {model_name}")
    return answers


def save_output(df, answers, model_name):
    """Save the results to a CSV file."""
    try:
        # Crear nombre de archivo seguro (reemplazar / por -)
        safe_model_name = model_name.replace("/", "-")
        output_file = os.path.join(OUTPUT_DIR, f"{safe_model_name}_answers.csv")

        df_copy = df.copy()
        df_copy["answer"] = answers
        df_copy.to_csv(output_file, index=False)

        logger.info(f"[OK] Done! Answers saved to {output_file}")
        print(f"\n[OK] Done! Answers for {model_name} saved to {output_file}")
        print("\n[INFO] Preview of results:")
        print(df_copy.head())
    except Exception as e:
        logger.error(f"[ERROR] Error saving output file for {model_name}: {str(e)}")
        raise


def process_single_model(model_config, df):
    """Process prompts with a single model."""
    model_name = model_config["name"]
    use_qa_pipeline = model_config.get("use_qa_pipeline", False)
    trust_remote_code = model_config.get("trust_remote_code", False)

    try:
        logger.info(f"[INFO] ===== Starting processing for model: {model_name} =====")

        # Step 1: Load model
        model_dict = load_model(model_name, use_qa_pipeline, trust_remote_code)

        # Step 2: Process prompts in parallel
        answers = process_prompts_parallel(df, model_dict, model_name)

        # Step 3: Save output
        save_output(df, answers, model_name)

        logger.info(f"[OK] ===== Completed processing for model: {model_name} =====\n")

    except Exception as e:
        logger.error(f"[ERROR] Error processing model {model_name}: {str(e)}")
        raise


def process_models_parallel(df):
    """Process all models in parallel using ThreadPoolExecutor."""
    total_models = len(MODELS)
    logger.info(f"[INFO] Starting parallel model processing with {MAX_MODEL_WORKERS} workers")

    successful_models = []
    failed_models = []

    with ThreadPoolExecutor(max_workers=MAX_MODEL_WORKERS) as executor:
        futures = {
            executor.submit(process_single_model, model_config, df): model_config["name"]
            for model_config in MODELS
        }

        completed_count = 0
        for future in as_completed(futures):
            model_name = futures[future]
            try:
                future.result()
                completed_count += 1
                successful_models.append(model_name)
                logger.info(f"[PROGRESS] Models completed successfully: {completed_count}/{total_models}")
                print(f"\n[OK] Model {model_name} completed successfully ({completed_count}/{total_models})")
            except Exception as e:
                failed_models.append((model_name, str(e)))
                logger.error(f"[ERROR] Model {model_name} failed: {str(e)}")
                print(f"\n[ERROR] Model {model_name} failed. Check logs for details.")

    logger.info(f"[OK] Processing finished: {len(successful_models)} successful, {len(failed_models)} failed")

    if successful_models:
        logger.info(f"[OK] Successful models: {successful_models}")
        print(f"\n[OK] Successfully processed {len(successful_models)} models:")
        for model in successful_models:
            print(f"  ✓ {model}")

    if failed_models:
        logger.warning(f"[WARNING] Failed models: {[m[0] for m in failed_models]}")
        print(f"\n[WARNING] {len(failed_models)} models failed:")
        for model, error in failed_models:
            print(f"  ✗ {model}")
            # Mostrar solo la primera línea del error para no saturar la consola
            error_summary = error.split('\n')[0][:100]
            print(f"    Error: {error_summary}...")

    return successful_models, failed_models


def main():
    """Main execution function."""
    try:
        logger.info(f"[INFO] ===== Starting multi-model processing =====")
        logger.info(f"[INFO] Models to process: {[m['name'] for m in MODELS]}")
        logger.info(f"[INFO] Parallel model workers: {MAX_MODEL_WORKERS}")

        # Crear directorio de salida si no existe
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Load input file (una sola vez)
        df = load_input_file()

        # Process all models in parallel
        successful_models, failed_models = process_models_parallel(df)

        logger.info(f"[OK] ===== Processing finished! =====")
        print(f"\n{'='*60}")
        print(f"[OK] Processing Summary:")
        print(f"  Total models: {len(MODELS)}")
        print(f"  Successful: {len(successful_models)}")
        print(f"  Failed: {len(failed_models)}")
        print(f"{'='*60}")

    except Exception as e:
        logger.error(f"[FATAL] Fatal error in main execution: {str(e)}")
        raise

# ============================================================================
# MAIN EXECUTION
# ============================================================================




if __name__ == "__main__":
    main()
