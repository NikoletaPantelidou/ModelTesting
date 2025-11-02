"""
Script to check the status of all answer files and generate a consolidated report.
"""

import os
import pandas as pd
from datetime import datetime

INPUT_FILE = "prompts/example.csv"
OUTPUT_DIR = "answers"
CSV_SEPARATOR = ";"

def load_input_file():
    """Load the input prompts file"""
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Could not find {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, sep=CSV_SEPARATOR)
    df.columns = df.columns.str.strip().str.replace('\ufeff', '')
    return df

def check_model_answers(answer_file, df):
    """Check the status of answers for a specific model"""
    if not os.path.exists(answer_file):
        return None

    try:
        answer_df = pd.read_csv(answer_file)

        total = len(answer_df)
        completed = answer_df['answer'].notna().sum()
        errors = sum(1 for a in answer_df['answer'] if pd.notna(a) and str(a).startswith("ERROR:"))
        pending = total - completed
        success = completed - errors

        # Get pending items
        pending_items = []
        error_items = []
        completed_items = []

        for idx, row in answer_df.iterrows():
            test_item = row.get('test_item', f'Row {idx+1}')
            answer = row.get('answer', '')

            if pd.isna(answer):
                pending_items.append(test_item)
            elif str(answer).startswith("ERROR:"):
                error_items.append((test_item, str(answer)[:80]))
            else:
                completed_items.append(test_item)

        return {
            'total': total,
            'completed': completed,
            'success': success,
            'errors': errors,
            'pending': pending,
            'pending_items': pending_items,
            'error_items': error_items,
            'completed_items': completed_items
        }
    except Exception as e:
        print(f"Error reading {answer_file}: {str(e)}")
        return None

def generate_consolidated_report(df):
    """Generate a consolidated report for all models"""
    report_file = os.path.join(OUTPUT_DIR, "consolidated_status_report.txt")

    # Find all answer files
    if not os.path.exists(OUTPUT_DIR):
        print(f"Output directory {OUTPUT_DIR} does not exist")
        return

    answer_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('_answers.csv')]

    if not answer_files:
        print("No answer files found")
        return

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("CONSOLIDATED STATUS REPORT - ALL MODELS\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total models: {len(answer_files)}\n")
        f.write("=" * 100 + "\n\n")

        all_models_status = []

        for answer_file in sorted(answer_files):
            model_name = answer_file.replace('_answers.csv', '')
            full_path = os.path.join(OUTPUT_DIR, answer_file)

            status = check_model_answers(full_path, df)

            if status is None:
                continue

            all_models_status.append((model_name, status))

        # Summary table
        f.write("SUMMARY TABLE:\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Model':<40} {'Total':>8} {'Success':>10} {'Errors':>8} {'Pending':>10} {'Complete %':>12}\n")
        f.write("-" * 100 + "\n")

        for model_name, status in all_models_status:
            complete_pct = (status['success'] / status['total'] * 100) if status['total'] > 0 else 0
            f.write(f"{model_name:<40} {status['total']:>8} {status['success']:>10} {status['errors']:>8} {status['pending']:>10} {complete_pct:>11.1f}%\n")

        f.write("-" * 100 + "\n\n")

        # Detailed status for each model
        for model_name, status in all_models_status:
            f.write("=" * 100 + "\n")
            f.write(f"MODEL: {model_name}\n")
            f.write("=" * 100 + "\n\n")

            f.write(f"Total prompts:      {status['total']}\n")
            f.write(f"Completed:          {status['completed']} ({100*status['completed']/status['total']:.1f}%)\n")
            f.write(f"Successful:         {status['success']} ({100*status['success']/status['total']:.1f}%)\n")
            f.write(f"Errors:             {status['errors']} ({100*status['errors']/status['total']:.1f}%)\n")
            f.write(f"Pending:            {status['pending']} ({100*status['pending']/status['total']:.1f}%)\n")
            f.write("\n")

            if status['pending'] > 0:
                f.write(f"PENDING ITEMS ({len(status['pending_items'])}):\n")
                for item in status['pending_items']:
                    f.write(f"  - {item}\n")
                f.write("\n")

            if status['errors'] > 0:
                f.write(f"ERROR ITEMS ({len(status['error_items'])}):\n")
                for item, error in status['error_items']:
                    f.write(f"  - {item}: {error}\n")
                f.write("\n")

            f.write("\n")

    print(f"[OK] Consolidated report saved to: {report_file}")

    # Also print summary to console
    print("\n" + "=" * 100)
    print("SUMMARY - ALL MODELS")
    print("=" * 100)
    print(f"{'Model':<40} {'Total':>8} {'Success':>10} {'Errors':>8} {'Pending':>10} {'Complete %':>12}")
    print("-" * 100)

    for model_name, status in all_models_status:
        complete_pct = (status['success'] / status['total'] * 100) if status['total'] > 0 else 0
        print(f"{model_name:<40} {status['total']:>8} {status['success']:>10} {status['errors']:>8} {status['pending']:>10} {complete_pct:>11.1f}%")

    print("-" * 100)

def main():
    try:
        print("[INFO] Checking status of all answer files...")
        df = load_input_file()
        print(f"[OK] Loaded input file with {len(df)} prompts")

        # Check if consolidated file exists
        consolidated_file = os.path.join(OUTPUT_DIR, "consolidated_answers.csv")
        if os.path.exists(consolidated_file):
            print(f"[INFO] Found consolidated file: {consolidated_file}")
            try:
                consolidated_df = pd.read_csv(consolidated_file)
                answer_columns = [col for col in consolidated_df.columns if col.startswith('answer_')]
                print(f"[INFO] Consolidated file contains {len(answer_columns)} models")
                print()
            except Exception as e:
                print(f"[WARNING] Could not read consolidated file: {str(e)}")

        generate_consolidated_report(df)
        print("[OK] Status check completed")
    except Exception as e:
        print(f"[ERROR] {str(e)}")

if __name__ == "__main__":
    main()

