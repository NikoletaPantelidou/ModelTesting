"""
Script to clean English translations from existing answer files.
This script removes English translations that were added by models when
answering in non-English languages.
"""

import os
import pandas as pd
import re
from datetime import datetime
import shutil

def clean_multilingual_answer(answer, target_language):
    """Clean answers that may contain multiple languages, keeping only the target language."""
    if not answer or not isinstance(answer, str) or not target_language:
        return answer

    lang_lower = str(target_language).lower()

    if lang_lower != "english":
        # Remove English translations in parentheses at the end
        # Examples: "(Yes, it's true.)", "(No hi ha una planxa a l'armari.)"
        answer = re.sub(r'\s*\([^)]*[A-Z][^)]*\)\s*$', '', answer)

        # Remove English translations in parentheses anywhere in the text
        # Look for common English words that indicate a translation
        answer = re.sub(r'\s*\([^)]*(?:Yes|No|If|It|The|There|According|Otherwise|True|False)[^)]*\)', '', answer, flags=re.IGNORECASE)

        # Patterns to detect English translations added by the model
        english_patterns = [
            r'\n\n\s*(?:Translation|In English|English version|English answer)[\s:]+.*',
            r'\n\s*English[\s:]+.*',
            r'\n\n\s*[A-Z][^.]*\.\s*(?:[A-Z][^.]*\.)+',  # Sentences that look like English after non-English
            r'\n\n\s*La traducción directa.*',  # Spanish translation phrases
            r'\n\n\s*La traducción.*',  # Spanish translation phrases
            r'\n\s*La traducción directa.*',  # Spanish translation phrases inline
        ]

        # Remove common English translation patterns
        for pattern in english_patterns:
            answer = re.sub(pattern, '', answer, flags=re.IGNORECASE | re.DOTALL)

        # If the answer has two distinct paragraphs and the second looks like English, remove it
        paragraphs = answer.split('\n\n')
        if len(paragraphs) > 1:
            # Keep only the first paragraph if it seems complete
            first = paragraphs[0].strip()
            if first and (first.endswith('.') or first.endswith('?') or first.endswith('!')):
                # Check if second paragraph looks like it's in English
                second = paragraphs[1].strip()
                # Check if it doesn't contain common non-English characters
                if second and not any(char in second.lower() for char in ['à', 'è', 'é', 'ò', 'í', 'ñ', 'ç', 'ü', 'ú', 'á', 'ï']):
                    # Likely English, remove it
                    answer = first

    return answer.strip()

def clean_answer_file(file_path):
    """Clean translations from a single answer file."""
    print(f"\n[INFO] Processing file: {file_path}")

    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return False

    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Check required columns
        if 'answer' not in df.columns or 'language' not in df.columns:
            print(f"[ERROR] File must contain 'answer' and 'language' columns")
            return False

        # Create backup
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = file_path.replace('.csv', f'_backup_{timestamp}.csv')
        shutil.copy2(file_path, backup_path)
        print(f"[INFO] Backup created: {backup_path}")

        # Clean answers
        cleaned_count = 0
        for idx, row in df.iterrows():
            if pd.notna(row['answer']) and pd.notna(row['language']):
                original_answer = str(row['answer'])
                cleaned_answer = clean_multilingual_answer(original_answer, row['language'])

                if original_answer != cleaned_answer:
                    df.at[idx, 'answer'] = cleaned_answer
                    cleaned_count += 1

                    # Show some examples of changes
                    if cleaned_count <= 5:
                        print(f"\n[EXAMPLE] Row {idx + 1} ({row['language']}):")
                        print(f"  Original: {original_answer[:100]}...")
                        print(f"  Cleaned:  {cleaned_answer[:100]}...")

        # Save the cleaned file
        df.to_csv(file_path, index=False)
        print(f"\n[OK] File cleaned successfully!")
        print(f"[STATS] Total rows: {len(df)}, Cleaned answers: {cleaned_count}")

        return True

    except Exception as e:
        print(f"[ERROR] Error processing file: {str(e)}")
        return False

def clean_all_answer_files():
    """Clean all answer files in the answers directory."""
    answers_dir = "answers"

    if not os.path.exists(answers_dir):
        print(f"[ERROR] Answers directory not found: {answers_dir}")
        return

    # Find all CSV files in the answers directory
    csv_files = [f for f in os.listdir(answers_dir) if f.endswith('_answers.csv')]

    if not csv_files:
        print("[INFO] No answer files found to clean")
        return

    print(f"[INFO] Found {len(csv_files)} answer file(s) to clean:")
    for f in csv_files:
        print(f"  - {f}")

    # Process each file
    successful = 0
    failed = 0

    for csv_file in csv_files:
        file_path = os.path.join(answers_dir, csv_file)
        if clean_answer_file(file_path):
            successful += 1
        else:
            failed += 1

    print(f"\n[SUMMARY]")
    print(f"Successfully cleaned: {successful} file(s)")
    print(f"Failed: {failed} file(s)")

if __name__ == "__main__":
    print("=" * 70)
    print("English Translation Cleaner for Answer Files")
    print("=" * 70)
    clean_all_answer_files()
    print("\n[DONE] Cleaning process completed!")

