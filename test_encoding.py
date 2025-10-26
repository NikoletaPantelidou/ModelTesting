"""Test encoding detection for CSV file"""
import pandas as pd

CSV_FILE = "example.csv"
CSV_SEPARATOR = ";"

encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

print("Testing different encodings...\n")

for encoding in encodings_to_try:
    try:
        df = pd.read_csv(CSV_FILE, sep=CSV_SEPARATOR, encoding=encoding)
        df.columns = df.columns.str.strip().str.replace('\ufeff', '')

        print(f"[OK] Encoding '{encoding}' works!")
        print(f"  - Rows: {len(df)}")
        print(f"  - Columns: {df.columns.tolist()}")
        print(f"  - First prompt preview: {df['prompt'].iloc[0][:50]}...\n")
        break
    except Exception as e:
        print(f"[FAIL] Encoding '{encoding}': {str(e)[:50]}\n")

