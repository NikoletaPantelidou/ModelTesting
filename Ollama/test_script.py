"""
Test script to verify CSV reading and answer saving
"""
import pandas as pd
import os

# Test reading the input file
INPUT_FILE = "prompts/example.csv"
CSV_SEPARATOR = ";"

print("Testing CSV reading...")
df = pd.read_csv(INPUT_FILE, sep=CSV_SEPARATOR)
print(f"Columns before cleaning: {df.columns.tolist()}")

df.columns = df.columns.str.strip().str.replace('\ufeff', '')
print(f"Columns after cleaning: {df.columns.tolist()}")

print(f"\nNumber of rows: {len(df)}")
print(f"\nFirst row prompt: {df.iloc[0]['prompt'][:100]}...")

# Test creating an answer column
print("\nTesting answer column creation...")
answers = ["Test answer " + str(i) for i in range(len(df))]
df_with_answers = df.assign(answer=answers)

print(f"Columns with answer: {df_with_answers.columns.tolist()}")
print(f"\nFirst few rows:")
print(df_with_answers.head(3))

# Test saving
output_file = "answers/test_output.csv"
os.makedirs("answers", exist_ok=True)
df_with_answers.to_csv(output_file, index=False)
print(f"\nSaved test file to: {output_file}")

# Test reading back
df_read = pd.read_csv(output_file)
print(f"\nRead back columns: {df_read.columns.tolist()}")
print(f"First answer: {df_read.iloc[0]['answer']}")
print("\n✓ Test completed successfully!")

