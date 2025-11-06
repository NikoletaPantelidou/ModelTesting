"""
Quick test script for phi-4 model to debug empty answers issue.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# Login
login(token="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")

MODEL_NAME = "microsoft/phi-4"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[INFO] Loading model {MODEL_NAME} on {DEVICE}...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=False,
    token="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"
)

tokenizer.padding_side = 'left'
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=False,
    token="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",
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto" if DEVICE == "cuda" else None,
)

if DEVICE != "cuda":
    model = model.to(DEVICE)

model.config.pad_token_id = tokenizer.pad_token_id

print("[OK] Model loaded successfully")

# Test prompt
context = "Here's the context: John is hungry and there is pizza in the oven. There is also pasta in the fridge. Mary says: \"If John is hungry, there is pizza in the oven\"."
question = "Based on what Mary says, is it true that there is pizza in the oven?"

full_prompt = f"Context: {context}\n\nQuestion: {question}\n\nProvide a direct answer in English only:"

print(f"\n[PROMPT] {full_prompt}\n")

inputs = tokenizer(full_prompt, return_tensors="pt").to(DEVICE)

print(f"[INFO] Input tokens: {inputs['input_ids'].shape[1]}")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        early_stopping=False,
        num_beams=1,
        do_sample=False
    )

print(f"[INFO] Output tokens: {outputs.shape[1]}")

full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"\n[FULL OUTPUT]\n{full_output}\n")
print(f"[OUTPUT LENGTH] {len(full_output)} characters")

# Try to extract answer
answer = full_output
if full_prompt in answer:
    answer = answer.replace(full_prompt, "").strip()
    print(f"\n[EXTRACTED BY REPLACEMENT]\n{answer}\n")
else:
    print(f"\n[WARNING] Full prompt not found in output, trying alternative methods...")
    # Try to find answer marker
    for marker in ["\n\nAnswer:", "\nAnswer:", "Answer:"]:
        if marker in full_output:
            parts = full_output.split(marker, 1)
            if len(parts) > 1:
                answer = parts[1].strip()
                print(f"\n[EXTRACTED BY MARKER '{marker}']\n{answer}\n")
                break
    else:
        # Try by length
        if len(full_output) > len(full_prompt):
            answer = full_output[len(full_prompt):].strip()
            print(f"\n[EXTRACTED BY LENGTH]\n{answer}\n")
        else:
            print(f"\n[ERROR] Could not extract answer!")

print(f"\n[FINAL ANSWER LENGTH] {len(answer)} characters")

