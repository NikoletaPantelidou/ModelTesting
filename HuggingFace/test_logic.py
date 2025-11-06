"""
Script para verificar la sintaxis y lógica del código sin ejecutar el modelo.
"""

# Simular las funciones clave para verificar la lógica

def test_phi_detection():
    """Test que la detección de modelo phi funciona"""
    model_names = [
        "microsoft/phi-4",
        "microsoft/phi-3",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "Phi-2-test"
    ]

    for name in model_names:
        is_phi = 'phi' in name.lower()
        print(f"Model: {name:50} -> is_phi: {is_phi}")

def test_chat_template_format():
    """Test del formato de chat"""
    system_message = "You are a helpful assistant."
    user_content = "Context: John is hungry.\nQuestion: Is John hungry?"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content}
    ]

    # Formato manual
    manual_format = f"<|system|>\n{system_message}<|end|>\n<|user|>\n{user_content}<|end|>\n<|assistant|>\n"

    print("\n=== Chat Format ===")
    print(manual_format)
    print(f"\nLength: {len(manual_format)} chars")

def test_answer_extraction():
    """Test de extracción de respuestas"""

    test_cases = [
        {
            "name": "With assistant tag",
            "output": "<|system|>\nYou are helpful<|end|>\n<|user|>\nQuestion?<|end|>\n<|assistant|>\nYes, that is correct.<|end|>",
            "expected": "Yes, that is correct."
        },
        {
            "name": "With Answer: marker",
            "output": "Context: Test\nQuestion: Test?\n\nAnswer: This is the answer.",
            "expected": "This is the answer."
        },
        {
            "name": "Direct text",
            "output": "Just the answer text here.",
            "expected": "Just the answer text here."
        }
    ]

    print("\n=== Answer Extraction Tests ===")
    for test in test_cases:
        answer = test["output"]

        # Simular extracción para phi
        assistant_markers = ["<|assistant|>", "assistant\n", "Assistant:", "assistant:"]
        for marker in assistant_markers:
            if marker in answer:
                parts = answer.split(marker, 1)
                if len(parts) > 1:
                    answer = parts[1].strip()
                    answer = answer.replace("<|end|>", "").strip()
                    break

        # Simular extracción por markers comunes
        for marker in ["\n\nAnswer:", "\nAnswer:", "Answer:"]:
            if marker in answer:
                parts = answer.split(marker, 1)
                if len(parts) > 1:
                    answer = parts[1].strip()
                    break

        success = answer.strip() == test["expected"].strip()
        status = "✓" if success else "✗"
        print(f"{status} {test['name']:30} -> '{answer[:50]}...'")

if __name__ == "__main__":
    print("=" * 70)
    print("Testing Phi-4 Code Logic")
    print("=" * 70)

    test_phi_detection()
    test_chat_template_format()
    test_answer_extraction()

    print("\n" + "=" * 70)
    print("All logic tests completed!")
    print("=" * 70)

