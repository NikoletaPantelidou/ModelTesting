import torch

print("="*50)
print("GPU Information Check")
print("="*50)

print(f"\nCUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Device capability: {torch.cuda.get_device_capability(0)}")
else:
    print("\n⚠ WARNING: No GPU detected!")
    print("Possible reasons:")
    print("  1. No NVIDIA GPU installed")
    print("  2. CUDA drivers not installed")
    print("  3. PyTorch installed without CUDA support (CPU-only version)")
    print("\nTo install PyTorch with CUDA support, visit:")
    print("  https://pytorch.org/get-started/locally/")

print("\n" + "="*50)

