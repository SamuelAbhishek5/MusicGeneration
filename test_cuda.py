import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
    print("CUDA capability:", torch.cuda.get_device_capability(0))
    print("\n✓ CUDA is ENABLED - server will run MUCH FASTER on GPU!")
else:
    print("\n✗ CUDA not available - server will run slow on CPU")
