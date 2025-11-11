import torch

print("-" * 50)
is_available = torch.cuda.is_available()
print(f"Is CUDA (GPU) available?  -> {is_available}")

if is_available:
    device_count = torch.cuda.device_count()
    print(f"Number of GPUs found:     -> {device_count}")
    
    current_device = torch.cuda.current_device()
    print(f"Current GPU index:        -> {current_device}")
    
    device_name = torch.cuda.get_device_name(current_device)
    print(f"Current GPU name:         -> {device_name}")
else:
    print("\n--- ERROR ---")
    print("PyTorch cannot detect your NVIDIA GPU.")
    print("This is why your script is taking 4 minutes (it's using the CPU).")
    print("To fix this, you MUST reinstall PyTorch with the correct CUDA version.")
    print("1. Go to: https://pytorch.org/get-started/locally/")
    print("2. Select 'Stable', 'Windows', 'Conda', 'Python', and your CUDA version.")
    print("3. Run the generated command (e.g., 'conda install pytorch ... -c pytorch -c nvidia')")

print("-" * 50)