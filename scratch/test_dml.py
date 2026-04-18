import torch
import torch_directml
import time

print(f"Torch version: {torch.__version__}")
print(f"DirectML available: {torch_directml.is_available()}")

if torch_directml.is_available():
    device = torch_directml.device()
    print(f"Using device: {torch_directml.device_name(0)}")
    
    # Test computation
    a = torch.tensor([1.0, 2.0]).to(device)
    b = torch.tensor([3.0, 4.0]).to(device)
    start = time.time()
    for _ in range(1000):
        c = a * b
    print(f"Computation test on GPU successful in {time.time()-start:.4f}s")
else:
    print("DirectML NOT available.")
