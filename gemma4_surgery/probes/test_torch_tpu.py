import time
import torch
import transformers

print("Accelerator available:", torch.accelerator.is_available())
acc = torch.accelerator.current_accelerator()
print("Current accelerator:", acc)

t0 = time.time()
x = torch.randn(2048, 2048, device=acc, dtype=torch.bfloat16)
y = x @ x.T
torch.accelerator.synchronize()
print(f"TPU matmul 2048x2048 bf16 ok in {time.time()-t0:.3f}s, mean={y.float().mean().item():.4f}")

print("Gemma4 classes in transformers:", [k for k in dir(transformers) if "gemma4" in k.lower()])
