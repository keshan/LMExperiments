import os
import platform
import shutil
import subprocess

print("=== HOST SYSTEM ===")
print("Platform:", platform.platform())
print("CPU count:", os.cpu_count())
try:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(("MemTotal", "MemAvailable")):
                print(line.strip())
except Exception as e:
    print("meminfo err:", e)

total, used, free = shutil.disk_usage("/content")
print(f"Disk /content: total={total/1e9:.1f}GB, free={free/1e9:.1f}GB")

print("\n=== INSTALLED PACKAGES CHECK ===")
for pkg in ["torch", "torch_xla", "jax", "flax", "transformers", "accelerate", "datasets", "vllm", "keras", "gemma"]:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "unknown")
        print(f"  {pkg}: {ver}")
    except Exception as e:
        print(f"  {pkg}: NOT INSTALLED ({type(e).__name__}: {e})")

print("\n=== TPU DEVICE CHECK ===")
try:
    import jax
    print("JAX backend:", jax.default_backend())
    print("JAX devices:", jax.devices())
    for d in jax.devices():
        try:
            print("  device memory_stats:", d.memory_stats())
        except Exception as e:
            print("  memory_stats err:", e)
except Exception as e:
    print("JAX TPU check failed:", e)

try:
    import torch_xla.core.xla_model as xm
    dev = xm.xla_device()
    print("PyTorch/XLA device:", dev)
    print("PyTorch/XLA mem info:", xm.get_memory_info(dev))
except Exception as e:
    print("torch_xla check:", e)
