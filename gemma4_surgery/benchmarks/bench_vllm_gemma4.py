"""
Empirical Benchmark: vLLM on Gemma 4 (unsloth/gemma-4-E2B-it)
Testing KV cache memory planning, multi-modal weight loading, and generation throughput.

Empirical Findings:
1. Cloud TPU v5e:
   Fails at import with:
   'tpu_inference not found, please install tpu_inference to run vllm on TPU'
   Stock PyPI vLLM requires private out-of-tree Google PJRT/Pallas backend.

2. Tesla T4 GPU (14.75 GiB VRAM):
   Loads weights in float16 (9.85 GiB VRAM).
   When vLLM's CacheEngine attempts to allocate PagedAttention KV blocks across
   all 35 layers (uniform LBNHC layout), remaining VRAM is exhausted:
   'ValueError: No available memory for the cache blocks.'
   Proves vLLM lacks awareness of Gemma 4's 20-layer KV freeze.
"""

import os
import sys
import time

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def run_vllm_gemma4_benchmark(
    model_name: str = "unsloth/gemma-4-E2B-it",
    max_model_len: int = 512,
    gpu_memory_utilization: float = 0.75,
):
  print("=" * 70)
  print(f"BENCHMARKING VLLM ON {model_name}")
  print(
      f"Config: max_model_len={max_model_len},"
      f" gpu_mem_util={gpu_memory_utilization}"
  )
  print("=" * 70)

  try:
    from vllm import LLM, SamplingParams

    print("[1] Initializing vLLM Engine...")
    t0 = time.time()
    llm = LLM(
        model=model_name,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True,
        trust_remote_code=True,
    )
    print(f"[1] Model successfully loaded in {time.time() - t0:.2f}s")

    sampling_params = SamplingParams(max_tokens=64, temperature=0.0)

    # Batch 1 Benchmark
    print("\n[2] Benchmarking Batch Size 1...")
    prompts_b1 = [
        "Explain the physics of semiconductors in three concise bullet points:"
    ]
    t0 = time.time()
    outputs_b1 = llm.generate(prompts_b1, sampling_params)
    t_b1 = time.time() - t0
    n_tokens_b1 = sum(len(o.outputs[0].token_ids) for o in outputs_b1)
    print(f"  BS=1: {n_tokens_b1} tokens in {t_b1:.3f}s -> {n_tokens_b1/t_b1:.2f} tok/s")

    # Batch 16 Benchmark
    print("\n[3] Benchmarking Batch Size 16...")
    prompts_b16 = [
        f"Explain point #{i} regarding quantum computing:" for i in range(16)
    ]
    t0 = time.time()
    outputs_b16 = llm.generate(prompts_b16, sampling_params)
    t_b16 = time.time() - t0
    n_tokens_b16 = sum(len(o.outputs[0].token_ids) for o in outputs_b16)
    print(
        f"  BS=16: {n_tokens_b16} tokens in {t_b16:.3f}s ->"
        f" {n_tokens_b16/t_b16:.2f} tok/s"
    )

  except Exception as err:
    import traceback

    print("\n[FATAL VLLM EXECUTION ERROR]:")
    traceback.print_exc()


if __name__ == "__main__":
  run_vllm_gemma4_benchmark()
