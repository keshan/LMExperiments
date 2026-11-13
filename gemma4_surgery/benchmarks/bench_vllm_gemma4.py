# Copyright 2026 Keshan Sodimana
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Standalone vLLM Benchmark on Gemma 4 (`unsloth/gemma-4-E2B-it`) for CUDA GPUs.

Verified on Colab Tesla T4 (`sm_75`, 14.56 GiB VRAM) with `vllm==0.33.0`:
- Uses native `float16` (`dtype="half"`) on `sm < 8.0` (Turing) and `bfloat16` on `sm >= 8.0`.
- Sets `VLLM_ENABLE_V1_MULTIPROCESSING=0` so vLLM runs cleanly inside an active
  notebook / Python process after releasing PyTorch & cuBLAS allocations.
- On `sm < 8.0` GPUs (Tesla T4's 65,536-byte shared-memory limit per SM), clamps
  vLLM's `TRITON_ATTN` `_get_tile_size(head_size >= 512)` to `16` so Gemma 4's
  512-dim global attention layers run without Triton `OutOfResources` crashes.
"""

import ctypes
import gc
import json
import os
import sys
import time
from typing import Dict, Any, Optional

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROCESSING", "0")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")


def clear_gpu_memory() -> Dict[str, Any]:
    """Release PyTorch allocations, cuBLAS workspaces, and OS heap pages."""
    gc.collect()
    import torch

    if not torch.cuda.is_available():
        return {"device": "cpu"}
    torch.cuda.synchronize()
    try:
        torch._C._cuda_clearCublasWorkspaces()
    except Exception:
        pass
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    free_b, total_b = torch.cuda.mem_get_info()
    return {
        "device": torch.cuda.get_device_name(0),
        "free_gib": round(free_b / (1024**3), 3),
        "total_gib": round(total_b / (1024**3), 3),
        "allocated_gib": round(torch.cuda.memory_allocated() / (1024**3), 3),
        "reserved_gib": round(torch.cuda.memory_reserved() / (1024**3), 3),
    }


def apply_turing_triton_tile_fix(torch_mod) -> bool:
    """Clamp TRITON_ATTN head_dim>=512 tile size to 16 on sm < 8.0 GPUs (64 KB SMEM)."""
    if not torch_mod.cuda.is_available() or torch_mod.cuda.get_device_capability(0)[0] >= 8:
        return False
    import vllm.v1.attention.ops.triton_unified_attention as tua

    if getattr(tua._get_tile_size, "_turing_patched", False):
        return True
    orig_get_tile_size = tua._get_tile_size

    def _turing_safe_tile_size(head_size, sliding_window, element_size, is_prefill):
        if head_size >= 512:
            return 16
        return orig_get_tile_size(head_size, sliding_window, element_size, is_prefill)

    _turing_safe_tile_size._turing_patched = True
    tua._get_tile_size = _turing_safe_tile_size
    return True


def run_vllm_gemma4_benchmark(
    model_name: str = "unsloth/gemma-4-E2B-it",
    max_model_len: int = 512,
    enforce_eager: bool = True,
    output_json: str = "/content/vllm_t4_gpu_report.json",
) -> Optional[Dict[str, Any]]:
    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )

    if not torch.cuda.is_available():
        print("No CUDA GPU detected; skipping vLLM GPU benchmark.")
        return None

    mem_before = clear_gpu_memory()
    major, minor = torch.cuda.get_device_capability(0)
    vllm_dtype = "half" if major < 8 else "bfloat16"
    turing_patched = apply_turing_triton_tile_fix(torch)

    print("=" * 72)
    print(f"BENCHMARKING vLLM {vllm.__version__} ON {model_name}")
    print(
        f"Device: {mem_before['device']} (sm_{major}{minor}) | dtype={vllm_dtype} "
        f"| Free VRAM={mem_before['free_gib']}/{mem_before['total_gib']} GiB "
        f"| Turing tile=16 fix={turing_patched}"
    )
    print("=" * 72)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    smoke_prompt = (
        "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes "
        "muffins for her friends every day with four. She sells the remainder at the farmers' market "
        "daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    )
    fmt_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": smoke_prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    llm = None
    try:
        print("\n[1] Initializing vLLM Engine...")
        t0 = time.perf_counter()
        llm = LLM(
            model=model_name,
            dtype=vllm_dtype,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
            trust_remote_code=False,
        )
        load_sec = time.perf_counter() - t0
        print(f"[1] vLLM loaded in {load_sec:.2f}s")

        report: Dict[str, Any] = {
            "vllm_version": vllm.__version__,
            "model": model_name,
            "device": mem_before["device"],
            "compute_capability": f"{major}.{minor}",
            "dtype": vllm_dtype,
            "max_model_len": max_model_len,
            "enforce_eager": enforce_eager,
            "turing_tile_fix_applied": turing_patched,
            "vram_before_init": mem_before,
            "load_sec": round(load_sec, 2),
            "batches": {},
        }

        g48 = llm.generate([fmt_prompt], SamplingParams(max_tokens=48, min_tokens=48, temperature=0.0))
        report["greedy_48"] = g48[0].outputs[0].text
        print("[1] vLLM greedy 48-token output:", repr(report["greedy_48"][:120]))

        # Warmup call
        _ = llm.generate([fmt_prompt], SamplingParams(max_tokens=16, min_tokens=16, temperature=0.0))

        for b_size, max_toks in [(1, 128), (8, 128), (16, 128), (32, 128)]:
            print(f"\n[2] Benchmarking Batch Size B={b_size} (max_tokens={max_toks})...")
            batch_prompts = [fmt_prompt] * b_size
            sp = SamplingParams(
                max_tokens=max_toks,
                min_tokens=max_toks,
                temperature=0.0,
                ignore_eos=True,
            )
            torch.cuda.synchronize()
            t_start = time.perf_counter()
            outputs = llm.generate(batch_prompts, sp)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t_start
            total_toks = sum(len(o.outputs[0].token_ids) for o in outputs)
            tok_s = total_toks / max(elapsed, 1e-6)
            peak_alloc = round(torch.cuda.max_memory_allocated() / (1024**3), 3)
            print(
                f"  B={b_size:2d}: {total_toks} tokens in {elapsed:.3f}s -> {tok_s:.2f} tok/s "
                f"(peak alloc={peak_alloc} GiB)"
            )
            report["batches"][f"B{b_size}"] = {
                "batch_size": b_size,
                "max_tokens": max_toks,
                "total_tokens": total_toks,
                "elapsed_sec": round(elapsed, 4),
                "tok_per_sec": round(tok_s, 2),
                "peak_allocated_gib": peak_alloc,
                "sample_text": outputs[0].outputs[0].text[:120] if b_size == 1 else "",
            }

        del llm
        llm = None
        destroy_model_parallel()
        destroy_distributed_environment()
        report["vram_after_cleanup"] = clear_gpu_memory()

        with open(output_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved vLLM GPU report to {output_json}")
        return report
    finally:
        if llm is not None:
            del llm
        try:
            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception:
            pass
        clear_gpu_memory()


if __name__ == "__main__":
    run_vllm_gemma4_benchmark()
