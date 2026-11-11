"""
Exp 2b: Benchmark 4-step XLA graph unrolling (`sync_every=4`) + B=1 / B=32 AND KV-Shared Layer Bypass speedup!
"""

import json
import os
import sys
import tarfile
import torch

if os.path.exists("/content/gemma4_pkg.tar.gz"):
    with tarfile.open("/content/gemma4_pkg.tar.gz", "r:gz") as tf:
        tf.extractall("/content", filter="data")

for _mod in list(sys.modules):
    if _mod == "benchmarks" or _mod.startswith("benchmarks.") or _mod == "surgeries" or _mod.startswith("surgeries."):
        del sys.modules[_mod]

sys.path.insert(0, "/content/gemma4_surgery")
from benchmarks.fast_infer import StaticGemma4Engine
from surgeries.weight_surgeon import WeightSurgeon

ENGINE = StaticGemma4Engine(MODEL, TOKENIZER, DEVICE)
SURGEON = WeightSurgeon(ENGINE.text_model)

smoke_q = (
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes "
    "muffins for her friends every day with four. She sells the remainder at the farmers' market "
    "daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
)
fmt_one = ENGINE.format_chat_prompts([smoke_q], enable_thinking=False)

# 1. B=1 with sync_every=4 (cold + warm)
b1_s4_cold = ENGINE.generate_batch_static(fmt_one, max_new_tokens=97, prompt_pad_bucket=128, early_stop_check_every=0, sync_every=4)
b1_s4_warm = ENGINE.generate_batch_static(fmt_one, max_new_tokens=97, prompt_pad_bucket=128, early_stop_check_every=0, sync_every=4)
print(
    f"B=1 sync=4 Cold: {b1_s4_cold['elapsed_sec']}s | Warm: {b1_s4_warm['elapsed_sec']}s ({b1_s4_warm['raw_step_tok_per_sec']} tok/s)",
    flush=True,
)
print("Output check:", repr(b1_s4_warm["texts"][0][:140]), flush=True)

# 2. B=32 with sync_every=4 (cold + warm)
batch32 = fmt_one * 32
b32_s4_cold = ENGINE.generate_batch_static(batch32, max_new_tokens=129, prompt_pad_bucket=128, early_stop_check_every=0, sync_every=4)
b32_s4_warm = ENGINE.generate_batch_static(batch32, max_new_tokens=129, prompt_pad_bucket=128, early_stop_check_every=0, sync_every=4)
print(
    f"B=32 sync=4 Cold: {b32_s4_cold['elapsed_sec']}s | Warm: {b32_s4_warm['elapsed_sec']}s ({b32_s4_warm['raw_step_tok_per_sec']} tok/s)",
    flush=True,
)

# 3. Measure Scale-Compensated KV-Shared Layer Bypass ([30, 31, 32]) pure hardware latency gain!
SURGEON.bypass_kv_shared_layers([30, 31, 32], compensate_scalar=True)
b1_byp_cold = ENGINE.generate_batch_static(fmt_one, max_new_tokens=97, prompt_pad_bucket=128, early_stop_check_every=0, sync_every=4)
b1_byp_warm = ENGINE.generate_batch_static(fmt_one, max_new_tokens=97, prompt_pad_bucket=128, early_stop_check_every=0, sync_every=4)
b32_byp_cold = ENGINE.generate_batch_static(batch32, max_new_tokens=129, prompt_pad_bucket=128, early_stop_check_every=0, sync_every=4)
b32_byp_warm = ENGINE.generate_batch_static(batch32, max_new_tokens=129, prompt_pad_bucket=128, early_stop_check_every=0, sync_every=4)
print(
    f"[Bypass L30-32] B=1 Warm: {b1_byp_warm['elapsed_sec']}s ({b1_byp_warm['raw_step_tok_per_sec']} tok/s) | "
    f"B=32 Warm: {b32_byp_warm['elapsed_sec']}s ({b32_byp_warm['raw_step_tok_per_sec']} tok/s)",
    flush=True,
)
print("[Bypass L30-32 output]:", repr(b1_byp_warm["texts"][0][:200]), flush=True)
SURGEON.restore_all()

out = {
    "b1_sync4_warm": {k: v for k, v in b1_s4_warm.items() if k != "texts"},
    "b32_sync4_warm": {k: v for k, v in b32_s4_warm.items() if k != "texts"},
    "b1_bypass30_32_warm": {k: v for k, v in b1_byp_warm.items() if k != "texts"},
    "b32_bypass30_32_warm": {k: v for k, v in b32_byp_warm.items() if k != "texts"},
    "b1_sync4_text": b1_s4_warm["texts"][0],
    "b1_bypass30_32_text": b1_byp_warm["texts"][0],
}
with open("/content/gemma4_surgery/results/exp2b_sync4_and_bypass_speed.json", "w") as fh:
    json.dump(out, fh, indent=2)
