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
Exp 2: Verify greedy equivalence and benchmark StaticGemma4Engine throughput on Colab TPU v5e-1.
"""

import json
import os
import sys
import tarfile
import time
import torch

if os.path.exists("/content/gemma4_pkg.tar.gz"):
    with tarfile.open("/content/gemma4_pkg.tar.gz", "r:gz") as tf:
        tf.extractall("/content", filter="data")

for _mod in list(sys.modules):
    if _mod == "benchmarks" or _mod.startswith("benchmarks.") or _mod == "surgeries" or _mod.startswith("surgeries."):
        del sys.modules[_mod]

sys.path.insert(0, "/content/gemma4_surgery")
from benchmarks.fast_infer import StaticGemma4Engine

os.makedirs("/content/gemma4_surgery/results", exist_ok=True)

# Reuse live global MODEL/TOKENIZER/DEVICE if already in Colab kernel
if "MODEL" not in globals():
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

    DEVICE = torch.accelerator.current_accelerator() if hasattr(torch, "accelerator") else torch.device("cpu")
    REPO_ID = "unsloth/gemma-4-E2B-it"
    print(f"[Reloading {REPO_ID} onto {DEVICE}...]", flush=True)
    TOKENIZER = AutoTokenizer.from_pretrained(REPO_ID)
    MODEL = Gemma4ForConditionalGeneration.from_pretrained(REPO_ID, dtype=torch.bfloat16)
    MODEL.model.vision_tower = None
    MODEL.model.audio_tower = None
    MODEL.model.embed_vision = None
    MODEL.model.embed_audio = None
    MODEL = MODEL.to(DEVICE)
    MODEL.eval()

ENGINE = StaticGemma4Engine(MODEL, TOKENIZER, DEVICE)
try:
    import torch_tpu  # type: ignore

    print("torch_tpu attrs:", [a for a in dir(torch_tpu) if not a.startswith("__")], flush=True)
except Exception as e:
    print("torch_tpu inspect:", e, flush=True)
print("Stop token IDs:", ENGINE.stop_ids_list, flush=True)

smoke_q = (
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes "
    "muffins for her friends every day with four. She sells the remainder at the farmers' market "
    "daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
)
fmt_one = ENGINE.format_chat_prompts([smoke_q], enable_thinking=False)

# 1. Reference HF generate (32 tokens to verify exact greedy match fast)
inp = TOKENIZER(fmt_one[0], return_tensors="pt", add_special_tokens=False).to(DEVICE)
with torch.no_grad():
    ref_ids = MODEL.generate(**inp, max_new_tokens=32, do_sample=False)
ref_gen_ids = ref_ids[0, inp["input_ids"].shape[1] :].cpu().tolist()
ref_text = TOKENIZER.decode(ref_gen_ids, skip_special_tokens=False)

# 2. StaticGemma4Engine B=1 (Run 1 = XLA compile + run; Run 2 = warm cached XLA graph!)
r1_cold = ENGINE.generate_batch_static(fmt_one, max_new_tokens=96, prompt_pad_bucket=128, early_stop_check_every=0)
r1_warm = ENGINE.generate_batch_static(fmt_one, max_new_tokens=96, prompt_pad_bucket=128, early_stop_check_every=0)

eng_first32_text = r1_warm["texts"][0]
print("\n[Reference first 32 toks]:", repr(ref_text[:160]), flush=True)
print("[StaticEng output 96 toks]:", repr(eng_first32_text[:220]), flush=True)
prefix_match = eng_first32_text.startswith(ref_text[:100])
print(f"Greedy prefix match (first 100 chars): {prefix_match}", flush=True)
print(
    f"B=1 Cold (compile+run): {r1_cold['elapsed_sec']}s ({r1_cold['raw_step_tok_per_sec']} tok/s) | "
    f"B=1 Warm: {r1_warm['elapsed_sec']}s ({r1_warm['raw_step_tok_per_sec']} raw tok/s, {r1_warm['effective_tok_per_sec']} eff tok/s)",
    flush=True,
)

# 3. Batched B=16 throughput (Cold compile vs Warm cached XLA execution)
batch16 = fmt_one * 16
r16_cold = ENGINE.generate_batch_static(batch16, max_new_tokens=128, prompt_pad_bucket=128, early_stop_check_every=0)
r16_warm = ENGINE.generate_batch_static(batch16, max_new_tokens=128, prompt_pad_bucket=128, early_stop_check_every=0)
print(
    f"B=16 Cold (compile+run): {r16_cold['elapsed_sec']}s ({r16_cold['raw_step_tok_per_sec']} raw tok/s) | "
    f"B=16 Warm: {r16_warm['elapsed_sec']}s ({r16_warm['raw_step_tok_per_sec']} raw tok/s)",
    flush=True,
)

report = {
    "baseline_hf_dynamic_cache_b1_tok_s": 3.17,
    "prefix_exact_match": prefix_match,
    "ref_preview": ref_text[:200],
    "static_engine_preview": eng_first32_text[:300],
    "b1_cold": {k: v for k, v in r1_cold.items() if k != "texts"},
    "b1_warm": {k: v for k, v in r1_warm.items() if k != "texts"},
    "b16_cold": {k: v for k, v in r16_cold.items() if k != "texts"},
    "b16_warm": {k: v for k, v in r16_warm.items() if k != "texts"},
}
with open("/content/gemma4_surgery/results/exp2_fast_engine_report.json", "w") as fh:
    json.dump(report, fh, indent=2)
print("Saved /content/gemma4_surgery/results/exp2_fast_engine_report.json", flush=True)
