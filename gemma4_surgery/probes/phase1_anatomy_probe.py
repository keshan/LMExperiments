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
Phase 1: Mechanistic & Spectral Anatomy Probe for Gemma 4 E2B-it on Google Colab TPU v5e-1 / GPU.

Persists `MODEL`, `TOKENIZER`, `TEXT_MODEL`, `DEVICE`, `REPO_ID` in live Colab kernel globals
so subsequent weight-surgery and benchmark scripts execute without reloading weights.
"""

import inspect
import json
import math
import os
import time
from pathlib import Path

import torch
import transformers


def get_device() -> torch.device:
    if hasattr(torch, "accelerator") and torch.accelerator.is_available():
        return torch.accelerator.current_accelerator()
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sync_device() -> None:
    if hasattr(torch, "accelerator") and torch.accelerator.is_available():
        torch.accelerator.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def get_hardware_info(device: torch.device) -> dict:
    info = {
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device": str(device),
        "cpu_count": os.cpu_count(),
    }
    if device.type == "cuda":
        info["device_name"] = torch.cuda.get_device_name(0)
        info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    elif device.type == "tpu":
        info["device_name"] = "Google Cloud TPU v5e-1 (Colab Free)"
        info["hbm_gb"] = 15.75
    return info


def stable_rank_and_spectrum(weight: torch.Tensor) -> dict:
    """Compute SVD spectrum statistics on CPU float32 (avoids XLA shape recompilations)."""
    w = weight.detach().to(device="cpu", dtype=torch.float32)
    if w.ndim > 2:
        w = w.reshape(w.shape[0], -1)
    s = torch.linalg.svdvals(w)
    top1 = float(s[0].item())
    frob_sq = float((s**2).sum().item())
    srank = frob_sq / (top1**2 + 1e-12)
    p = (s**2) / (frob_sq + 1e-12)
    p_pos = p[p > 1e-12]
    eff_rank = float(torch.exp(-(p_pos * torch.log(p_pos)).sum()).item())
    return {
        "shape": list(weight.shape),
        "top1_sv": round(top1, 5),
        "sv_median": round(float(s[len(s) // 2].item()), 5),
        "sv_min": round(float(s[-1].item()), 6),
        "frob_norm": round(math.sqrt(frob_sq), 5),
        "stable_rank": round(srank, 3),
        "effective_rank": round(eff_rank, 3),
        "top10_energy_ratio": round(float((s[:10] ** 2).sum().item()) / (frob_sq + 1e-12), 5),
        "top64_energy_ratio": round(float((s[:64] ** 2).sum().item()) / (frob_sq + 1e-12), 5),
    }


def load_or_reuse_gemma4(device: torch.device, force_reload: bool = False):
    g = globals()
    if (not force_reload) and "MODEL" in g and g["MODEL"] is not None and type(g["MODEL"]).__name__ == "Gemma4ForConditionalGeneration":
        print(f"[probe] Reusing already-loaded {g.get('REPO_ID')} on {device}")
        return g["REPO_ID"], g["CFG"], g["TOKENIZER"], g["MODEL"]

    # Free any wrongly-loaded causal wrapper first
    if "MODEL" in g and g["MODEL"] is not None:
        del g["MODEL"]
        import gc
        gc.collect()

    from transformers import AutoConfig, AutoTokenizer, Gemma4ForConditionalGeneration

    candidates = ["unsloth/gemma-4-E2B-it", "google/gemma-4-E2B-it"]
    hf_token = os.environ.get("HF_TOKEN") or None
    if hf_token:
        candidates = ["google/gemma-4-E2B-it", "unsloth/gemma-4-E2B-it"]

    last_err = None
    for repo_id in candidates:
        try:
            print(f"[probe] Loading {repo_id} via Gemma4ForConditionalGeneration (bfloat16)...", flush=True)
            t0 = time.time()
            cfg = AutoConfig.from_pretrained(repo_id, token=hf_token)
            tokenizer = AutoTokenizer.from_pretrained(repo_id, token=hf_token)
            model = Gemma4ForConditionalGeneration.from_pretrained(
                repo_id,
                dtype=torch.bfloat16,
                token=hf_token,
            )
            # Offload unused multimodal encoders from TPU HBM while keeping 100% of language_model + lm_head intact
            inner = getattr(model, "model", model)
            for tower_name in ["vision_tower", "audio_tower", "embed_vision", "embed_audio"]:
                if hasattr(inner, tower_name) and getattr(inner, tower_name) is not None:
                    setattr(inner, tower_name, None)
            import gc
            gc.collect()

            model.eval()
            model.to(device)
            sync_device()
            print(f"[probe] Gemma4ForConditionalGeneration ready on {device} in {time.time() - t0:.2f}s", flush=True)
            return repo_id, cfg, tokenizer, model
        except Exception as e:
            print(f"[probe] Candidate {repo_id} failed: {e}", flush=True)
            last_err = e
    raise RuntimeError(f"Could not load Gemma 4 E2B-it: {last_err}")


def locate_text_backbone(model):
    for attr_path in [
        ("model", "language_model"),
        ("language_model", "model"),
        ("language_model",),
        ("text_model",),
        ("model",),
    ]:
        obj = model
        ok = True
        for a in attr_path:
            if hasattr(obj, a):
                obj = getattr(obj, a)
            else:
                ok = False
                break
        if ok and hasattr(obj, "layers"):
            return ".".join(attr_path), obj
    raise AttributeError(f"Cannot locate decoder layers in {type(model).__name__}: {[n for n, _ in model.named_children()]}")


device = get_device()
out_dir = Path("/content/gemma4_probe")
out_dir.mkdir(parents=True, exist_ok=True)

report = {"hardware": get_hardware_info(device)}
print("[probe] Hardware:", json.dumps(report["hardware"], indent=2), flush=True)

REPO_ID, CFG, TOKENIZER, MODEL = load_or_reuse_gemma4(device)
DEVICE = device
BACKBONE_PATH, TEXT_MODEL = locate_text_backbone(MODEL)

report["repo_id"] = REPO_ID
report["model_class"] = type(MODEL).__name__
report["backbone_path"] = BACKBONE_PATH
report["text_model_class"] = type(TEXT_MODEL).__name__
report["decoder_layer_class"] = type(TEXT_MODEL.layers[0]).__name__

# Dump exact modeling_gemma4 module source file
import transformers.models.gemma4.modeling_gemma4 as mod_g4

mod_src = inspect.getsource(mod_g4)
(out_dir / "modeling_gemma4_exact.py").write_text(mod_src)
print(f"[probe] Dumped modeling_gemma4.py ({len(mod_src)} chars)", flush=True)

# Parameter census
total_params = sum(p.numel() for p in MODEL.parameters())
text_params = sum(p.numel() for p in TEXT_MODEL.parameters())
ple_table = getattr(TEXT_MODEL, "embed_tokens_per_layer", None)
ple_table_params = ple_table.weight.numel() if ple_table is not None else 0
embed_params = TEXT_MODEL.embed_tokens.weight.numel() if hasattr(TEXT_MODEL, "embed_tokens") else 0

report["param_census"] = {
    "total_loaded_params": total_params,
    "text_backbone_params": text_params,
    "main_embed_tokens_params": embed_params,
    "ple_embed_table_params": ple_table_params,
    "non_embedding_compute_params": text_params - embed_params - ple_table_params,
    "num_layers": len(TEXT_MODEL.layers),
    "text_config": TEXT_MODEL.config.to_dict(),
    "text_model_children": [name for name, _ in TEXT_MODEL.named_children()],
    "layer0_children": [name for name, _ in TEXT_MODEL.layers[0].named_children()],
    "attn0_children": [name for name, _ in TEXT_MODEL.layers[0].self_attn.named_children()],
    "layer_last_attn_children": [name for name, _ in TEXT_MODEL.layers[-1].self_attn.named_children()],
}
print("[probe] Census:", json.dumps({k: v for k, v in report["param_census"].items() if k != "text_config"}, indent=2), flush=True)

# Layer-by-layer weight & spectral anatomy
print("[probe] Computing SVD & norm anatomy across all layers...", flush=True)
layer_stats = []
for idx, layer in enumerate(TEXT_MODEL.layers):
    entry = {"layer": idx}
    for attr in ["layer_type", "is_sliding"]:
        if hasattr(layer, attr):
            entry[attr] = getattr(layer, attr)
    attn = layer.self_attn
    for attr in ["is_kv_shared_layer", "kv_shared_layer_index", "is_sliding", "head_dim", "use_k_eq_v", "k_eq_v"]:
        if hasattr(attn, attr):
            val = getattr(attn, attr)
            entry[f"attn_{attr}"] = val if isinstance(val, (int, float, bool, str, type(None))) else str(val)
    entry["attn_submodules"] = [n for n, _ in attn.named_children()]

    if hasattr(layer, "layer_scalar") and layer.layer_scalar is not None:
        ls = layer.layer_scalar.detach().to("cpu", dtype=torch.float32)
        entry["layer_scalar"] = round(float(ls.mean().item()), 6) if ls.numel() == 1 else [round(x, 5) for x in ls.flatten()[:8].tolist()]

    for norm_name in [
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
        "post_per_layer_input_norm",
    ]:
        mod_n = getattr(layer, norm_name, None)
        if mod_n is not None and hasattr(mod_n, "weight") and mod_n.weight is not None:
            nw = mod_n.weight.detach().to("cpu", dtype=torch.float32)
            entry[f"{norm_name}_rms"] = round(float(nw.pow(2).mean().sqrt().item()), 5)
            entry[f"{norm_name}_mean"] = round(float(nw.mean().item()), 5)

    for qk_norm in ["q_norm", "k_norm", "v_norm"]:
        mod_qk = getattr(attn, qk_norm, None)
        if mod_qk is not None and hasattr(mod_qk, "weight") and mod_qk.weight is not None:
            nw = mod_qk.weight.detach().to("cpu", dtype=torch.float32)
            entry[f"attn_{qk_norm}_rms"] = round(float(nw.pow(2).mean().sqrt().item()), 5)

    if hasattr(layer, "per_layer_input_gate") and layer.per_layer_input_gate is not None:
        entry["ple_gate_svd"] = stable_rank_and_spectrum(layer.per_layer_input_gate.weight)
    if hasattr(layer, "per_layer_projection") and layer.per_layer_projection is not None:
        entry["ple_proj_svd"] = stable_rank_and_spectrum(layer.per_layer_projection.weight)
    if hasattr(attn, "q_proj") and attn.q_proj is not None:
        entry["q_proj_svd"] = stable_rank_and_spectrum(attn.q_proj.weight)
    if hasattr(attn, "o_proj") and attn.o_proj is not None:
        entry["o_proj_svd"] = stable_rank_and_spectrum(attn.o_proj.weight)
    if hasattr(layer.mlp, "down_proj") and layer.mlp.down_proj is not None:
        entry["mlp_down_svd"] = stable_rank_and_spectrum(layer.mlp.down_proj.weight)

    layer_stats.append(entry)

report["layer_weight_anatomy"] = layer_stats

# Inspect PLE embedding table norms across token types
if ple_table is not None:
    ple_w = ple_table.weight.detach().to("cpu", dtype=torch.float32)  # [V, L * d_ple]
    n_layers = len(TEXT_MODEL.layers)
    d_ple = ple_w.shape[1] // n_layers
    ple_w_3d = ple_w.view(ple_w.shape[0], n_layers, d_ple)
    sample_tokens = {
        "syntax_stop": ["the", " is", ",", ".", "\n", " of", " and", "="],
        "numbers_math": ["2", "16", "100", "+", "*", "/", "x", "0"],
        "domain_content": ["quantum", "mitochondria", "eigenvalue", "Python", "derivative", "velocity", "theorem", "algorithm"],
    }
    ple_token_probe = {}
    for group, toks in sample_tokens.items():
        ids = []
        for t in toks:
            enc_ids = TOKENIZER.encode(t, add_special_tokens=False)
            if enc_ids:
                ids.append(enc_ids[-1])
        sub = ple_w_3d[ids]  # [k, L, d_ple]
        norms_by_layer = sub.norm(dim=-1).mean(dim=0).tolist()
        ple_token_probe[group] = {
            "token_ids": ids,
            "mean_norm_L0": round(norms_by_layer[0], 5),
            "mean_norm_L14": round(norms_by_layer[14], 5),
            "mean_norm_L25": round(norms_by_layer[25], 5),
            "mean_norm_L34": round(norms_by_layer[-1], 5),
            "all_layers_mean_norm": [round(x, 4) for x in norms_by_layer],
        }
    report["ple_token_group_norms"] = ple_token_probe

# Check special tokens and chat template (including thinking mode!)
special_probe = {}
for tok_str in ["<|think|>", "<|thought|>", "<thought>", "</thought>", "<start_of_turn>", "<end_of_turn>"]:
    special_probe[tok_str] = TOKENIZER.convert_tokens_to_ids(tok_str)
report["special_tokens"] = special_probe
report["chat_template"] = getattr(TOKENIZER, "chat_template", None)

# Activation MRI on diagnostic reasoning prompts
prompts = [
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
    "In quantum mechanics, consider a particle in a 1D infinite square well of width L. If the width is suddenly doubled to 2L without changing the wavefunction, what is the probability of finding the particle in the new ground state?",
    "Write a Python function `first_repeated_char(s)` to find the first repeated character in a given string.",
]

formatted = []
for p in prompts:
    msgs = [{"role": "user", "content": p}]
    try:
        formatted.append(TOKENIZER.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    except Exception:
        formatted.append(p)
report["formatted_prompt_sample"] = formatted[0]

act_records = {
    i: {
        "h_in_norm": [],
        "attn_post_norm": [],
        "mlp_post_norm": [],
        "ple_pre_norm_mean": [],
        "ple_pre_norm_cv": [],
        "ple_post_norm_mean": [],
        "ple_post_norm_cv": [],
        "h_out_norm": [],
        "cos_in_out": [],
    }
    for i in range(len(TEXT_MODEL.layers))
}

hooks = []
for idx, layer in enumerate(TEXT_MODEL.layers):
    if getattr(layer, "post_attention_layernorm", None) is not None:
        hooks.append(
            layer.post_attention_layernorm.register_forward_hook(
                lambda _m, _i, o, l=idx: act_records[l]["attn_post_norm"].append(
                    float(o.detach().to("cpu", dtype=torch.float32).norm(dim=-1).mean().item())
                )
            )
        )
    if getattr(layer, "post_feedforward_layernorm", None) is not None:
        hooks.append(
            layer.post_feedforward_layernorm.register_forward_hook(
                lambda _m, _i, o, l=idx: act_records[l]["mlp_post_norm"].append(
                    float(o.detach().to("cpu", dtype=torch.float32).norm(dim=-1).mean().item())
                )
            )
        )
    if getattr(layer, "per_layer_projection", None) is not None:
        def _mk_pre(l):
            def _hk(_m, _i, o):
                n = o.detach().to("cpu", dtype=torch.float32).norm(dim=-1)
                act_records[l]["ple_pre_norm_mean"].append(float(n.mean().item()))
                act_records[l]["ple_pre_norm_cv"].append(float((n.std() / (n.mean() + 1e-8)).item()))
            return _hk
        hooks.append(layer.per_layer_projection.register_forward_hook(_mk_pre(idx)))
    if getattr(layer, "post_per_layer_input_norm", None) is not None:
        def _mk_post(l):
            def _hk(_m, _i, o):
                n = o.detach().to("cpu", dtype=torch.float32).norm(dim=-1)
                act_records[l]["ple_post_norm_mean"].append(float(n.mean().item()))
                act_records[l]["ple_post_norm_cv"].append(float((n.std() / (n.mean() + 1e-8)).item()))
            return _hk
        hooks.append(layer.post_per_layer_input_norm.register_forward_hook(_mk_post(idx)))

    def _mk_layer(l):
        def _hk(_m, inp, out):
            hi = inp[0].detach().to("cpu", dtype=torch.float32)
            ho = (out[0] if isinstance(out, tuple) else out).detach().to("cpu", dtype=torch.float32)
            act_records[l]["h_in_norm"].append(float(hi.norm(dim=-1).mean().item()))
            act_records[l]["h_out_norm"].append(float(ho.norm(dim=-1).mean().item()))
            cos = torch.nn.functional.cosine_similarity(hi, ho, dim=-1)
            act_records[l]["cos_in_out"].append(float(cos.mean().item()))
        return _hk
    hooks.append(layer.register_forward_hook(_mk_layer(idx)))

print("[probe] Running forward activation MRI...", flush=True)
with torch.no_grad():
    for text in formatted:
        enc = TOKENIZER(text, return_tensors="pt").to(device)
        _ = MODEL(**enc)
        sync_device()

for h in hooks:
    h.remove()

def _avg(lst):
    return round(sum(lst) / max(1, len(lst)), 5)

act_summary = []
for idx in range(len(TEXT_MODEL.layers)):
    r = act_records[idx]
    act_summary.append({
        "layer": idx,
        "h_in_norm": _avg(r["h_in_norm"]),
        "attn_update_norm": _avg(r["attn_post_norm"]),
        "mlp_update_norm": _avg(r["mlp_post_norm"]),
        "ple_pre_norm_mean": _avg(r["ple_pre_norm_mean"]),
        "ple_pre_norm_cv": _avg(r["ple_pre_norm_cv"]),
        "ple_post_norm_mean": _avg(r["ple_post_norm_mean"]),
        "ple_post_norm_cv": _avg(r["ple_post_norm_cv"]),
        "h_out_norm": _avg(r["h_out_norm"]),
        "cos_in_out": _avg(r["cos_in_out"]),
    })
report["activation_mri"] = act_summary

# Quick generation & latency smoke test (warmup + timed decode)
print("[probe] Running generation & throughput smoke test...", flush=True)
test_enc = TOKENIZER(formatted[0], return_tensors="pt").to(device)
with torch.no_grad():
    _ = MODEL.generate(**test_enc, max_new_tokens=16, do_sample=False)
    sync_device()
    t_gen = time.time()
    gen_out = MODEL.generate(**test_enc, max_new_tokens=96, do_sample=False)
    sync_device()
    dt_gen = time.time() - t_gen

new_toks = gen_out.shape[1] - test_enc["input_ids"].shape[1]
gen_text = TOKENIZER.decode(gen_out[0, test_enc["input_ids"].shape[1]:], skip_special_tokens=False)
report["smoke_generation"] = {
    "new_tokens": int(new_toks),
    "elapsed_sec": round(dt_gen, 3),
    "tokens_per_sec": round(new_toks / max(dt_gen, 1e-6), 2),
    "output_preview": gen_text,
}
print(f"[probe] Smoke decode: {new_toks} toks in {dt_gen:.2f}s ({new_toks/max(dt_gen,1e-6):.1f} tok/s)", flush=True)
print("[probe] Output preview:", gen_text[:300], flush=True)

out_json = out_dir / "phase1_anatomy_report.json"
out_json.write_text(json.dumps(report, indent=2))
print(f"\n[probe] Saved complete MRI report -> {out_json}", flush=True)

print("\n=== LAYER-BY-LAYER MRI SUMMARY ===")
for row in act_summary:
    l = row["layer"]
    w = layer_stats[l]
    print(
        f"L{l:02d} ({str(w.get('layer_type','?'))[:7]:7s}, kv_sh={int(bool(w.get('attn_is_kv_shared_layer')))}<-{str(w.get('attn_kv_shared_layer_index')):>4s}) | "
        f"sc={w.get('layer_scalar', 1.0):.3f} | "
        f"h:{row['h_in_norm']:6.1f}->{row['h_out_norm']:6.1f} | "
        f"attn={row['attn_update_norm']:5.1f} mlp={row['mlp_update_norm']:5.1f} ple={row['ple_post_norm_mean']:5.1f} | "
        f"PLE_CV:{row['ple_pre_norm_cv']:.3f}->{row['ple_post_norm_cv']:.3f} | "
        f"cos={row['cos_in_out']:.4f}"
    )
