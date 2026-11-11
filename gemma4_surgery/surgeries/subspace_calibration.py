"""
Closed-Form (Zero-Gradient) Static-Shape Subspace Extraction for Gemma 4 Weight Surgeries.

All forward passes use left-padded fixed `[batch_size, pad_len]` shapes matching `StaticGemma4Engine`
so TPU/XLA (`torch_tpu`) performs ZERO extra graph compilations!
"""

from typing import Any, Dict, List, Sequence
import torch
import torch.nn as nn


SURFACE_STOP_STRINGS = [
    "\n",
    "\n\n",
    " ",
    "the",
    " the",
    "The",
    "a",
    " a",
    "is",
    " is",
    "of",
    " of",
    "to",
    " to",
    "and",
    " and",
    "in",
    " in",
    ",",
    ".",
    ":",
    ";",
    "(",
    ")",
]


PROSE_FILLER_STRINGS = [
    "the",
    " the",
    "The",
    "a",
    " a",
    "an",
    " an",
    "of",
    " of",
    "to",
    " to",
    "that",
    " that",
    "it",
    " it",
    "Here",
    "Sure",
    "Let",
    "Well",
    "So",
    "Basically",
]


@torch.no_grad()
def compute_prose_filler_subspace(
    lm_head_weight: torch.Tensor,
    tokenizer: Any,
    rank: int = 8,
) -> torch.Tensor:
    """
    Code-safe analytical extraction of the top-`rank` orthonormal unembedding subspace
    spanned strictly by conversational English prose-filler/article tokens (preserving
    all Python whitespace `\n`/` `, punctuation `:()`, and keyword tokens `in`/`is`/`and`).
    """
    tok_ids = set()
    for s in PROSE_FILLER_STRINGS:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            tok_ids.add(ids[0])
    sorted_ids = sorted(tok_ids)
    rows = lm_head_weight[sorted_ids].detach().float().cpu()
    rows = rows / (rows.norm(dim=-1, keepdim=True) + 1e-8)
    u, _, _ = torch.linalg.svd(rows.T, full_matrices=False)
    r = min(rank, u.shape[1])
    return u[:, :r].contiguous()


@torch.no_grad()
def compute_surface_stop_subspace(
    lm_head_weight: torch.Tensor,
    tokenizer: Any,
    rank: int = 8,
) -> torch.Tensor:
    """
    Analytical (zero-data) extraction of the top-`rank` orthonormal unembedding subspace
    spanned by high-frequency surface/formatting/stop tokens (`\n`, `:`, `the`, etc.).
    """
    tok_ids = set()
    for s in SURFACE_STOP_STRINGS:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            tok_ids.add(ids[0])
    sorted_ids = sorted(tok_ids)
    rows = lm_head_weight[sorted_ids].detach().float().cpu()  # [K, D]
    rows = rows / (rows.norm(dim=-1, keepdim=True) + 1e-8)
    u, _, _ = torch.linalg.svd(rows.T, full_matrices=False)
    r = min(rank, u.shape[1])
    return u[:, :r].contiguous()


def _build_forbidden_tag_basis(lm_head_weight: torch.Tensor, tokenizer: Any) -> torch.Tensor:
    tag_ids = set()
    for tag in ["<|channel>", "<channel|>", "<|think|>", "thought", "<|thought|>"]:
        tid = tokenizer.convert_tokens_to_ids(tag)
        if isinstance(tid, int) and tid >= 0 and tid != tokenizer.unk_token_id:
            tag_ids.add(tid)
        tag_ids.update(tokenizer.encode(tag, add_special_tokens=False))
    tag_vecs = lm_head_weight[sorted(tag_ids)].detach().float().cpu()
    q_forbid, _ = torch.linalg.qr(tag_vecs.T)
    return q_forbid


@torch.no_grad()
def extract_last_token_states_static(
    text_model: nn.Module,
    tokenizer: Any,
    formatted_prompts: Sequence[str],
    target_layers: Sequence[int],
    batch_size: int = 32,
    pad_len: int = 512,
    device: torch.device = torch.device("cpu"),
) -> Dict[int, torch.Tensor]:
    """
    Extract final-prompt-token residual states `[N, hidden_size]` at `target_layers` using
    strictly fixed `[batch_size, pad_len]` left-padded batches (zero XLA recompilation!).
    """
    pad_id = int(getattr(text_model.config, "pad_token_id", 0) or 0)
    sliding_window = int(getattr(text_model.config, "sliding_window", 512) or 512)
    dtype = next(text_model.parameters()).dtype

    captured: Dict[int, torch.Tensor] = {}
    hooks = []

    def make_hook(l_idx: int):
        def _hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured[l_idx] = h[:, -1, :].detach().float().cpu()

        return _hook

    for l in target_layers:
        hooks.append(text_model.layers[l].register_forward_hook(make_hook(l)))

    out_lists: Dict[int, List[torch.Tensor]] = {l: [] for l in target_layers}
    n = len(formatted_prompts)
    zero_bf16 = torch.zeros((), device=device, dtype=dtype)
    neg_inf_bf16 = torch.full((), torch.finfo(dtype).min, device=device, dtype=dtype)
    arange_p = torch.arange(pad_len, device=device, dtype=torch.long)
    q_idx = arange_p[None, None, :, None]
    k_idx = arange_p[None, None, None, :]

    try:
        for start in range(0, n, batch_size):
            chunk = list(formatted_prompts[start : start + batch_size])
            real_n = len(chunk)
            if real_n < batch_size:
                chunk.extend([chunk[-1]] * (batch_size - real_n))

            inp_cpu = torch.full((batch_size, pad_len), pad_id, dtype=torch.long)
            lens_cpu = torch.zeros((batch_size,), dtype=torch.long)
            for i, txt in enumerate(chunk):
                ids = tokenizer.encode(txt, add_special_tokens=False)[-pad_len:]
                L = len(ids)
                inp_cpu[i, pad_len - L : pad_len] = torch.tensor(ids, dtype=torch.long)
                lens_cpu[i] = L

            input_ids = inp_cpu.to(device)
            valid_start = (pad_len - lens_cpu.to(device))[:, None, None, None]
            is_pad_q = q_idx < valid_start
            pad_self = is_pad_q & (k_idx == q_idx)
            real_causal = (~is_pad_q) & (k_idx >= valid_start) & (k_idx <= q_idx)
            masks = {
                "full_attention": torch.where(pad_self | real_causal, zero_bf16, neg_inf_bf16),
                "sliding_attention": torch.where(
                    pad_self | (real_causal & ((q_idx - k_idx) < sliding_window)),
                    zero_bf16,
                    neg_inf_bf16,
                ),
            }
            pos_ids = (q_idx.squeeze(1).squeeze(-1) - valid_start.squeeze(-1).squeeze(-1)).clamp(min=0)

            captured.clear()
            _ = text_model(
                input_ids=input_ids,
                attention_mask=masks,
                position_ids=pos_ids,
                use_cache=False,
            )
            if device.type == "tpu" and hasattr(torch.accelerator, "synchronize"):
                torch.accelerator.synchronize()

            for l in target_layers:
                out_lists[l].append(captured[l][:real_n])
    finally:
        for hk in hooks:
            hk.remove()

    return {l: torch.cat(out_lists[l], dim=0) for l in target_layers}


@torch.no_grad()
def compute_latent_think_subspace(
    text_model: nn.Module,
    lm_head_weight: torch.Tensor,
    tokenizer: Any,
    dev_prompts: Sequence[str],
    target_layers: Sequence[int] = (33, 34),
    rank: int = 4,
    batch_size: int = 32,
    pad_len: int = 512,
    device: torch.device = torch.device("cpu"),
) -> Dict[int, torch.Tensor]:
    """
    Forward-only extraction of top-`rank` principal directions of `h(prompt|think) - h(prompt|direct)`
    on `dev_prompts` in fixed `[batch_size, pad_len]` shape, orthogonalized against thought tags.
    """
    q_forbid = _build_forbidden_tag_basis(lm_head_weight, tokenizer)
    fmt_dir = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        for p in dev_prompts
    ]
    fmt_thk = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        for p in dev_prompts
    ]
    states_dir = extract_last_token_states_static(
        text_model, tokenizer, fmt_dir, target_layers, batch_size=batch_size, pad_len=pad_len, device=device
    )
    states_thk = extract_last_token_states_static(
        text_model, tokenizer, fmt_thk, target_layers, batch_size=batch_size, pad_len=pad_len, device=device
    )

    basis_by_layer: Dict[int, torch.Tensor] = {}
    for l in target_layers:
        mat = (states_thk[l] - states_dir[l]).T  # [D, N_dev]
        mat = mat - q_forbid @ (q_forbid.T @ mat)
        u, _, _ = torch.linalg.svd(mat, full_matrices=False)
        r = min(rank, u.shape[1])
        basis_by_layer[l] = u[:, :r].contiguous()
    return basis_by_layer


@torch.no_grad()
def compute_domain_centered_correctness_subspace(
    text_model: nn.Module,
    lm_head_weight: torch.Tensor,
    tokenizer: Any,
    dev_suites_prompts: Dict[str, Sequence[str]],
    dev_baseline_correct: Dict[str, Sequence[bool]],
    target_layers: Sequence[int] = (33, 34),
    rank: int = 2,
    batch_size: int = 32,
    pad_len: int = 512,
    device: torch.device = torch.device("cpu"),
) -> Dict[int, torch.Tensor]:
    """
    Extract domain-invariant contrastive correctness subspace across `dev` benchmarks:
    Within each benchmark `b`, computes unit-normalized `mean(h | correct, b) - mean(h | wrong, b)`
    so topic differences across benchmarks cancel out, then takes top-`rank` SVD across domains!
    """
    stop_q = compute_surface_stop_subspace(lm_head_weight, tokenizer, rank=8)
    domain_diffs: Dict[int, List[torch.Tensor]] = {l: [] for l in target_layers}

    for bname, raw_prompts in dev_suites_prompts.items():
        flags = list(dev_baseline_correct[bname])
        if sum(flags) == 0 or sum(flags) == len(flags):
            continue
        fmt = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            for p in raw_prompts
        ]
        st = extract_last_token_states_static(
            text_model, tokenizer, fmt, target_layers, batch_size=batch_size, pad_len=pad_len, device=device
        )
        pos_idx = torch.tensor([i for i, f in enumerate(flags) if f], dtype=torch.long)
        neg_idx = torch.tensor([i for i, f in enumerate(flags) if not f], dtype=torch.long)
        for l in target_layers:
            diff = st[l][pos_idx].mean(dim=0) - st[l][neg_idx].mean(dim=0)
            diff = diff - stop_q @ (stop_q.T @ diff)
            diff = diff / (diff.norm() + 1e-8)
            domain_diffs[l].append(diff)

    basis_by_layer: Dict[int, torch.Tensor] = {}
    for l in target_layers:
        mat = torch.stack(domain_diffs[l], dim=1)  # [D, num_domains]
        u, _, _ = torch.linalg.svd(mat, full_matrices=False)
        r = min(rank, u.shape[1])
        basis_by_layer[l] = u[:, :r].contiguous()
    return basis_by_layer
