"""
Static-Graph Pure-Backbone Batched Inference Engine for Gemma 4 (`Gemma4TextModel` + `lm_head`).

Eliminates two major bottlenecks observed in default `Gemma4ForConditionalGeneration.generate`:
1. Bypasses outer multimodal wrapper overhead (`get_placeholder_mask` + `[B, S, 1536]` clones).
2. Replaces growing `DynamicCache` and Python-int slice constants with:
   - Pre-allocated KV buffers ONLY for the 15 KV-computing layers (`L0..L14`), saving 57% KV HBM.
   - Device-tensor `index_copy_(2, cur_pos_tensor, ...)` updates and tensor-derived 4D masks so
     every decode step `s = 0 .. max_new_tokens - 1` produces a 100% identical HLO graph on TPU/XLA!
"""

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch
import torch.nn as nn


def get_step_barrier(device: torch.device):
    """Return per-step XLA graph boundary function on TPU, or no-op on CUDA/CPU."""
    if device.type == "tpu":
        return torch.accelerator.synchronize
    elif device.type == "xla":
        import torch_xla.core.xla_model as xm  # type: ignore

        return xm.mark_step
    return lambda: None


class Gemma4StaticKVCache:
    """
    Shape-invariant KV cache allocating buffers ONLY for non-KV-shared layers (`L0..L14`).
    Uses device-tensor `index_copy_` during decode so XLA compiles the decode step ONCE.
    """

    def __init__(
        self,
        text_config: Any,
        batch_size: int,
        max_total_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.batch_size = batch_size
        self.max_total_len = max_total_len
        self.device = device
        self.dtype = dtype
        self.first_kv_shared = text_config.num_hidden_layers - getattr(text_config, "num_kv_shared_layers", 0)
        self.k_buf: Dict[int, torch.Tensor] = {}
        self.v_buf: Dict[int, torch.Tensor] = {}

        for l in range(self.first_kv_shared):
            l_cfg = text_config.per_layer_config[l]
            kv_heads = l_cfg.num_key_value_heads
            head_dim = l_cfg.head_dim
            self.k_buf[l] = torch.zeros(
                (batch_size, kv_heads, max_total_len, head_dim), device=device, dtype=dtype
            )
            self.v_buf[l] = torch.zeros(
                (batch_size, kv_heads, max_total_len, head_dim), device=device, dtype=dtype
            )

        self.mode: str = "prefill"
        self.prefill_positions: Optional[torch.Tensor] = None
        self.decode_pos: Optional[torch.Tensor] = None  # shape [1], int64 on device

    def reset_zero_(self) -> None:
        for l in self.k_buf:
            self.k_buf[l].zero_()
            self.v_buf[l].zero_()

    def set_prefill(self, prefill_positions: torch.Tensor) -> None:
        """`prefill_positions`: 1D int64 device tensor `arange(P)`."""
        self.mode = "prefill"
        self.prefill_positions = prefill_positions
        self.decode_pos = None

    def set_decode(self, decode_pos: torch.Tensor) -> None:
        """`decode_pos`: 1D int64 device tensor of shape `[1]`."""
        self.mode = "decode"
        self.decode_pos = decode_pos

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        kb = self.k_buf[layer_idx]
        vb = self.v_buf[layer_idx]
        if self.mode == "prefill":
            assert self.prefill_positions is not None
            kb.index_copy_(2, self.prefill_positions, key_states)
            vb.index_copy_(2, self.prefill_positions, value_states)
            return key_states, value_states
        else:
            assert self.decode_pos is not None
            kb.index_copy_(2, self.decode_pos, key_states)
            vb.index_copy_(2, self.decode_pos, value_states)
            return kb, vb


class StaticGemma4Engine:
    """High-throughput deterministic greedy generation engine on Gemma 4's pure text backbone."""

    def __init__(
        self,
        cond_or_causal_model: nn.Module,
        tokenizer: Any,
        device: torch.device,
    ):
        self.outer_model = cond_or_causal_model
        if hasattr(cond_or_causal_model, "model") and hasattr(cond_or_causal_model.model, "language_model"):
            self.text_model = cond_or_causal_model.model.language_model
        elif hasattr(cond_or_causal_model, "model"):
            self.text_model = cond_or_causal_model.model
        else:
            raise ValueError("Unsupported Gemma 4 wrapper structure")

        self.lm_head = cond_or_causal_model.lm_head
        self.config = self.text_model.config
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = next(self.text_model.parameters()).dtype
        self.softcap: Optional[float] = getattr(self.config, "final_logit_softcapping", None)
        self.sliding_window: int = int(getattr(self.config, "sliding_window", 512) or 512)
        self.pad_token_id: int = int(getattr(self.config, "pad_token_id", 0) or 0)
        self.mark_step = get_step_barrier(device)

        # Collect all stop token IDs (<eos>, <turn|>, <end_of_turn>)
        stop_ids = set()
        eos_cfg = getattr(self.config, "eos_token_id", None)
        if isinstance(eos_cfg, int):
            stop_ids.add(eos_cfg)
        elif isinstance(eos_cfg, (list, tuple)):
            stop_ids.update(int(x) for x in eos_cfg)
        for tok_str in ["<eos>", "<turn|>", "<end_of_turn>"]:
            tid = tokenizer.convert_tokens_to_ids(tok_str)
            if isinstance(tid, int) and tid >= 0 and tid != tokenizer.unk_token_id:
                stop_ids.add(tid)
        self.stop_ids_list = sorted(stop_ids)
        self.stop_ids_dev = torch.tensor(self.stop_ids_list, device=device, dtype=torch.long)

        # Reusable KV cache pool keyed by (batch_size, max_total_len)
        self._cache_pool: Dict[Tuple[int, int], Gemma4StaticKVCache] = {}

    def _get_cache(self, batch_size: int, max_total_len: int) -> Gemma4StaticKVCache:
        key = (batch_size, max_total_len)
        if key not in self._cache_pool:
            # Keep pool bounded to avoid HBM fragmentation
            if len(self._cache_pool) >= 2:
                self._cache_pool.clear()
            self._cache_pool[key] = Gemma4StaticKVCache(
                self.config, batch_size, max_total_len, self.device, self.dtype
            )
        cache = self._cache_pool[key]
        cache.reset_zero_()
        return cache

    def _head_argmax(self, last_hidden: torch.Tensor) -> torch.Tensor:
        """
        Compute greedy next token `[B, 1]` from `last_hidden` `[B, 1, D]`.
        Note: `tanh(x / c) * c` is strictly monotonic increasing on (-inf, +inf), so
        `argmax(softcap(logits)) == argmax(logits)` identically while saving a 262k tanh!
        """
        logits = self.lm_head(last_hidden)  # [B, 1, V]
        return logits.argmax(dim=-1)  # [B, 1]

    def format_chat_prompts(
        self,
        user_prompts: Sequence[str],
        enable_thinking: bool = False,
        system_prompt: Optional[str] = None,
    ) -> List[str]:
        formatted = []
        for p in user_prompts:
            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({"role": "user", "content": p})
            txt = self.tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            formatted.append(txt)
        return formatted

    @torch.no_grad()
    def generate_batch_static(
        self,
        formatted_prompts: Sequence[str],
        max_new_tokens: int = 256,
        prompt_pad_bucket: int = 256,
        early_stop_check_every: int = 16,
        sync_every: int = 4,
    ) -> Dict[str, Any]:
        """
        Fixed-shape bucketed greedy generation for a single batch of `B` formatted strings.
        All tensors inside the decode loop have invariant shapes `[B, 1]` and `[B, 1, M, d]`.
        """
        B = len(formatted_prompts)
        enc = [self.tokenizer.encode(t, add_special_tokens=False) for t in formatted_prompts]
        max_p_actual = max(len(ids) for ids in enc)
        # Round prompt_pad_len up to a multiple of prompt_pad_bucket so shapes stay bucketed
        P = max(prompt_pad_bucket, ((max_p_actual + prompt_pad_bucket - 1) // prompt_pad_bucket) * prompt_pad_bucket)
        sync_k = max(1, int(sync_every))
        # Align (G - 1) to an exact multiple of sync_k so every decode block has identical HLO shape!
        rem_steps = max(0, max_new_tokens - 1)
        aligned_rem = ((rem_steps + sync_k - 1) // sync_k) * sync_k
        G = 1 + aligned_rem
        M = P + G

        input_cpu = torch.full((B, P), self.pad_token_id, dtype=torch.long)
        prompt_lens_cpu = torch.zeros((B,), dtype=torch.long)
        for i, ids in enumerate(enc):
            if len(ids) > P:
                ids = ids[-P:]
            L = len(ids)
            input_cpu[i, P - L : P] = torch.tensor(ids, dtype=torch.long)
            prompt_lens_cpu[i] = L

        t_start = time.perf_counter()
        input_ids = input_cpu.to(self.device)
        prompt_lens = prompt_lens_cpu.to(self.device)  # [B]
        valid_start = (P - prompt_lens)[:, None, None, None]  # [B, 1, 1, 1]

        cache = self._get_cache(B, M)
        zero_bf16 = torch.zeros((), device=self.device, dtype=self.dtype)
        neg_inf_bf16 = torch.full((), torch.finfo(self.dtype).min, device=self.device, dtype=self.dtype)

        # ---- 1. Static-Shape Prefill ([B, P]) ----
        prefill_arange = torch.arange(P, device=self.device, dtype=torch.long)
        q_idx = prefill_arange[None, None, :, None]  # [1, 1, P, 1]
        k_idx_p = prefill_arange[None, None, None, :]  # [1, 1, 1, P]

        is_pad_q = q_idx < valid_start
        pad_self = is_pad_q & (k_idx_p == q_idx)
        real_causal = (~is_pad_q) & (k_idx_p >= valid_start) & (k_idx_p <= q_idx)

        allow_full_p = pad_self | real_causal
        allow_slid_p = pad_self | (real_causal & ((q_idx - k_idx_p) < self.sliding_window))

        prefill_masks = {
            "full_attention": torch.where(allow_full_p, zero_bf16, neg_inf_bf16),
            "sliding_attention": torch.where(allow_slid_p, zero_bf16, neg_inf_bf16),
        }
        prefill_pos_ids = (q_idx.squeeze(1).squeeze(-1) - valid_start.squeeze(-1).squeeze(-1)).clamp(min=0)  # [B, P]

        cache.set_prefill(prefill_arange)
        out_p = self.text_model(
            input_ids=input_ids,
            attention_mask=prefill_masks,
            position_ids=prefill_pos_ids,
            past_key_values=cache,
            use_cache=True,
        )
        next_tok = self._head_argmax(out_p.last_hidden_state[:, -1:, :])  # [B, 1]

        gen_tokens = torch.full((B, G), self.pad_token_id, device=self.device, dtype=torch.long)
        finished = torch.zeros((B, 1), device=self.device, dtype=torch.bool)

        gen_tokens[:, 0:1] = next_tok
        finished = finished | torch.isin(next_tok, self.stop_ids_dev)
        self.mark_step()

        # ---- 2. Shape-Invariant Decode Loop ([B, 1] against [B, 1, M, d]) ----
        k_idx_m = torch.arange(M, device=self.device, dtype=torch.long)[None, None, None, :]  # [1, 1, 1, M]
        # Pre-materialize step index tensors on device so zero Python ints enter HLO!
        step_cache_positions = torch.arange(P, P + G, device=self.device, dtype=torch.long).unsqueeze(1)  # [G, 1]
        step_offsets = torch.arange(0, G, device=self.device, dtype=torch.long).unsqueeze(1)  # [G, 1]
        pad_tok_tensor = torch.full((B, 1), self.pad_token_id, device=self.device, dtype=torch.long)

        steps_run = 1
        for s in range(G - 1):
            cur_pos_1d = step_cache_positions[s]  # shape [1] on device
            cur_pos_4d = cur_pos_1d.view(1, 1, 1, 1)
            dec_pos_ids = prompt_lens[:, None] + step_offsets[s]  # [B, 1] on device

            allow_full_d = (k_idx_m >= valid_start) & (k_idx_m <= cur_pos_4d)
            allow_slid_d = allow_full_d & ((cur_pos_4d - k_idx_m) < self.sliding_window)
            decode_masks = {
                "full_attention": torch.where(allow_full_d, zero_bf16, neg_inf_bf16),
                "sliding_attention": torch.where(allow_slid_d, zero_bf16, neg_inf_bf16),
            }

            cache.set_decode(cur_pos_1d)
            out_d = self.text_model(
                input_ids=next_tok,
                attention_mask=decode_masks,
                position_ids=dec_pos_ids,
                past_key_values=cache,
                use_cache=True,
            )
            cand_tok = self._head_argmax(out_d.last_hidden_state)  # [B, 1]
            next_tok = torch.where(finished, pad_tok_tensor, cand_tok)
            # Device-indexed write into gen_tokens[:, s+1] without Python slice constant if desired:
            write_col = step_offsets[s + 1]  # shape [1]
            gen_tokens.index_copy_(1, write_col, next_tok)
            finished = finished | torch.isin(next_tok, self.stop_ids_dev)
            steps_run += 1

            if (s + 1) % sync_k == 0:
                self.mark_step()
                if early_stop_check_every > 0 and ((s + 1) % early_stop_check_every == 0):
                    if bool(finished.all().cpu().item()):
                        break

        if self.device.type == "tpu" and hasattr(torch.accelerator, "synchronize"):
            torch.accelerator.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t_start

        actual_limit = min(steps_run, max_new_tokens)
        gen_cpu = gen_tokens[:, :actual_limit].cpu().tolist()
        stop_set = set(self.stop_ids_list)
        texts: List[str] = []
        tok_counts: List[int] = []
        for row in gen_cpu:
            trimmed = []
            for tid in row:
                if tid in stop_set:
                    break
                trimmed.append(tid)
            tok_counts.append(len(trimmed))
            texts.append(self.tokenizer.decode(trimmed, skip_special_tokens=False))

        total_gen_toks = sum(tok_counts)
        return {
            "texts": texts,
            "token_counts": tok_counts,
            "total_generated_tokens": total_gen_toks,
            "steps_executed": actual_limit,
            "batch_size": B,
            "prompt_pad_len": P,
            "elapsed_sec": round(elapsed, 4),
            "effective_tok_per_sec": round(total_gen_toks / max(elapsed, 1e-6), 2),
            "raw_step_tok_per_sec": round((B * actual_limit) / max(elapsed, 1e-6), 2),
        }

    def generate_dataset(
        self,
        prompts: Sequence[str],
        batch_size: int = 32,
        max_new_tokens: int = 256,
        prompt_pad_bucket: int = 256,
        enable_thinking: bool = False,
        system_prompt: Optional[str] = None,
        sync_every: int = 4,
    ) -> Dict[str, Any]:
        """Run batched generation across an entire dataset, padding tail batch for shape reuse."""
        formatted = self.format_chat_prompts(
            prompts, enable_thinking=enable_thinking, system_prompt=system_prompt
        )
        all_texts: List[str] = []
        all_counts: List[int] = []
        total_elapsed = 0.0

        n = len(formatted)
        for start in range(0, n, batch_size):
            chunk = list(formatted[start : start + batch_size])
            real_len = len(chunk)
            if real_len < batch_size:
                # Pad tail batch with copies of last prompt so batch_size stays identical (zero XLA recompile!)
                chunk.extend([chunk[-1]] * (batch_size - real_len))
            res = self.generate_batch_static(
                chunk,
                max_new_tokens=max_new_tokens,
                prompt_pad_bucket=prompt_pad_bucket,
                sync_every=sync_every,
            )
            all_texts.extend(res["texts"][:real_len])
            all_counts.extend(res["token_counts"][:real_len])
            total_elapsed += res["elapsed_sec"]

        tot_toks = sum(all_counts)
        return {
            "texts": all_texts,
            "token_counts": all_counts,
            "total_tokens": tot_toks,
            "total_elapsed_sec": round(total_elapsed, 3),
            "tok_per_sec": round(tot_toks / max(total_elapsed, 1e-6), 2),
        }
