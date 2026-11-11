"""
Pure JAX / XLA Static-Graph Batched Inference Engine for Gemma 4 E2B-it (`JaxStaticGemma4Engine`).

Implements two idiomatic JAX/XLA static-graph execution modes to compare against PyTorch `StaticGemma4Engine`:
1. `generate_batch_jax_step_loop`:
   Step-by-step `@jax.jit(donate_argnums=...)` loop with `jax.lax.dynamic_update_slice` on the 15-layer
   KV cache (`L0..L14`, `-57.14%` KV HBM) — the direct 1-to-1 JAX analogue to PyTorch `index_copy_`.
2. `generate_batch_jax_scan`:
   Whole-loop compiled XLA execution fusing Prefill + all `G - 1` Autoregressive Decode steps inside a
   single `@jax.jit` + `jax.lax.scan` program, with Zone 1 (`L0..L14`) and Zone 2 (`L15..L34`) structured
   as stacked superblocks. Eliminates 100% of per-token Python host-loop and PJRT dispatch overhead.
"""

import time
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import torch


def torch_bf16_to_jax(t: torch.Tensor, device: Any) -> jax.Array:
    """Transfer a PyTorch tensor to a JAX device preserving exact bfloat16/float32 bits."""
    t_cpu = t.detach().cpu().contiguous()
    if t_cpu.dtype == torch.bfloat16:
        arr_u16 = t_cpu.view(torch.uint16).numpy()
        arr_bf16 = arr_u16.view(jnp.bfloat16)
        return jax.device_put(arr_bf16, device)
    elif t_cpu.dtype == torch.float32:
        return jax.device_put(t_cpu.numpy().astype(jnp.float32), device)
    else:
        return jax.device_put(t_cpu.float().numpy().astype(jnp.bfloat16), device)


def rms_norm(x: jax.Array, weight: Optional[jax.Array], eps: float = 1e-6) -> jax.Array:
    """Exact Gemma4RMSNorm in JAX: float32 variance with power(-0.5) matching modeling_gemma4.py."""
    x_f32 = x.astype(jnp.float32)
    mean_sq = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True) + eps
    normed = x_f32 * jnp.power(mean_sq, -0.5)
    if weight is not None:
        normed = normed * weight.astype(jnp.float32)
    return normed.astype(x.dtype)


def rotate_half(x: jax.Array) -> jax.Array:
    half = x.shape[-1] // 2
    return jnp.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """`x`: `[B, S, H, D]`, `cos`/`sin`: `[B, S, D]`. Broadcasts over heads `H` (axis 2)."""
    cos_b = jnp.expand_dims(cos, axis=2)
    sin_b = jnp.expand_dims(sin, axis=2)
    return (x * cos_b) + (rotate_half(x) * sin_b)


def repeat_kv_jax(x: jax.Array, n_rep: int) -> jax.Array:
    """`x`: `[B, H_kv, S, D]` -> `[B, H_kv * n_rep, S, D]`."""
    if n_rep == 1:
        return x
    B, H_kv, S, D = x.shape
    return jnp.broadcast_to(x[:, :, None, :, :], (B, H_kv, n_rep, S, D)).reshape(B, H_kv * n_rep, S, D)


def attention_core(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: jax.Array,
    n_rep: int,
) -> jax.Array:
    """
    Exact Gemma4TextAttention core (`scaling = 1.0`, float32 softmax, `bfloat16` matmul).
    `q`: `[B, H_q, S_q, D]`, `k`/`v`: `[B, H_kv, S_kv, D]`, `mask`: `[B, 1, S_q, S_kv]`.
    """
    k_rep = repeat_kv_jax(k, n_rep)
    v_rep = repeat_kv_jax(v, n_rep)
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k_rep) + mask
    probs = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(q.dtype)
    out = jnp.einsum("bhqk,bhkd->bhqd", probs, v_rep)
    B, H_q, S_q, D = out.shape
    return jnp.swapaxes(out, 1, 2).reshape(B, S_q, H_q * D)


def extract_layer_group_weights(
    layers: Sequence[torch.nn.Module],
    indices: Sequence[int],
    dev: Any,
    has_kv: bool,
    grid_shape: Optional[Tuple[int, ...]] = None,
) -> Dict[str, jax.Array]:
    """Stack weights for a homogeneous group of layers on CPU and transfer once to TPU."""
    fields = [
        ("input_norm_w", lambda m: m.input_layernorm.weight),
        ("post_attn_norm_w", lambda m: m.post_attention_layernorm.weight),
        ("pre_mlp_norm_w", lambda m: m.pre_feedforward_layernorm.weight),
        ("post_mlp_norm_w", lambda m: m.post_feedforward_layernorm.weight),
        ("post_ple_norm_w", lambda m: m.post_per_layer_input_norm.weight),
        ("q_proj_w", lambda m: m.self_attn.q_proj.weight),
        ("q_norm_w", lambda m: m.self_attn.q_norm.weight),
        ("o_proj_w", lambda m: m.self_attn.o_proj.weight),
        ("gate_proj_w", lambda m: m.mlp.gate_proj.weight),
        ("up_proj_w", lambda m: m.mlp.up_proj.weight),
        ("down_proj_w", lambda m: m.mlp.down_proj.weight),
        ("ple_gate_w", lambda m: m.per_layer_input_gate.weight),
        ("ple_proj_w", lambda m: m.per_layer_projection.weight),
        ("layer_scalar", lambda m: m.layer_scalar.reshape(())),
    ]
    if has_kv:
        fields.extend([
            ("k_proj_w", lambda m: m.self_attn.k_proj.weight),
            ("v_proj_w", lambda m: m.self_attn.v_proj.weight),
            ("k_norm_w", lambda m: m.self_attn.k_norm.weight),
        ])

    out: Dict[str, jax.Array] = {}
    for name, getter in fields:
        stacked_pt = torch.stack([getter(layers[i]).detach().cpu() for i in indices], dim=0)
        if grid_shape is not None:
            stacked_pt = stacked_pt.reshape(*grid_shape, *stacked_pt.shape[1:])
        out[name] = torch_bf16_to_jax(stacked_pt, dev)
    return out


class JaxStaticGemma4Engine:
    """
    Pure JAX/XLA Static-Graph Engine for Gemma 4 E2B-it on Cloud TPU v5e-1.
    """

    def __init__(self, pt_cond_model: torch.nn.Module, tokenizer: Any, jax_device: Optional[Any] = None):
        self.tokenizer = tokenizer
        self.dev = jax_device or jax.devices()[0]
        text_model = pt_cond_model.model.language_model
        cfg = text_model.config
        self.cfg = cfg

        self.hidden_size = int(cfg.hidden_size)                        # 1536
        self.ple_dim = int(cfg.hidden_size_per_layer_input)            # 256
        self.num_layers = int(cfg.num_hidden_layers)                   # 35
        self.num_kv_shared = int(cfg.num_kv_shared_layers)             # 20
        self.first_kv_shared = self.num_layers - self.num_kv_shared    # 15
        self.num_q_heads = int(cfg.num_attention_heads)                # 8
        self.slid_n_kv = int(cfg.per_layer_config[0].num_key_value_heads)
        self.slid_head_dim = int(cfg.per_layer_config[0].head_dim)
        self.slid_n_rep = self.num_q_heads // self.slid_n_kv
        self.glob_n_kv = int(cfg.per_layer_config[4].num_key_value_heads)
        self.glob_head_dim = int(cfg.per_layer_config[4].head_dim)
        self.glob_n_rep = self.num_q_heads // self.glob_n_kv
        self.eps = float(cfg.rms_norm_eps)                             # 1e-6
        self.sliding_window = int(cfg.sliding_window)                  # 512
        self.pad_token_id = int(getattr(cfg, "pad_token_id", 0) or 0)

        # Collect stop token IDs
        stop_ids = set()
        eos_cfg = getattr(cfg, "eos_token_id", None)
        if isinstance(eos_cfg, int):
            stop_ids.add(eos_cfg)
        elif isinstance(eos_cfg, (list, tuple)):
            stop_ids.update(int(x) for x in eos_cfg)
        for tok_str in ["<eos>", "<turn|>", "<end_of_turn>"]:
            tid = tokenizer.convert_tokens_to_ids(tok_str)
            if isinstance(tid, int) and tid >= 0 and tid != tokenizer.unk_token_id:
                stop_ids.add(tid)
        self.stop_ids_list = sorted(stop_ids)
        self.stop_ids_jax = jax.device_put(jnp.array(self.stop_ids_list, dtype=jnp.int32), self.dev)

        # Global & Top-level Weights on TPU
        self.embed_w = torch_bf16_to_jax(text_model.embed_tokens.weight, self.dev)  # [262144, 1536] (tied with lm_head)
        self.embed_scale = jax.device_put(
            jnp.asarray(float(text_model.embed_tokens.embed_scale.to(torch.bfloat16).item()), dtype=jnp.bfloat16),
            self.dev,
        )
        self.ple_embed_w = torch_bf16_to_jax(text_model.embed_tokens_per_layer.weight, self.dev)  # [262144, 8960]
        self.ple_embed_scale = jax.device_put(
            jnp.asarray(float(text_model.embed_tokens_per_layer.embed_scale.to(torch.bfloat16).item()), dtype=jnp.bfloat16),
            self.dev,
        )
        self.ple_model_proj_w = torch_bf16_to_jax(text_model.per_layer_model_projection.weight, self.dev)  # [8960, 1536]
        self.ple_model_proj_scale = jax.device_put(
            jnp.asarray(float(text_model.per_layer_model_projection_scale), dtype=jnp.bfloat16),
            self.dev,
        )
        self.ple_proj_norm_w = torch_bf16_to_jax(text_model.per_layer_projection_norm.weight, self.dev)  # [256]
        self.ple_input_scale = jax.device_put(
            jnp.asarray(float(text_model.per_layer_input_scale), dtype=jnp.bfloat16),
            self.dev,
        )
        self.final_norm_w = torch_bf16_to_jax(text_model.norm.weight, self.dev)  # [1536]

        # RoPE inverse frequencies
        rot = text_model.rotary_emb
        self.inv_freq_slid = torch_bf16_to_jax(rot.sliding_attention_inv_freq.float(), self.dev)  # [128] float32
        self.inv_freq_glob = torch_bf16_to_jax(rot.full_attention_inv_freq.float(), self.dev)     # [256] float32

        # Group the 35 layers into 4 homogeneous stacked tensor groups:
        z1_slid_idx = [l for l in range(15) if cfg.layer_types[l] == "sliding_attention"]
        z1_glob_idx = [l for l in range(15) if cfg.layer_types[l] == "full_attention"]
        z2_slid_idx = [l for l in range(15, 35) if cfg.layer_types[l] == "sliding_attention"]
        z2_glob_idx = [l for l in range(15, 35) if cfg.layer_types[l] == "full_attention"]

        self.z1_slid = extract_layer_group_weights(text_model.layers, z1_slid_idx, self.dev, has_kv=True, grid_shape=(3, 4))
        self.z1_glob = extract_layer_group_weights(text_model.layers, z1_glob_idx, self.dev, has_kv=True)
        self.z2_slid = extract_layer_group_weights(text_model.layers, z2_slid_idx, self.dev, has_kv=False, grid_shape=(4, 4))
        self.z2_glob = extract_layer_group_weights(text_model.layers, z2_glob_idx, self.dev, has_kv=False)

        self._orig_z2_slid_scalar = self.z2_slid["layer_scalar"]
        self._orig_z2_glob_scalar = self.z2_glob["layer_scalar"]
        self._orig_z2_slid_ple_proj = self.z2_slid["ple_proj_w"]
        self._orig_z2_glob_ple_proj = self.z2_glob["ple_proj_w"]

        self.z2_slid_active = jax.device_put(jnp.ones((4, 4), dtype=jnp.bool_), self.dev)
        self.z2_glob_active = jax.device_put(jnp.ones((4,), dtype=jnp.bool_), self.dev)

        self._compile_kernels()

    def restore_defaults(self) -> None:
        """Restore unmodified Zone-2 scalars, PLE projections, and active layer masks (0 recompiles!)."""
        self.z2_slid["layer_scalar"] = self._orig_z2_slid_scalar
        self.z2_glob["layer_scalar"] = self._orig_z2_glob_scalar
        self.z2_slid["ple_proj_w"] = self._orig_z2_slid_ple_proj
        self.z2_glob["ple_proj_w"] = self._orig_z2_glob_ple_proj
        self.z2_slid_active = jax.device_put(jnp.ones((4, 4), dtype=jnp.bool_), self.dev)
        self.z2_glob_active = jax.device_put(jnp.ones((4,), dtype=jnp.bool_), self.dev)

    def apply_layer_bypass(self, bypass_layers: Sequence[int], compensate_scalar: bool = True) -> None:
        """
        Configure dormant KV-shared layer bypass (`l >= 15`) with optional `layer_scalar` grafting.
        Triggers ZERO XLA recompiles because `z2_slid_active` and `layer_scalar` are dynamic array inputs!
        """
        self.restore_defaults()
        bypass_set = sorted(set(bypass_layers))
        slid_sc = np.array(self._orig_z2_slid_scalar.astype(jnp.float32), dtype=np.float32)  # [4, 4]
        glob_sc = np.array(self._orig_z2_glob_scalar.astype(jnp.float32), dtype=np.float32)  # [4]
        slid_act = np.ones((4, 4), dtype=bool)
        glob_act = np.ones((4,), dtype=bool)

        def get_sc(l_idx: int) -> float:
            b = (l_idx - 15) // 5
            rem = (l_idx - 15) % 5
            return float(glob_sc[b]) if rem == 4 else float(slid_sc[b, rem])

        def mult_sc(l_idx: int, factor: float) -> None:
            b = (l_idx - 15) // 5
            rem = (l_idx - 15) % 5
            if rem == 4:
                glob_sc[b] *= factor
            else:
                slid_sc[b, rem] *= factor

        if compensate_scalar:
            for l in bypass_set:
                lam = get_sc(l)
                prev = l - 1
                while prev in bypass_set and prev >= 15:
                    prev -= 1
                mult_sc(prev, lam)

        for l in bypass_set:
            b = (l - 15) // 5
            rem = (l - 15) % 5
            if rem == 4:
                glob_act[b] = False
            else:
                slid_act[b, rem] = False

        self.z2_slid["layer_scalar"] = jax.device_put(jnp.asarray(slid_sc, dtype=jnp.bfloat16), self.dev)
        self.z2_glob["layer_scalar"] = jax.device_put(jnp.asarray(glob_sc, dtype=jnp.bfloat16), self.dev)
        self.z2_slid_active = jax.device_put(jnp.asarray(slid_act, dtype=jnp.bool_), self.dev)
        self.z2_glob_active = jax.device_put(jnp.asarray(glob_act, dtype=jnp.bool_), self.dev)

    def apply_s1d_prose_subspace_surgery(self, prose_basis_pt: torch.Tensor) -> None:
        """
        Project out rank-8 conversational prose-filler basis `Q` (`[1536, 8]`) from `L25..L34`
        `per_layer_projection.weight` (`b = 2, 3` in Zone 2) on TPU with zero XLA recompilation!
        """
        q_jax = torch_bf16_to_jax(prose_basis_pt.float(), self.dev)  # [1536, 8] float32

        def proj_out(w_bf16: jax.Array) -> jax.Array:
            w_f32 = w_bf16.astype(jnp.float32)
            proj = jnp.einsum("dr,...dp->...rp", q_jax, w_f32)
            w_new = w_f32 - jnp.einsum("dr,...rp->...dp", q_jax, proj)
            return w_new.astype(jnp.bfloat16)

        slid_w = self._orig_z2_slid_ple_proj
        glob_w = self._orig_z2_glob_ple_proj
        self.z2_slid["ple_proj_w"] = slid_w.at[2:].set(proj_out(slid_w[2:]))
        self.z2_glob["ple_proj_w"] = glob_w.at[2:].set(proj_out(glob_w[2:]))

    def _compile_kernels(self) -> None:
        eps = self.eps

        def compute_embeddings_and_ple(
            input_ids: jax.Array,
            embed_w: jax.Array,
            embed_scale: jax.Array,
            ple_embed_w: jax.Array,
            ple_embed_scale: jax.Array,
            ple_model_proj_w: jax.Array,
            ple_model_proj_scale: jax.Array,
            ple_proj_norm_w: jax.Array,
            ple_input_scale: jax.Array,
        ) -> Tuple[jax.Array, jax.Array]:
            B, S = input_ids.shape
            inputs_embeds = jnp.take(embed_w, input_ids, axis=0) * embed_scale  # [B, S, 1536]
            ple_tok = jnp.take(ple_embed_w, input_ids, axis=0) * ple_embed_scale
            ple_tok = ple_tok.reshape(B, S, 35, 256)
            ple_proj = jnp.einsum("bsd,pd->bsp", inputs_embeds, ple_model_proj_w) * ple_model_proj_scale
            ple_proj = ple_proj.reshape(B, S, 35, 256)
            ple_proj = rms_norm(ple_proj, ple_proj_norm_w, eps)
            per_layer_inputs = (ple_proj + ple_tok) * ple_input_scale  # [B, S, 35, 256]
            return inputs_embeds, per_layer_inputs

        def compute_rope(pos_ids: jax.Array, inv_freq: jax.Array) -> Tuple[jax.Array, jax.Array]:
            freqs = pos_ids[..., None].astype(jnp.float32) * inv_freq[None, None, :]
            emb = jnp.concatenate([freqs, freqs], axis=-1)
            return jnp.cos(emb).astype(jnp.bfloat16), jnp.sin(emb).astype(jnp.bfloat16)

        def post_attn_mlp_ple(
            h: jax.Array,
            attn_out: jax.Array,
            ple_in: jax.Array,
            w: Dict[str, jax.Array],
        ) -> jax.Array:
            h = h + rms_norm(attn_out, w["post_attn_norm_w"], eps)
            r = h
            h_norm = rms_norm(h, w["pre_mlp_norm_w"], eps)
            gate = jax.nn.gelu(jnp.einsum("bsd,fd->bsf", h_norm, w["gate_proj_w"]), approximate=True)
            up = jnp.einsum("bsd,fd->bsf", h_norm, w["up_proj_w"])
            mlp_out = jnp.einsum("bsf,df->bsd", gate * up, w["down_proj_w"])
            h = r + rms_norm(mlp_out, w["post_mlp_norm_w"], eps)

            r_ple = h
            ple_g = jax.nn.gelu(jnp.einsum("bsd,pd->bsp", h, w["ple_gate_w"]), approximate=True)
            ple_u = jnp.einsum("bsp,dp->bsd", ple_g * ple_in, w["ple_proj_w"])
            h = r_ple + rms_norm(ple_u, w["post_ple_norm_w"], eps)
            return h * w["layer_scalar"]

        def run_active_layer(
            h: jax.Array,
            ple_in: jax.Array,
            cos: jax.Array,
            sin: jax.Array,
            mask: jax.Array,
            k_slot: jax.Array,
            v_slot: jax.Array,
            write_pos: jax.Array,
            w: Dict[str, jax.Array],
            head_dim: int,
            n_kv: int,
            n_rep: int,
            is_prefill: bool,
        ) -> Tuple[jax.Array, jax.Array, jax.Array]:
            B, S, _ = h.shape
            h_norm = rms_norm(h, w["input_norm_w"], eps)
            q = jnp.einsum("bsd,hd->bsh", h_norm, w["q_proj_w"]).reshape(B, S, self.num_q_heads, head_dim)
            k = jnp.einsum("bsd,hd->bsh", h_norm, w["k_proj_w"]).reshape(B, S, n_kv, head_dim)
            v = jnp.einsum("bsd,hd->bsh", h_norm, w["v_proj_w"]).reshape(B, S, n_kv, head_dim)

            q = apply_rope(rms_norm(q, w["q_norm_w"], eps), cos, sin)
            k = apply_rope(rms_norm(k, w["k_norm_w"], eps), cos, sin)
            v = rms_norm(v, None, eps)

            q_t = jnp.swapaxes(q, 1, 2)  # [B, 8, S, D]
            k_t = jnp.swapaxes(k, 1, 2)  # [B, n_kv, S, D]
            v_t = jnp.swapaxes(v, 1, 2)  # [B, n_kv, S, D]

            k_slot_new = jax.lax.dynamic_update_slice(k_slot, k_t, (0, 0, write_pos, 0))
            v_slot_new = jax.lax.dynamic_update_slice(v_slot, v_t, (0, 0, write_pos, 0))

            k_attn = k_t if is_prefill else k_slot_new
            v_attn = v_t if is_prefill else v_slot_new

            attn_vec = attention_core(q_t, k_attn, v_attn, mask, n_rep=n_rep)
            attn_out = jnp.einsum("bsd,od->bso", attn_vec, w["o_proj_w"])
            h_next = post_attn_mlp_ple(h, attn_out, ple_in, w)
            return h_next, k_slot_new, v_slot_new

        def run_shared_layer(
            h: jax.Array,
            ple_in: jax.Array,
            cos: jax.Array,
            sin: jax.Array,
            mask: jax.Array,
            k_shared: jax.Array,
            v_shared: jax.Array,
            w: Dict[str, jax.Array],
            is_active: jax.Array,
            head_dim: int,
            n_rep: int,
        ) -> jax.Array:
            def _do_layer(h_in: jax.Array) -> jax.Array:
                B, S, _ = h_in.shape
                h_norm = rms_norm(h_in, w["input_norm_w"], eps)
                q = jnp.einsum("bsd,hd->bsh", h_norm, w["q_proj_w"]).reshape(B, S, self.num_q_heads, head_dim)
                q = apply_rope(rms_norm(q, w["q_norm_w"], eps), cos, sin)
                q_t = jnp.swapaxes(q, 1, 2)
                attn_vec = attention_core(q_t, k_shared, v_shared, mask, n_rep=n_rep)
                attn_out = jnp.einsum("bsd,od->bso", attn_vec, w["o_proj_w"])
                return post_attn_mlp_ple(h_in, attn_out, ple_in, w)

            return jax.lax.cond(is_active, _do_layer, lambda h_in: h_in, h)

        def forward_35_layers(
            input_ids: jax.Array,
            pos_ids: jax.Array,
            mask_slid: jax.Array,
            mask_glob: jax.Array,
            write_pos: jax.Array,
            k_slid_cache: jax.Array,
            v_slid_cache: jax.Array,
            k_glob_cache: jax.Array,
            v_glob_cache: jax.Array,
            params: Dict[str, Any],
            is_prefill: bool,
        ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            h, ple_all = compute_embeddings_and_ple(
                input_ids,
                params["embed_w"],
                params["embed_scale"],
                params["ple_embed_w"],
                params["ple_embed_scale"],
                params["ple_model_proj_w"],
                params["ple_model_proj_scale"],
                params["ple_proj_norm_w"],
                params["ple_input_scale"],
            )
            cos_slid, sin_slid = compute_rope(pos_ids, params["inv_freq_slid"])
            cos_glob, sin_glob = compute_rope(pos_ids, params["inv_freq_glob"])

            ple_t = jnp.moveaxis(ple_all, 2, 0)  # [35, B, S, 256]
            ple_z1 = ple_t[:15].reshape(3, 5, *ple_t.shape[1:])
            ple_z2 = ple_t[15:].reshape(4, 5, *ple_t.shape[1:])

            k_slid_3d = k_slid_cache.reshape(3, 4, *k_slid_cache.shape[1:])
            v_slid_3d = v_slid_cache.reshape(3, 4, *v_slid_cache.shape[1:])

            def z1_block_fn(h_curr, block_inp):
                w_slid_b, w_glob_b, ple_b, k_sb, v_sb, k_gb, v_gb = block_inp

                def z1_slid_fn(h_s, s_inp):
                    w_s, ple_s, ks, vs = s_inp
                    h_next, ks_new, vs_new = run_active_layer(
                        h_s, ple_s, cos_slid, sin_slid, mask_slid, ks, vs, write_pos,
                        w_s, head_dim=self.slid_head_dim, n_kv=self.slid_n_kv, n_rep=self.slid_n_rep, is_prefill=is_prefill
                    )
                    return h_next, (ks_new, vs_new)

                h_after_slid, (ks_out, vs_out) = jax.lax.scan(
                    z1_slid_fn, h_curr, (w_slid_b, ple_b[:4], k_sb, v_sb)
                )
                h_after_glob, kg_out, vg_out = run_active_layer(
                    h_after_slid, ple_b[4], cos_glob, sin_glob, mask_glob, k_gb, v_gb, write_pos,
                    w_glob_b, head_dim=self.glob_head_dim, n_kv=self.glob_n_kv, n_rep=self.glob_n_rep, is_prefill=is_prefill
                )
                return h_after_glob, (ks_out, vs_out, kg_out, vg_out)

            h, (k_slid_out, v_slid_out, k_glob_out, v_glob_out) = jax.lax.scan(
                z1_block_fn,
                h,
                (params["z1_slid"], params["z1_glob"], ple_z1, k_slid_3d, v_slid_3d, k_glob_cache, v_glob_cache),
            )
            k_slid_new = k_slid_out.reshape(12, *k_slid_cache.shape[1:])
            v_slid_new = v_slid_out.reshape(12, *k_slid_cache.shape[1:])
            k_glob_new = k_glob_out
            v_glob_new = v_glob_out

            P = input_ids.shape[1]
            k_anchor_slid = k_slid_new[11, :, :, :P, :] if is_prefill else k_slid_new[11]
            v_anchor_slid = v_slid_new[11, :, :, :P, :] if is_prefill else v_slid_new[11]
            k_anchor_glob = k_glob_new[2, :, :, :P, :] if is_prefill else k_glob_new[2]
            v_anchor_glob = v_glob_new[2, :, :, :P, :] if is_prefill else v_glob_new[2]

            def z2_block_fn(h_curr, block_inp):
                w_slid_b, w_glob_b, act_slid_b, act_glob_b, ple_b = block_inp

                def z2_slid_fn(h_s, s_inp):
                    w_s, act_s, ple_s = s_inp
                    h_next = run_shared_layer(
                        h_s, ple_s, cos_slid, sin_slid, mask_slid, k_anchor_slid, v_anchor_slid,
                        w_s, act_s, head_dim=self.slid_head_dim, n_rep=self.slid_n_rep
                    )
                    return h_next, None

                h_after_slid, _ = jax.lax.scan(
                    z2_slid_fn, h_curr, (w_slid_b, act_slid_b, ple_b[:4])
                )
                h_after_glob = run_shared_layer(
                    h_after_slid, ple_b[4], cos_glob, sin_glob, mask_glob, k_anchor_glob, v_anchor_glob,
                    w_glob_b, act_glob_b, head_dim=self.glob_head_dim, n_rep=self.glob_n_rep
                )
                return h_after_glob, None

            h, _ = jax.lax.scan(
                z2_block_fn,
                h,
                (params["z2_slid"], params["z2_glob"], params["z2_slid_active"], params["z2_glob_active"], ple_z2),
            )

            h_final = rms_norm(h, params["final_norm_w"], eps)
            last_h = h_final[:, -1:, :]
            logits = jnp.einsum("bsd,vd->bsv", last_h, params["embed_w"])
            next_tok = jnp.argmax(logits, axis=-1).astype(jnp.int32)
            return next_tok, k_slid_new, v_slid_new, k_glob_new, v_glob_new

        def _prefill_impl(
            input_ids: jax.Array,
            valid_start: jax.Array,
            prompt_lens: jax.Array,
            k_slid_cache: jax.Array,
            v_slid_cache: jax.Array,
            k_glob_cache: jax.Array,
            v_glob_cache: jax.Array,
            params: Dict[str, Any],
        ):
            B, P = input_ids.shape
            arange_p = jnp.arange(P, dtype=jnp.int32)
            q_idx = arange_p[None, None, :, None]
            k_idx = arange_p[None, None, None, :]
            vs_4d = valid_start[:, None, None, None]

            is_pad_q = q_idx < vs_4d
            pad_self = is_pad_q & (k_idx == q_idx)
            real_causal = (~is_pad_q) & (k_idx >= vs_4d) & (k_idx <= q_idx)

            allow_full = pad_self | real_causal
            allow_slid = pad_self | (real_causal & ((q_idx - k_idx) < self.sliding_window))

            zero_bf16 = jnp.asarray(0.0, dtype=jnp.bfloat16)
            neg_inf = jnp.asarray(-3.38953139e38, dtype=jnp.bfloat16)
            mask_glob = jnp.where(allow_full, zero_bf16, neg_inf)
            mask_slid = jnp.where(allow_slid, zero_bf16, neg_inf)

            pos_ids = jnp.maximum(arange_p[None, :] - valid_start[:, None], 0)
            write_pos = jnp.asarray(0, dtype=jnp.int32)

            return forward_35_layers(
                input_ids, pos_ids, mask_slid, mask_glob, write_pos,
                k_slid_cache, v_slid_cache, k_glob_cache, v_glob_cache,
                params, is_prefill=True
            )

        def _decode_step_impl(
            next_tok: jax.Array,
            cur_pos: jax.Array,
            step_offset: jax.Array,
            k_slid_cache: jax.Array,
            v_slid_cache: jax.Array,
            k_glob_cache: jax.Array,
            v_glob_cache: jax.Array,
            valid_start: jax.Array,
            prompt_lens: jax.Array,
            params: Dict[str, Any],
        ):
            M = k_slid_cache.shape[3]
            k_idx_m = jnp.arange(M, dtype=jnp.int32)[None, None, None, :]
            vs_4d = valid_start[:, None, None, None]
            cp_4d = cur_pos.reshape(1, 1, 1, 1)

            allow_full = (k_idx_m >= vs_4d) & (k_idx_m <= cp_4d)
            allow_slid = allow_full & ((cp_4d - k_idx_m) < self.sliding_window)

            zero_bf16 = jnp.asarray(0.0, dtype=jnp.bfloat16)
            neg_inf = jnp.asarray(-3.38953139e38, dtype=jnp.bfloat16)
            mask_glob = jnp.where(allow_full, zero_bf16, neg_inf)
            mask_slid = jnp.where(allow_slid, zero_bf16, neg_inf)

            pos_ids = prompt_lens[:, None] + step_offset.reshape(1, 1)
            return forward_35_layers(
                next_tok, pos_ids, mask_slid, mask_glob, cur_pos,
                k_slid_cache, v_slid_cache, k_glob_cache, v_glob_cache,
                params, is_prefill=False
            )

        @partial(jax.jit, static_argnums=(3,))
        def jit_prefill_and_decode_scan(
            input_ids: jax.Array,
            valid_start: jax.Array,
            prompt_lens: jax.Array,
            max_new_tokens: int,
            stop_ids_dev: jax.Array,
            params: Dict[str, Any],
        ):
            B, P = input_ids.shape
            G = max_new_tokens
            M = P + G

            k_slid_init = jnp.zeros((12, B, self.slid_n_kv, M, self.slid_head_dim), dtype=jnp.bfloat16)
            v_slid_init = jnp.zeros((12, B, self.slid_n_kv, M, self.slid_head_dim), dtype=jnp.bfloat16)
            k_glob_init = jnp.zeros((3, B, self.glob_n_kv, M, self.glob_head_dim), dtype=jnp.bfloat16)
            v_glob_init = jnp.zeros((3, B, self.glob_n_kv, M, self.glob_head_dim), dtype=jnp.bfloat16)

            first_tok, ks, vs, kg, vg = _prefill_impl(
                input_ids, valid_start, prompt_lens, k_slid_init, v_slid_init, k_glob_init, v_glob_init, params
            )
            finished_init = jnp.isin(first_tok, stop_ids_dev)
            step_positions = jnp.arange(P, P + G - 1, dtype=jnp.int32)
            step_offsets = jnp.arange(0, G - 1, dtype=jnp.int32)
            pad_tok_vec = jnp.full((B, 1), self.pad_token_id, dtype=jnp.int32)

            def scan_body(carry, step_inp):
                cur_tok, fin, k_s, v_s, k_g, v_g = carry
                cur_pos, step_off = step_inp
                cand_tok, k_s_new, v_s_new, k_g_new, v_g_new = _decode_step_impl(
                    cur_tok, cur_pos, step_off, k_s, v_s, k_g, v_g, valid_start, prompt_lens, params
                )
                next_t = jnp.where(fin, pad_tok_vec, cand_tok)
                fin_new = fin | jnp.isin(next_t, stop_ids_dev)
                return (next_t, fin_new, k_s_new, v_s_new, k_g_new, v_g_new), next_t

            _, gen_seq = jax.lax.scan(
                scan_body,
                (first_tok, finished_init, ks, vs, kg, vg),
                (step_positions, step_offsets),
            )
            gen_tokens = jnp.concatenate([first_tok, jnp.swapaxes(gen_seq.squeeze(-1), 0, 1)], axis=1)
            return gen_tokens

        self._jit_prefill = partial(jax.jit, donate_argnums=(3, 4, 5, 6))(_prefill_impl)
        self._jit_decode_step = partial(jax.jit, donate_argnums=(3, 4, 5, 6))(_decode_step_impl)
        self._jit_prefill_and_decode_scan = jit_prefill_and_decode_scan

    def _get_params_dict(self) -> Dict[str, Any]:
        return {
            "embed_w": self.embed_w,
            "embed_scale": self.embed_scale,
            "ple_embed_w": self.ple_embed_w,
            "ple_embed_scale": self.ple_embed_scale,
            "ple_model_proj_w": self.ple_model_proj_w,
            "ple_model_proj_scale": self.ple_model_proj_scale,
            "ple_proj_norm_w": self.ple_proj_norm_w,
            "ple_input_scale": self.ple_input_scale,
            "final_norm_w": self.final_norm_w,
            "inv_freq_slid": self.inv_freq_slid,
            "inv_freq_glob": self.inv_freq_glob,
            "z1_slid": self.z1_slid,
            "z1_glob": self.z1_glob,
            "z2_slid": self.z2_slid,
            "z2_glob": self.z2_glob,
            "z2_slid_active": self.z2_slid_active,
            "z2_glob_active": self.z2_glob_active,
        }

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
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
            )
            formatted.append(txt)
        return formatted

    def _allocate_kv_caches(self, B: int, M: int) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        k_slid = jax.device_put(jnp.zeros((12, B, self.slid_n_kv, M, self.slid_head_dim), dtype=jnp.bfloat16), self.dev)
        v_slid = jax.device_put(jnp.zeros((12, B, self.slid_n_kv, M, self.slid_head_dim), dtype=jnp.bfloat16), self.dev)
        k_glob = jax.device_put(jnp.zeros((3, B, self.glob_n_kv, M, self.glob_head_dim), dtype=jnp.bfloat16), self.dev)
        v_glob = jax.device_put(jnp.zeros((3, B, self.glob_n_kv, M, self.glob_head_dim), dtype=jnp.bfloat16), self.dev)
        return k_slid, v_slid, k_glob, v_glob

    def generate_batch_jax_scan(
        self,
        formatted_prompts: Sequence[str],
        max_new_tokens: int = 256,
        prompt_pad_bucket: int = 128,
    ) -> Dict[str, Any]:
        """
        Optimal Whole-Loop Compiled JAX Generation (Fused Prefill + `jax.lax.scan` Decode).
        Executes prefill and all decode steps inside a SINGLE XLA launch with zero host-loop overhead.
        """
        B = len(formatted_prompts)
        enc = [self.tokenizer.encode(t, add_special_tokens=False) for t in formatted_prompts]
        max_p_actual = max(len(ids) for ids in enc)
        P = max(prompt_pad_bucket, ((max_p_actual + prompt_pad_bucket - 1) // prompt_pad_bucket) * prompt_pad_bucket)
        G = max_new_tokens

        input_np = np.full((B, P), self.pad_token_id, dtype=np.int32)
        lens_np = np.zeros((B,), dtype=np.int32)
        for i, ids in enumerate(enc):
            if len(ids) > P:
                ids = ids[-P:]
            L = len(ids)
            input_np[i, P - L : P] = np.asarray(ids, dtype=np.int32)
            lens_np[i] = L

        params = self._get_params_dict()

        t_start = time.perf_counter()
        input_ids = jax.device_put(jnp.asarray(input_np), self.dev)
        prompt_lens = jax.device_put(jnp.asarray(lens_np), self.dev)
        valid_start = P - prompt_lens

        gen_tokens = self._jit_prefill_and_decode_scan(
            input_ids, valid_start, prompt_lens, G, self.stop_ids_jax, params
        )
        gen_tokens.block_until_ready()
        elapsed = time.perf_counter() - t_start

        gen_cpu = np.asarray(gen_tokens).tolist()
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
            "steps_executed": G,
            "batch_size": B,
            "prompt_pad_len": P,
            "elapsed_sec": round(elapsed, 4),
            "effective_tok_per_sec": round(total_gen_toks / max(elapsed, 1e-6), 2),
            "raw_step_tok_per_sec": round((B * G) / max(elapsed, 1e-6), 2),
        }

    def generate_batch_jax_step_loop(
        self,
        formatted_prompts: Sequence[str],
        max_new_tokens: int = 97,
        prompt_pad_bucket: int = 128,
    ) -> Dict[str, Any]:
        """
        Python-loop step-by-step `@jax.jit` with `dynamic_update_slice` (direct 1-to-1 JAX analogue
        to PyTorch's `StaticGemma4Engine` before fusing the decode loop with `jax.lax.scan`).
        """
        B = len(formatted_prompts)
        enc = [self.tokenizer.encode(t, add_special_tokens=False) for t in formatted_prompts]
        max_p_actual = max(len(ids) for ids in enc)
        P = max(prompt_pad_bucket, ((max_p_actual + prompt_pad_bucket - 1) // prompt_pad_bucket) * prompt_pad_bucket)
        G = max_new_tokens
        M = P + G

        input_np = np.full((B, P), self.pad_token_id, dtype=np.int32)
        lens_np = np.zeros((B,), dtype=np.int32)
        for i, ids in enumerate(enc):
            if len(ids) > P:
                ids = ids[-P:]
            L = len(ids)
            input_np[i, P - L : P] = np.asarray(ids, dtype=np.int32)
            lens_np[i] = L

        k_slid, v_slid, k_glob, v_glob = self._allocate_kv_caches(B, M)
        params = self._get_params_dict()

        t_start = time.perf_counter()
        input_ids = jax.device_put(jnp.asarray(input_np), self.dev)
        prompt_lens = jax.device_put(jnp.asarray(lens_np), self.dev)
        valid_start = P - prompt_lens

        next_tok, k_slid, v_slid, k_glob, v_glob = self._jit_prefill(
            input_ids, valid_start, prompt_lens, k_slid, v_slid, k_glob, v_glob, params
        )
        step_positions = jax.device_put(jnp.arange(P, P + G - 1, dtype=jnp.int32), self.dev)
        step_offsets = jax.device_put(jnp.arange(0, G - 1, dtype=jnp.int32), self.dev)

        tok_list = [next_tok]
        for s in range(G - 1):
            next_tok, k_slid, v_slid, k_glob, v_glob = self._jit_decode_step(
                next_tok, step_positions[s], step_offsets[s],
                k_slid, v_slid, k_glob, v_glob, valid_start, prompt_lens, params
            )
            tok_list.append(next_tok)

        gen_tokens = jnp.concatenate(tok_list, axis=1)
        gen_tokens.block_until_ready()
        elapsed = time.perf_counter() - t_start

        gen_cpu = np.asarray(gen_tokens).tolist()
        texts = [self.tokenizer.decode(row, skip_special_tokens=False) for row in gen_cpu]
        return {
            "texts": texts,
            "steps_executed": G,
            "batch_size": B,
            "elapsed_sec": round(elapsed, 4),
            "raw_step_tok_per_sec": round((B * G) / max(elapsed, 1e-6), 2),
        }

    def generate_dataset(
        self,
        user_prompts: Sequence[str],
        batch_size: int = 8,
        max_new_tokens: int = 256,
        prompt_pad_bucket: int = 256,
        enable_thinking: bool = False,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates an entire dataset of prompts in uniform static batches using
        `generate_batch_jax_scan` (with right-padding stripped from tail batches).
        """
        formatted = self.format_chat_prompts(
            user_prompts, enable_thinking=enable_thinking, system_prompt=system_prompt
        )
        all_texts: List[str] = []
        all_counts: List[int] = []
        total_steps = 0
        total_elapsed = 0.0

        for start in range(0, len(formatted), batch_size):
            chunk = formatted[start : start + batch_size]
            actual_b = len(chunk)
            if actual_b < batch_size:
                chunk = list(chunk) + [chunk[-1]] * (batch_size - actual_b)

            res = self.generate_batch_jax_scan(
                chunk,
                max_new_tokens=max_new_tokens,
                prompt_pad_bucket=prompt_pad_bucket,
            )
            all_texts.extend(res["texts"][:actual_b])
            all_counts.extend(res["token_counts"][:actual_b])
            total_steps += res["steps_executed"]
            total_elapsed += res["elapsed_sec"]

        total_gen_toks = sum(all_counts)
        return {
            "texts": all_texts,
            "token_counts": all_counts,
            "total_generated_tokens": total_gen_toks,
            "total_steps": total_steps,
            "elapsed_sec": round(total_elapsed, 3),
            "effective_tok_per_sec": round(total_gen_toks / max(total_elapsed, 1e-6), 2),
            "mean_tokens_per_item": round(total_gen_toks / max(len(user_prompts), 1), 1),
        }

