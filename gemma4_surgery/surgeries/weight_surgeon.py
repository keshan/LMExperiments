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
Reversible, In-Place Weight & Structural Surgery Toolkit for Gemma 4.

Supports:
1. Bit-exact host-RAM backups and restoration for BOTH `nn.Parameter` and `nn.Buffer`
   (critical because Gemma 4's `layers.{l}.layer_scalar` is an `nn.Buffer`!).
2. PLE De-Clamping & Lexical Attenuation (`post_per_layer_input_norm`, `embed_tokens_per_layer`).
3. KV-Shared Query Inverse-Temperature & Context-Retrieval Gain Calibration
   (`self_attn.q_norm.weight`, `post_attention_layernorm.weight`).
4. Spectral SVD Rank-Filtering (`per_layer_projection`, `mlp.down_proj`, `self_attn.o_proj`).
5. Scale-Compensated KV-Shared Layer Bypass (`BypassDecoderLayer` + cumulative `layer_scalar`
   grafting), safely eliminating redundant upper Double-Wide MLP + shared-KV layers.
6. Rank-k Subspace Projection / Contrastive Direction Grafting directly into projection weights.
"""

from typing import Dict, Iterable, List, Optional
import torch
import torch.nn as nn


class BypassDecoderLayer(nn.Module):
    """Zero-FLOP pass-through replacement for redundant KV-shared Gemma 4 decoder layers."""

    def __init__(self, orig_layer: nn.Module):
        super().__init__()
        self.layer_idx = getattr(orig_layer, "layer_idx", -1)
        # Preserve attribute interface expected by Gemma4TextModel utilities
        self.self_attn = orig_layer.self_attn

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return hidden_states


class WeightSurgeon:
    """Manages bit-exact backups in host RAM and applies composable, reversible surgeries."""

    def __init__(self, text_model: nn.Module):
        self.text_model = text_model
        self.tensor_map: Dict[str, torch.Tensor] = {}
        for k, p in text_model.named_parameters():
            self.tensor_map[k] = p
        for k, b in text_model.named_buffers():
            self.tensor_map[k] = b

        self.backups: Dict[str, torch.Tensor] = {}
        self.layer_module_backups: Dict[int, nn.Module] = {}
        self.active_log: List[str] = []

    def _ensure_backup(self, name: str) -> None:
        if name not in self.backups:
            t = self.tensor_map[name]
            self.backups[name] = t.detach().to(device="cpu").clone()

    def restore_all(self) -> int:
        """Bit-exact restoration of all modified parameters, buffers, and bypassed layers."""
        # First restore any swapped decoder layer modules
        for idx, orig_mod in self.layer_module_backups.items():
            self.text_model.layers[idx] = orig_mod
        restored_layers = len(self.layer_module_backups)
        self.layer_module_backups.clear()

        # Refresh tensor_map in case modules were restored
        self.tensor_map.clear()
        for k, p in self.text_model.named_parameters():
            self.tensor_map[k] = p
        for k, b in self.text_model.named_buffers():
            self.tensor_map[k] = b

        count = 0
        with torch.no_grad():
            for name, orig_cpu in self.backups.items():
                t = self.tensor_map[name]
                t.data.copy_(orig_cpu.to(device=t.device, dtype=t.dtype))
                count += 1
        self.active_log.clear()
        return count + restored_layers

    # ------------------------------------------------------------------
    # S1. PLE Post-Norm Gain Schedule Surgery (Hypothesis H1)
    # ------------------------------------------------------------------
    def scale_ple_post_norm(self, layers: Iterable[int], scale: float | Dict[int, float]) -> None:
        """
        Rescale `layers.{l}.post_per_layer_input_norm.weight` (Gemma4RMSNorm uses direct `w` scaling)
        to attenuate clamped static Layer-0 PLE injections relative to contextual updates.
        """
        layers_list = list(layers)
        with torch.no_grad():
            for l in layers_list:
                alpha = scale[l] if isinstance(scale, dict) else float(scale)
                name = f"layers.{l}.post_per_layer_input_norm.weight"
                if name not in self.tensor_map:
                    continue
                self._ensure_backup(name)
                orig = self.backups[name].float()
                t = self.tensor_map[name]
                t.data.copy_((orig * alpha).to(device=t.device, dtype=t.dtype))
        self.active_log.append(f"scale_ple_post_norm(layers={layers_list}, scale={scale})")

    # ------------------------------------------------------------------
    # S2a. KV-Shared Query-Norm Inverse-Temperature Surgery (Hypothesis H2)
    # ------------------------------------------------------------------
    def scale_q_norm_temperature(
        self,
        layers: Iterable[int],
        temp_scale: float | Dict[int, float],
    ) -> None:
        """
        Modulate exact attention inverse-temperature in target layers via `self_attn.q_norm.weight`.
        Because Gemma4TextAttention sets `self.scaling = 1.0` and Gemma4RMSNorm multiplies directly
        by `weight`, scaling `q_norm.weight` by `tau` multiplies QK^T logits by exact factor `tau`.
        """
        layers_list = list(layers)
        with torch.no_grad():
            for l in layers_list:
                tau = temp_scale[l] if isinstance(temp_scale, dict) else float(temp_scale)
                name = f"layers.{l}.self_attn.q_norm.weight"
                if name not in self.tensor_map:
                    continue
                self._ensure_backup(name)
                orig = self.backups[name].float()
                t = self.tensor_map[name]
                t.data.copy_((orig * tau).to(device=t.device, dtype=t.dtype))
        self.active_log.append(f"scale_q_norm_temperature(layers={layers_list}, tau={temp_scale})")

    # ------------------------------------------------------------------
    # S2b. Contextual Attention Retrieval Gain Surgery
    # ------------------------------------------------------------------
    def scale_post_attn_norm(
        self,
        layers: Iterable[int],
        scale: float | Dict[int, float],
    ) -> None:
        """
        Modulate `layers.{l}.post_attention_layernorm.weight` to strengthen or rebalance
        contextual attention updates against static PLE / MLP updates in KV-shared layers.
        """
        layers_list = list(layers)
        with torch.no_grad():
            for l in layers_list:
                alpha = scale[l] if isinstance(scale, dict) else float(scale)
                name = f"layers.{l}.post_attention_layernorm.weight"
                if name not in self.tensor_map:
                    continue
                self._ensure_backup(name)
                orig = self.backups[name].float()
                t = self.tensor_map[name]
                t.data.copy_((orig * alpha).to(device=t.device, dtype=t.dtype))
        self.active_log.append(f"scale_post_attn_norm(layers={layers_list}, scale={scale})")

    # ------------------------------------------------------------------
    # S3. Spectral SVD Rank-Filtering Surgery (Hypothesis H3)
    # ------------------------------------------------------------------
    def svd_filter_matrix(
        self,
        layers: Iterable[int],
        module_suffix: str,
        keep_ratio: float = 0.5,
        energy_preserve: bool = True,
    ) -> None:
        """
        Truncate trailing singular values of `layers.{l}.{module_suffix}.weight`
        (e.g. `per_layer_projection`, `mlp.down_proj`, `self_attn.o_proj`) to top `keep_ratio`.
        """
        layers_list = list(layers)
        with torch.no_grad():
            for l in layers_list:
                name = f"layers.{l}.{module_suffix}.weight"
                if name not in self.tensor_map:
                    continue
                self._ensure_backup(name)
                w = self.backups[name].float()
                u, s, vh = torch.linalg.svd(w, full_matrices=False)
                k = max(1, int(round(len(s) * keep_ratio)))
                s_trunc = s.clone()
                s_trunc[k:] = 0.0
                if energy_preserve:
                    orig_norm = s.norm()
                    new_norm = s_trunc.norm()
                    if new_norm > 1e-8:
                        s_trunc = s_trunc * (orig_norm / new_norm)
                w_new = (u * s_trunc.unsqueeze(0)) @ vh
                t = self.tensor_map[name]
                t.data.copy_(w_new.to(device=t.device, dtype=t.dtype))
        self.active_log.append(
            f"svd_filter_matrix(layers={layers_list}, suffix={module_suffix}, keep_ratio={keep_ratio}, energy={energy_preserve})"
        )

    # ------------------------------------------------------------------
    # S4. Scale-Compensated KV-Shared Layer Bypass Surgery
    # ------------------------------------------------------------------
    def bypass_kv_shared_layers(
        self,
        bypass_layers: Iterable[int],
        compensate_scalar: bool = True,
    ) -> Dict[str, float]:
        """
        Bypass specified KV-shared layers (`l >= 15`) using `BypassDecoderLayer`.
        When `compensate_scalar=True`, folds each bypassed block's `layer_scalar` (`lambda_l`)
        multiplicatively into the nearest preceding non-bypassed layer's `layer_scalar` buffer
        so downstream residual stream norm matches the unpruned trajectory!
        """
        bypass_set = sorted(set(bypass_layers))
        for l in bypass_set:
            if l < 15:
                raise ValueError(f"Layer {l} is NOT a KV-shared layer (must be >= 15 to preserve KV anchors).")

        applied_scales: Dict[str, float] = {}
        with torch.no_grad():
            if compensate_scalar:
                # Accumulate scalar product onto nearest preceding active layer
                accum: Dict[int, float] = {}
                for l in bypass_set:
                    sc_name = f"layers.{l}.layer_scalar"
                    self._ensure_backup(sc_name)
                    lam = float(self.backups[sc_name].float().item())
                    prev = l - 1
                    while prev in bypass_set and prev >= 0:
                        prev -= 1
                    if prev < 0:
                        raise ValueError("Cannot bypass Layer 0.")
                    accum[prev] = accum.get(prev, 1.0) * lam

                for prev_l, mult in accum.items():
                    prev_name = f"layers.{prev_l}.layer_scalar"
                    self._ensure_backup(prev_name)
                    orig_prev = self.backups[prev_name].float()
                    t = self.tensor_map[prev_name]
                    t.data.copy_((orig_prev * mult).to(device=t.device, dtype=t.dtype))
                    applied_scales[f"layer_{prev_l}_mult"] = round(mult, 6)

            for l in bypass_set:
                if l not in self.layer_module_backups:
                    self.layer_module_backups[l] = self.text_model.layers[l]
                self.text_model.layers[l] = BypassDecoderLayer(self.layer_module_backups[l])

        self.active_log.append(
            f"bypass_kv_shared_layers(layers={bypass_set}, compensate_scalar={compensate_scalar}, scales={applied_scales})"
        )
        return applied_scales

    # ------------------------------------------------------------------
    # S5. Direct Layer Scalar Modulation
    # ------------------------------------------------------------------
    def modulate_layer_scalar(self, layer_scales: Dict[int, float]) -> None:
        with torch.no_grad():
            for l, mult in layer_scales.items():
                name = f"layers.{l}.layer_scalar"
                if name not in self.tensor_map:
                    continue
                self._ensure_backup(name)
                orig = self.backups[name].float()
                t = self.tensor_map[name]
                t.data.copy_((orig * mult).to(device=t.device, dtype=t.dtype))
        self.active_log.append(f"modulate_layer_scalar({layer_scales})")

    # ------------------------------------------------------------------
    # S6. Low-Rank Subspace Gain / Contrastive Projection Surgery
    # ------------------------------------------------------------------
    def graft_residual_subspace_gain(
        self,
        module_weight_name: str,
        basis_vectors: torch.Tensor,
        gain_delta: float,
    ) -> None:
        """
        Given orthonormal basis `V` of shape `[d_out, r]` (`V^T V = I_r`) in residual space,
        transforms output projection `W` (`[d_out, d_in]`) via:
            W_new = (I + gain_delta * V V^T) @ W = W + gain_delta * V @ (V^T @ W)
        For `gain_delta > 0`, amplifies components along `V`; for `gain_delta = -1`, projects out `V`!
        Zero runtime FLOP or parameter overhead.
        """
        if module_weight_name not in self.tensor_map:
            raise KeyError(f"{module_weight_name} not found in tensor_map")
        self._ensure_backup(module_weight_name)
        with torch.no_grad():
            w = self.backups[module_weight_name].float()
            v = basis_vectors.detach().cpu().float()
            if v.ndim == 1:
                v = v.unsqueeze(1)
            # Orthonormalize columns via QR
            q, _ = torch.linalg.qr(v)
            w_new = w + float(gain_delta) * (q @ (q.T @ w))
            t = self.tensor_map[module_weight_name]
            t.data.copy_(w_new.to(device=t.device, dtype=t.dtype))
        self.active_log.append(
            f"graft_residual_subspace_gain(target={module_weight_name}, rank={v.shape[1]}, gain_delta={gain_delta})"
        )
