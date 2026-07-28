# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
from typing import Callable, Optional, Union

import torch
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...integrations import use_kernel_forward_from_hub
from ...masking_utils import create_causal_mask
from ...modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.generic import check_model_inputs
from .configuration_llama import LlamaConfig

import numpy as np
from collections import deque
import os
import os as _os
import atexit
import random

logger = logging.get_logger(__name__)

# Configuration for adaptive sparsification (CATS: target 85% quality)
TARGET_QUALITY_ACTIVATION = 0.85
QUALITY_CHECK_PROB = 0.005

# --- contribution-metric signal, MIMD-cadence variant ------------------------
# Identical signal swap to the PID-contrib CATS file (modeling_acas_cats.py): we change
# ONLY the per-layer quality SIGNAL fed to the controller; the MIMD adaptive check-interval
# machinery (force_check + _mimd_interval) is left untouched, so it now adapts on contrib drift.
#   ACAS_CATS_SIGNAL : "activation" (paper L2 drift on activated gate, default) | "contrib"
#                      ("contrib" = ||y_dense - y_sparse|| at the MLP OUTPUT, normalized by ||residual||)
#   ACAS_CATS_TARGET : quality target for the chosen signal (default = TARGET_QUALITY_ACTIVATION)
CATS_SIGNAL = os.environ.get('ACAS_CATS_SIGNAL', 'activation').lower()
CATS_TARGET = float(os.environ.get('ACAS_CATS_TARGET', str(TARGET_QUALITY_ACTIVATION)))
# controller-swap ablation: 'pid' (default, byte-identical) | 'rl' (tabular Q-learning, same signal)
CATS_CONTROLLER = os.environ.get('ACAS_CATS_CONTROLLER', 'pid').lower()

# MIMD per-layer adaptive check interval (inspired by TCP congestion control)
# Multiplicative both ways — no shared resource, so MIMD fairness property unnecessary
# Asymmetric: fast decrease (react to drift), slower increase (probe cautiously)
MIMD_INCREASE = 1.5              # multiply interval by 1.5 when stable
MIMD_DECREASE = 0.5              # halve interval when drifting
MIMD_MIN_INTERVAL = int(os.environ.get("ACAS_NMIN", "50"))
MIMD_MAX_INTERVAL = int(os.environ.get("ACAS_NMAX", "500"))
MIMD_INIT_INTERVAL = 200
MIMD_ERROR_THRESHOLD = 0.01


class Sample:
    def __init__(self, error, step):
        self.error = error
        self.step = step


class BufferedPID:
    """PID controller with buffered samples for smooth control"""
    def __init__(self, kp, ki, kd, buffer_size, max_age):
        self._kp = kp
        self._ki = ki
        self._kd = kd
        self._buffer_size = buffer_size
        self._max_age = max_age
        self._buffer = deque()

    def add_sample(self, error, current_step):
        if len(self._buffer) >= self._buffer_size:
            self._buffer.popleft()
        self._buffer.append(Sample(error, current_step))

    def get_update(self, current_step):
        valid_samples = []
        for sample in self._buffer:
            if current_step - sample.step <= self._max_age:
                valid_samples.append(sample)

        if len(valid_samples) == 0:
            return 0.0, 0.0, 0.0, 0.0

        current_error = valid_samples[-1].error
        integral = sum(s.error for s in valid_samples)
        derivative = 0.0
        if len(valid_samples) >= 2:
            derivative = current_error - valid_samples[-2].error

        p_term = self._kp * current_error
        i_term = self._ki * integral
        d_term = self._kd * derivative
        return p_term + i_term + d_term, p_term, i_term, d_term


def calculate_error(current_value, target_value, sensitivity=5.0):
    """Calculate error for PID controller with logit transformation"""
    def logit(x):
        x = np.clip(x, 1e-10, 1 - 1e-10)
        return np.log(x / (1 - x))

    if current_value is None:
        current_value = target_value
    else:
        current_value = np.clip(current_value, 1e-10, 1 - 1e-10)
    target_value = np.clip(target_value, 1e-10, 1 - 1e-10)

    diff = logit(target_value) - logit(current_value)
    return np.sign(diff) * (2 / (1 + np.exp(-sensitivity * np.abs(diff))) - 1)


# === RL controller: tabular Q-learning drop-in for the PID threshold update. =========
# Enabled via ACAS_*_CONTROLLER=rl. Same information as the PID (quality vs target) + same knob;
# learns the proportional control law from reward = -|quality - target|. Validated within 0.3% of
# the PID setpoint (validated offline). Caller scales the returned push.
class RLThresholdController:
    _SIGMA_FALLBACK = 0.01   # used only until the running window has a few samples
    _SIGMA_WINDOW = 40       # checks for the ONLINE per-check noise estimate sigma_q (no hand-set scale)
    _ACTIONS = (-5.0, -0.7, -0.1, 0.0, 0.1, 0.7, 5.0)
    def __init__(self, target, alpha=0.3, gamma=0.85, eps0=0.30, eps_min=0.02, eps_decay_steps=200, seed=0):
        self.target = float(target); self.alpha = alpha; self.gamma = gamma
        self.eps0 = eps0; self.eps_min = eps_min; self.eps_decay_steps = eps_decay_steps
        self.Q = {}; self.t = 0; self._prev_s = None; self._prev_a = None
        self.rng = random.Random(seed); self.last_push = 0.0
        self._qhist = deque(maxlen=self._SIGMA_WINDOW)   # recent quality -> online sigma_q
    def _sigma(self):
        n = len(self._qhist)
        if n < 8: return self._SIGMA_FALLBACK
        m = sum(self._qhist) / n; var = sum((q - m) ** 2 for q in self._qhist) / n
        return max(var ** 0.5, 1e-4)
    def _state(self, e):
        sig = self._sigma(); s = 0          # bins at +/-{1,2,3}*sigma, sigma online
        for k in (-3, -2, -1, 1, 2, 3):
            if e > k * sig: s += 1
        return s
    def _eps(self):
        frac = min(1.0, self.t / max(1, self.eps_decay_steps))
        return self.eps0 + (self.eps_min - self.eps0) * frac
    def update(self, quality):
        self._qhist.append(quality)
        e = quality - self.target; s = self._state(e)
        qrow = self.Q.setdefault(s, [0.0] * len(self._ACTIONS))
        if self._prev_s is not None:
            r = -abs(e); prow = self.Q.setdefault(self._prev_s, [0.0] * len(self._ACTIONS))
            prow[self._prev_a] += self.alpha * (r + self.gamma * max(qrow) - prow[self._prev_a])
        if self.rng.random() < self._eps():
            a = self.rng.randrange(len(self._ACTIONS))
        else:
            mx = max(qrow); a = self.rng.choice([i for i, v in enumerate(qrow) if v == mx])
        self._prev_s = s; self._prev_a = a; self.t += 1
        self.last_push = self._ACTIONS[a]
        return self.last_push


class AdaptiveActivationSparsityController:
    """Controller for adaptive activation sparsification with quality monitoring"""
    def __init__(self, layer_idx, target_quality=CATS_TARGET, quality_check_prob=QUALITY_CHECK_PROB):
        self.layer_idx = layer_idx
        self.threshold = None  # Will be initialized on first forward
        self.pid = BufferedPID(kp=0.5, ki=0.1, kd=0.05, buffer_size=5, max_age=5)
        self.target_quality = target_quality
        self._rl = RLThresholdController(self.target_quality, seed=layer_idx) if CATS_CONTROLLER == 'rl' else None
        self.step = 0
        self.quality_check_prob = quality_check_prob
        self.last_quality_check_step = 0
        self.sparsity_history = deque(maxlen=100)
        self.quality_history = deque(maxlen=100)
        self.threshold_history = deque(maxlen=100)
        self.pid_history = deque(maxlen=100)
        self.error_history = deque(maxlen=100)

        self.log_dir = "sparsity_logs"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def update(self, activated_gate, force_check=False, up_output=None, down_proj=None, residual=None):
        """Update threshold based on quality monitoring of activated gate values.

        Args:
            activated_gate: The activated gate tensor to threshold.
            force_check: If True, force a quality check this step (used by MIMD interval logic).
            up_output/down_proj/residual: used only when ACAS_CATS_SIGNAL='contrib' (output drift).
        """
        if self.threshold is None:
            activated_abs_float = activated_gate.abs().float()
            self.threshold = torch.quantile(activated_abs_float, 0.01).item()
            return self.threshold, 0.01, 1.0

        mask = activated_gate.abs() > self.threshold
        sparsity = (~mask).float().mean().item()
        quality = 1.0
        should_check = force_check or random.random() < self.quality_check_prob
        if self.step - self.last_quality_check_step > 500:
            should_check = True

        if should_check:
            with torch.no_grad():
                activated_sparse = activated_gate * mask
                if CATS_SIGNAL == "contrib" and up_output is not None and down_proj is not None and residual is not None:
                    # output-CONTRIBUTION drift: ||y_dense - y_sparse|| at the MLP OUTPUT (post up/down
                    # proj), normalized by ||residual|| -> drift relative to the residual stream. Same
                    # formula as the PID-contrib CATS file; only the downstream cadence is MIMD.
                    y_dense  = down_proj(activated_gate  * up_output)
                    y_sparse = down_proj(activated_sparse * up_output)
                    error = (y_dense - y_sparse).norm() / (residual.norm() + 1e-8)
                else:
                    # paper signal: L2 drift on the activated gate values (matches sparse_gate_2d C++ kernel)
                    error = (activated_gate - activated_sparse).norm() / (activated_gate.norm() + 1e-8)
                quality = max(0.0, min(1.0, (1 - error).item()))

                if CATS_CONTROLLER == 'rl':
                    push = self._rl.update(quality)
                    adjustment = p_term = i_term = d_term = error_for_pid = 0.0
                    threshold_change = push * self.threshold * 0.010   # validated RL apply scale
                    self.threshold = max(1e-6, self.threshold + threshold_change)
                else:
                    error_for_pid = calculate_error(quality, self.target_quality)
                    self.pid.add_sample(-error_for_pid, self.step)
                    adjustment, p_term, i_term, d_term = self.pid.get_update(self.step)
                    threshold_change = adjustment * self.threshold * 0.02
                    self.threshold = max(1e-6, self.threshold + threshold_change)

                activated_abs_float = activated_gate.abs().float()
                max_threshold = torch.quantile(activated_abs_float, 0.95).item()
                min_threshold = torch.quantile(activated_abs_float, 0.001).item()
                self.threshold = max(min_threshold, min(max_threshold, self.threshold))

                self.sparsity_history.append(sparsity)
                self.quality_history.append(quality)
                self.threshold_history.append(self.threshold)
                self.error_history.append(error_for_pid)
                self.pid_history.append((adjustment, p_term, i_term, d_term))
                self.write_logs()
                self.last_quality_check_step = self.step

        self.step += 1
        return self.threshold, sparsity, quality

    def write_logs(self):
        """Write metrics to log file"""
        log_file = os.path.join(self.log_dir, f"layer_{self.layer_idx}_activation.log")
        if not os.path.exists(log_file):
            with open(log_file, "w") as f:
                f.write("Step\tSparsity\tQuality\tThreshold\tError\tAdjustment\tP_term\tI_term\tD_term\tAvg_Sparsity\tAvg_Quality\n")
        current_sparsity = self.sparsity_history[-1] if self.sparsity_history else 0
        current_quality = self.quality_history[-1] if self.quality_history else 1
        avg_sparsity = np.mean(self.sparsity_history) if self.sparsity_history else 0
        avg_quality = np.mean(self.quality_history) if self.quality_history else 1
        current_error = self.error_history[-1] if self.error_history else 0
        if self.pid_history:
            adjustment, p_term, i_term, d_term = self.pid_history[-1]
        else:
            adjustment, p_term, i_term, d_term = 0, 0, 0, 0
        with open(log_file, "a") as f:
            f.write(f"{self.step}\t{current_sparsity:.4f}\t{current_quality:.4f}\t{self.threshold:.6f}\t"
                    f"{current_error:.4f}\t{adjustment:.4f}\t{p_term:.4f}\t{i_term:.4f}\t{d_term:.4f}\t"
                    f"{avg_sparsity:.4f}\t{avg_quality:.4f}\n")
            f.flush()


@use_kernel_forward_from_hub("RMSNorm")
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

        # Adaptive activation sparsity controller (replaces CATS static thresholds)
        self.activation_sparsity_controller = AdaptiveActivationSparsityController(layer_idx)
        self.enable_adaptive_sparsity = True

        # Statistics tracking
        self.total_tokens = 0
        self.total_sparsity = 0.0
        self.avg_quality = 1.0

    def forward(self, x, force_check=False, residual=None):
        if x.size(1) == 1 and self.enable_adaptive_sparsity:
            # CATS-style activation sparsification with adaptive threshold
            gate_output = self.gate_proj(x)
            up_output = self.up_proj(x)
            activated = self.act_fn(gate_output)

            # Apply adaptive activation sparsity. up_output/down_proj/residual let the controller
            # measure output-CONTRIBUTION drift when ACAS_CATS_SIGNAL='contrib' (else ignored).
            activation_threshold, activation_sparsity, activation_quality = self.activation_sparsity_controller.update(
                activated, force_check=force_check, up_output=up_output, down_proj=self.down_proj, residual=residual)
            activation_mask = activated.abs() > activation_threshold
            activated_sparse = activated * activation_mask

            # Final computation with activation sparsity
            down_proj = self.down_proj(activated_sparse * up_output)

            # Track statistics
            self.total_tokens += 1
            self.total_sparsity += activation_sparsity
            self.avg_quality = 0.99 * self.avg_quality + 0.01 * activation_quality

        else:
            # Normal forward pass (training or prefill)
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config, layer_idx=layer_idx)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        force_check: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        # pass the pre-MLP residual stream so the contrib signal can normalize by ||residual||
        hidden_states = self.mlp(hidden_states, force_check=force_check, residual=residual)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class LlamaPreTrainedModel(PreTrainedModel):
    config: LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": LlamaDecoderLayer,
        "attentions": LlamaAttention,
    }


@auto_docstring
class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

        # MIMD interval state (per layer)
        self._mimd_interval = [MIMD_INIT_INTERVAL] * config.num_hidden_layers
        self._mimd_tokens_since = [0] * config.num_hidden_layers
        self._global_step = 0

        # === Tracking ===
        self._track_enabled = True
        self._track_seq_id = 0
        self._track_token_in_seq = 0
        # Per-step records: list of dicts
        self._track_records = []
        # Per-sequence summary
        self._track_seq_summaries = []
        # Auto-dump on exit
        self._track_output_dir = _os.environ.get('ACAS_TRACK_DIR', '/tmp/acas_tracking')

        def _auto_dump(model_ref=self):
            try:
                model_ref.dump_tracking(model_ref._track_output_dir)
            except Exception as e:
                print(f"[TRACKING] Failed to auto-dump: {e}")
        atexit.register(_auto_dump)

    def _get_thresholds(self):
        """Get current activation thresholds for all layers."""
        thresholds = []
        for layer in self.layers:
            ctrl = layer.mlp.activation_sparsity_controller
            thresholds.append(ctrl.threshold if ctrl.threshold is not None else 0.0)
        return thresholds

    def dump_tracking(self, output_dir):
        """Save all tracking data to output_dir."""
        import os, json, csv
        os.makedirs(output_dir, exist_ok=True)

        # Save final summary for the last sequence
        if self._track_seq_id > 0:
            self._track_seq_summaries.append({
                'seq_id': self._track_seq_id - 1,
                'final_thresholds': self._get_thresholds(),
                'final_intervals': [self._mimd_interval[i] for i in range(self.config.num_hidden_layers)],
                'tokens_generated': self._track_token_in_seq,
                'updates_this_seq': self._global_step - getattr(self, '_track_step_at_seq_start', 0),
            })

        # 1. Per-update threshold trace CSV
        if self._track_records:
            csv_path = os.path.join(output_dir, 'threshold_trace.csv')
            fieldnames = list(self._track_records[0].keys())
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                records = self._track_records
                if len(records) > 100000:
                    records = records[::10]
                writer.writerows(records)
            print(f"[TRACKING] Threshold trace: {len(self._track_records)} records -> {csv_path}")

        # 2. Per-sequence summaries JSON
        if self._track_seq_summaries:
            summary_path = os.path.join(output_dir, 'seq_summaries.json')
            with open(summary_path, 'w') as f:
                json.dump(self._track_seq_summaries, f, indent=2)
            print(f"[TRACKING] Sequence summaries: {len(self._track_seq_summaries)} seqs -> {summary_path}")

        # 3. Aggregate stats
        stats = {
            'total_sequences': len(self._track_seq_summaries),
            'total_updates': self._global_step,
            'final_thresholds': self._get_thresholds(),
            'threshold_range_per_layer': {},
        }
        for layer in range(self.config.num_hidden_layers):
            key = f'threshold_L{layer}'
            vals = [r[key] for r in self._track_records if key in r]
            if vals:
                stats['threshold_range_per_layer'][f'L{layer}'] = {
                    'min': min(vals),
                    'max': max(vals),
                    'final': vals[-1],
                    'mean': sum(vals) / len(vals),
                }
        stats_path = os.path.join(output_dir, 'tracking_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"[TRACKING] Stats -> {stats_path}")

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
            # New sequence detected — save summary of previous sequence
            if self._track_enabled and self._track_seq_id > 0:
                self._track_seq_summaries.append({
                    'seq_id': self._track_seq_id - 1,
                    'final_thresholds': self._get_thresholds(),
                    'final_intervals': [self._mimd_interval[i] for i in range(self.config.num_hidden_layers)],
                    'tokens_generated': self._track_token_in_seq,
                    'updates_this_seq': self._global_step - getattr(self, '_track_step_at_seq_start', 0),
                })
                self._track_step_at_seq_start = self._global_step
            elif self._track_enabled:
                self._track_step_at_seq_start = 0
            self._track_token_in_seq = 0
            self._track_seq_id += 1

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        seq_length = inputs_embeds.shape[1]

        # Track which layers had quality checks this step
        checked_layers = []

        for idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            # Per-layer MIMD injection scheduling
            force_check = False
            if seq_length == 1:
                self._mimd_tokens_since[idx] += 1
                if self._mimd_tokens_since[idx] >= int(self._mimd_interval[idx]):
                    force_check = True
                    self._mimd_tokens_since[idx] = 0

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                force_check=force_check,
                **kwargs,
            )

            # MIMD interval update: adjust interval based on quality error
            if force_check and seq_length == 1:
                ctrl = decoder_layer.mlp.activation_sparsity_controller
                if ctrl.quality_history:
                    latest_quality = ctrl.quality_history[-1]
                    abs_error = abs(latest_quality - ctrl.target_quality)
                    if abs_error > MIMD_ERROR_THRESHOLD:
                        self._mimd_interval[idx] = max(MIMD_MIN_INTERVAL, self._mimd_interval[idx] * MIMD_DECREASE)
                    else:
                        self._mimd_interval[idx] = min(MIMD_MAX_INTERVAL, self._mimd_interval[idx] * MIMD_INCREASE)
                checked_layers.append(idx)

        # === Tracking: record per-update state ===
        if self._track_enabled and seq_length == 1 and checked_layers:
            record = {
                'step': self._global_step,
                'seq_id': self._track_seq_id,
                'token_in_seq': self._track_token_in_seq,
            }
            for idx2 in range(self.config.num_hidden_layers):
                ctrl = self.layers[idx2].mlp.activation_sparsity_controller
                record[f'threshold_L{idx2}'] = ctrl.threshold if ctrl.threshold is not None else 0.0
                record[f'quality_L{idx2}'] = ctrl.quality_history[-1] if ctrl.quality_history else -1
                record[f'interval_L{idx2}'] = self._mimd_interval[idx2]
            self._track_records.append(record)
            self._global_step += 1

        # Track token count
        if self._track_enabled and seq_length == 1:
            self._track_token_in_seq += 1

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LlamaForSequenceClassification(GenericForSequenceClassification, LlamaPreTrainedModel): ...


class LlamaForQuestionAnswering(GenericForQuestionAnswering, LlamaPreTrainedModel):
    base_model_prefix = "transformer"  # For BC, where `transformer` was used instead of `model`


class LlamaForTokenClassification(GenericForTokenClassification, LlamaPreTrainedModel): ...


__all__ = [
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaPreTrainedModel",
    "LlamaForSequenceClassification",
    "LlamaForQuestionAnswering",
    "LlamaForTokenClassification",
]
