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

logger = logging.get_logger(__name__)

import numpy as np
from collections import deque
import os
import random

# Configuration for activation thresholding
TARGET_QUALITY_ACTIVATION = 0.80
QUALITY_CHECK_PROB = 0.005

# --- contribution-metric signal -------------------------------------------------
# Mirrors the Grasp/SI L2-swap: change ONLY the per-layer PID quality signal, nothing else.
#   ACAS_CATS_SIGNAL  : "activation" (paper L2 drift on activated gate, default) | "contrib"
#                       ("contrib" = ||y_dense - y_sparse|| at the MLP OUTPUT, normalized by ||residual||)
#   ACAS_CATS_TARGET  : PID target for the chosen signal (default 0.80, the paper's activation target)
#   ACAS_CATS_BACKSTOP: "1" keep paper's 500-step force-check (default; use for parity) |
#                       "0" pure 0.5% Bernoulli checks (matches Grasp/SI for cross-method comparison)
CATS_SIGNAL   = os.environ.get('ACAS_CATS_SIGNAL', 'activation').lower()
CATS_TARGET   = float(os.environ.get('ACAS_CATS_TARGET', str(TARGET_QUALITY_ACTIVATION)))
CATS_BACKSTOP = os.environ.get('ACAS_CATS_BACKSTOP', '1') == '1'
# controller-swap ablation: 'pid' (default, byte-identical) | 'rl' (tabular Q-learning, same signal)
CATS_CONTROLLER = os.environ.get('ACAS_CATS_CONTROLLER', 'pid').lower()


# =========================================================================
# ACAS-Q1 sparsity-pattern tap  (optional instrumentation)
# Accumulates per-layer zero ratio + nonzero-magnitude histogram on every
# generation forward pass. Dumps a single sparsity_summary.json on process
# exit. Disable with ACAS_Q1_TAP=0. Output dir: $ACAS_TRACK_DIR.
# =========================================================================
import json as _json
import atexit as _atexit
import math as _math

_ACAS_Q1_ENABLED  = os.environ.get("ACAS_Q1_TAP", "1") != "0"
_ACAS_Q1_DIR      = os.environ.get("ACAS_TRACK_DIR", "sparsity_logs")
_ACAS_Q1_HIST_BINS = 50
_ACAS_Q1_LOG_MIN  = _math.log(1e-6)
_ACAS_Q1_LOG_MAX  = _math.log(10.0)
_ACAS_Q1_STATS    = {}  # layer_idx -> dict of GPU tensors

# Frozen-threshold mode: load per-layer thresholds from JSON, bypass PID.
# Lets us reuse this single modeling file for both:
#   (i)  CATS-50% measurement pass — load CATS-50%'s static thresholds
#   (ii) ACAS-CATS measurement pass — load converged thresholds from prior PID run
_ACAS_FROZEN_PATH = os.environ.get("ACAS_FROZEN_THRESHOLDS", "")
_ACAS_FROZEN_THR  = {}  # layer_idx (int) -> threshold (float)
if _ACAS_FROZEN_PATH and os.path.exists(_ACAS_FROZEN_PATH):
    with open(_ACAS_FROZEN_PATH) as _f:
        _data = _json.load(_f)
    if isinstance(_data, dict):
        _ACAS_FROZEN_THR = {int(k): float(v) for k, v in _data.items()}
    elif isinstance(_data, list):
        _ACAS_FROZEN_THR = {i: float(v) for i, v in enumerate(_data)}
    print(f"[ACAS-Q1] Frozen-threshold mode: loaded {len(_ACAS_FROZEN_THR)} thresholds from {_ACAS_FROZEN_PATH}")

def _acas_q1_tap(layer_idx, activated, mask):
    if not _ACAS_Q1_ENABLED:
        return
    dev = mask.device
    s = _ACAS_Q1_STATS.get(layer_idx)
    if s is None:
        s = {
            "n_forward":  torch.zeros((), device=dev, dtype=torch.long),
            "total_elem": torch.zeros((), device=dev, dtype=torch.long),
            "zero_elem":  torch.zeros((), device=dev, dtype=torch.long),
            "hist":       torch.zeros(_ACAS_Q1_HIST_BINS, device=dev, dtype=torch.long),
            "sum_mag":    torch.zeros((), device=dev, dtype=torch.float64),
            "sum_sq_mag": torch.zeros((), device=dev, dtype=torch.float64),
            "max_mag":    torch.zeros((), device=dev, dtype=torch.float32),
            "min_mag":    torch.full((), float('inf'), device=dev, dtype=torch.float32),
        }
        _ACAS_Q1_STATS[layer_idx] = s
    with torch.no_grad():
        s["n_forward"]  += 1
        s["total_elem"] += mask.numel()
        s["zero_elem"]  += (~mask).sum()
        nz = activated.abs()[mask]
        if nz.numel() > 0:
            nz_f = nz.float()
            s["sum_mag"]    += nz_f.sum().double()
            s["sum_sq_mag"] += (nz_f * nz_f).sum().double()
            s["max_mag"]     = torch.maximum(s["max_mag"], nz_f.max())
            s["min_mag"]     = torch.minimum(s["min_mag"], nz_f.min())
            log_nz = nz_f.clamp(min=1e-6, max=10.0).log()
            h = torch.histc(log_nz, bins=_ACAS_Q1_HIST_BINS,
                            min=_ACAS_Q1_LOG_MIN, max=_ACAS_Q1_LOG_MAX)
            s["hist"] += h.long()

def _acas_q1_dump():
    if not _ACAS_Q1_ENABLED or not _ACAS_Q1_STATS:
        return
    os.makedirs(_ACAS_Q1_DIR, exist_ok=True)
    layers = {}
    for li, s in sorted(_ACAS_Q1_STATS.items()):
        total = int(s["total_elem"].item())
        if total == 0:
            continue
        zeros = int(s["zero_elem"].item())
        n_nz  = total - zeros
        sum_m = float(s["sum_mag"].item())
        sumsq = float(s["sum_sq_mag"].item())
        mean  = sum_m / n_nz if n_nz else 0.0
        var   = max(0.0, sumsq / n_nz - mean * mean) if n_nz else 0.0
        mn    = float(s["min_mag"].item())
        layers[f"layer_{li}"] = {
            "n_forward":         int(s["n_forward"].item()),
            "total_elements":    total,
            "zero_elements":     zeros,
            "nonzero_elements":  n_nz,
            "zero_ratio":        zeros / total,
            "nonzero_mag_mean":  mean,
            "nonzero_mag_std":   var ** 0.5,
            "nonzero_mag_min":   mn if _math.isfinite(mn) else 0.0,
            "nonzero_mag_max":   float(s["max_mag"].item()),
            "mag_hist_log_bins": s["hist"].tolist(),
        }
    meta = {
        "model":        os.environ.get("ACAS_RUN_MODEL", ""),
        "benchmark":    os.environ.get("ACAS_RUN_BENCHMARK", ""),
        "variant":      os.environ.get("ACAS_RUN_VARIANT", ""),
        "seed":         os.environ.get("ACAS_RUN_SEED", ""),
        "num_layers":   len(layers),
        "hist_bins":    _ACAS_Q1_HIST_BINS,
        "hist_log_min": _ACAS_Q1_LOG_MIN,
        "hist_log_max": _ACAS_Q1_LOG_MAX,
        "hist_min":     1e-6,
        "hist_max":     10.0,
    }
    path = os.path.join(_ACAS_Q1_DIR, "sparsity_summary.json")
    with open(path, "w") as f:
        _json.dump({"meta": meta, "layers": layers}, f, indent=2)
    print(f"[ACAS-Q1] sparsity summary written to {path} ({len(layers)} layers)")

_atexit.register(_acas_q1_dump)
# =========================================================================


# Helper classes for adaptive sparsification
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
        self.samples = deque(maxlen=buffer_size)
        
    def add_sample(self, error, step):
        self.samples.append(Sample(error, step))
        
    def get_update(self, current_step):
        if not self.samples:
            return 0.0, 0.0, 0.0, 0.0
            
        # Filter samples by age
        valid_samples = [s for s in self.samples 
                        if current_step - s.step <= self._max_age]
        
        if not valid_samples:
            return 0.0, 0.0, 0.0, 0.0
            
        current_error = valid_samples[-1].error
        
        # Calculate integral (sum of all valid errors)
        integral = sum(s.error for s in valid_samples)
        
        # Calculate derivative
        derivative = 0.0
        if len(valid_samples) >= 2:
            derivative = current_error - valid_samples[len(valid_samples) - 2].error

        p_term = self._kp * current_error
        i_term = self._ki * integral
        d_term = self._kd * derivative

        adjustment = p_term + i_term + d_term

        return adjustment, p_term, i_term, d_term

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
    
    # Calculate the logit difference
    diff = logit(target_value) - logit(current_value)
    
    # Custom sigmoid function to scale the error
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
    def __init__(self, layer_idx, target_quality=None, quality_check_prob=QUALITY_CHECK_PROB):
        self.layer_idx = layer_idx
        # Frozen mode: if a per-layer threshold table was loaded, use it and skip PID
        self.frozen = bool(_ACAS_FROZEN_THR) and (layer_idx in _ACAS_FROZEN_THR)
        if self.frozen:
            self.threshold = float(_ACAS_FROZEN_THR[layer_idx])
        else:
            self.threshold = None  # initialized on first forward
        self.pid = BufferedPID(kp=0.5, ki=0.1, kd=0.05, buffer_size=5, max_age=5)
        # contribution-metric swap: drive the PID to ACAS_CATS_TARGET (defaults to the paper's 0.80)
        self.target_quality = CATS_TARGET if target_quality is None else target_quality
        self._rl = RLThresholdController(self.target_quality, seed=layer_idx) if CATS_CONTROLLER == 'rl' else None
        self.step = 0
        self.quality_check_prob = quality_check_prob
        self.last_quality_check_step = 0
        self.sparsity_history = deque(maxlen=100)
        self.quality_history = deque(maxlen=100)
        self.threshold_history = deque(maxlen=100)
        self.pid_history = deque(maxlen=100)
        self.error_history = deque(maxlen=100)

        # Create logging directory structure
        self.log_dir = "sparsity_logs/activation"
        self.summary_dir = "sparsity_logs/summary"
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.summary_dir, exist_ok=True)

    def update(self, activated_gate, up_output=None, down_proj=None, residual=None):
        """Update threshold based on quality monitoring of activated gate values.
        up_output/down_proj/residual are only used when ACAS_CATS_SIGNAL='contrib' (output drift)."""
        # Frozen mode: just apply the fixed threshold and return — no PID, no logging
        if self.frozen:
            mask = activated_gate.abs() > self.threshold
            sparsity = (~mask).float().mean().item()
            self.step += 1
            return self.threshold, sparsity, 1.0

        # Initialize threshold on first pass
        if self.threshold is None:
            # Start with 1% sparsity (conservative)
            activated_abs_float = activated_gate.abs().float()
            self.threshold = torch.quantile(activated_abs_float, 0.01).item()
            return self.threshold, 0.01, 1.0

        # Apply current threshold to activated gate output
        mask = activated_gate.abs() > self.threshold
        sparsity = (~mask).float().mean().item()
        
        # Random quality check
        quality = 1.0  # Default to perfect quality
        should_check = random.random() < self.quality_check_prob
        
        # Force a check if we haven't checked in 500 steps (paper backstop; ACAS_CATS_BACKSTOP=0
        # disables it -> pure 0.5% Bernoulli checks, matching Grasp/SI for cross-method comparison)
        if CATS_BACKSTOP and (self.step - self.last_quality_check_step > 500):
            should_check = True

        if should_check:
            with torch.no_grad():
                # Compute quality on this dense-check token: how much does masking drift the layer?
                activated_sparse = activated_gate * mask

                if CATS_SIGNAL == "contrib" and up_output is not None and down_proj is not None and residual is not None:
                    # output-CONTRIBUTION drift: ||y_dense - y_sparse|| at the MLP OUTPUT (post up/down proj,
                    # so it counts only drift that survives the projections), normalized by ||residual|| ->
                    # drift relative to the residual stream the MLP output is added to (closer to logit impact).
                    y_dense  = down_proj(activated_gate  * up_output)
                    y_sparse = down_proj(activated_sparse * up_output)
                    error = (y_dense - y_sparse).norm() / (residual.norm() + 1e-8)
                else:
                    # paper signal: L2 drift on the activated gate values (matches sparse_gate_2d kernel in C++)
                    error = (activated_gate - activated_sparse).norm() / (activated_gate.norm() + 1e-8)
                quality = (1 - error).item()
                quality = max(0.0, min(1.0, quality))
                
                # controller step (PID default; RL drop-in learns the same setpoint from reward)
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
                
                # Apply bounds
                activated_abs_float = activated_gate.abs().float()
                max_threshold = torch.quantile(activated_abs_float, 0.95).item()
                min_threshold = torch.quantile(activated_abs_float, 0.001).item()
                self.threshold = max(min_threshold, min(max_threshold, self.threshold))
                
                # Log metrics
                self.sparsity_history.append(sparsity)
                self.quality_history.append(quality)
                self.threshold_history.append(self.threshold)
                self.error_history.append(error_for_pid)
                self.pid_history.append((adjustment, p_term, i_term, d_term))
                
                # Write to log file
                self.write_logs()
                
                # Update last quality check step
                self.last_quality_check_step = self.step
        
        self.step += 1
        return self.threshold, sparsity, quality

    def write_logs(self):
            """Write metrics to CSV file"""
            # Individual layer CSV file
            log_file = os.path.join(self.log_dir, f"layer_{self.layer_idx}.csv")
            
            # Write header if file is new
            if not os.path.exists(log_file):
                with open(log_file, "w") as f:
                    f.write("Step,Sparsity,Quality,Threshold,Error,Adjustment,P_term,I_term,D_term,Avg_Sparsity,Avg_Quality\n")
            
            # Write current metrics
            current_sparsity = self.sparsity_history[-1] if self.sparsity_history else 0
            current_quality = self.quality_history[-1] if self.quality_history else 1
            avg_sparsity = np.mean(self.sparsity_history) if self.sparsity_history else 0
            avg_quality = np.mean(self.quality_history) if self.quality_history else 1
            
            # Get PID data
            current_error = self.error_history[-1] if self.error_history else 0
            if self.pid_history:
                adjustment, p_term, i_term, d_term = self.pid_history[-1]
            else:
                adjustment, p_term, i_term, d_term = 0, 0, 0, 0
            
            with open(log_file, "a") as f:
                f.write(f"{self.step},")
                f.write(f"{current_sparsity:.4f},")
                f.write(f"{current_quality:.4f},")
                f.write(f"{self.threshold:.6f},")
                f.write(f"{current_error:.4f},")
                f.write(f"{adjustment:.4f},")
                f.write(f"{p_term:.4f},")
                f.write(f"{i_term:.4f},")
                f.write(f"{d_term:.4f},")
                f.write(f"{avg_sparsity:.4f},")
                f.write(f"{avg_quality:.4f}\n")
                f.flush()
            
            # Summary CSV file
            summary_file = os.path.join(self.summary_dir, "activation_summary.csv")
            
            # Write header if file is new
            if not os.path.exists(summary_file):
                with open(summary_file, "w") as f:
                    f.write("Step,Layer,Sparsity,Quality,Threshold\n")
            
            # Append to summary
            with open(summary_file, "a") as f:
                f.write(f"{self.step},{self.layer_idx},{current_sparsity:.4f},")
                f.write(f"{current_quality:.4f},{self.threshold:.6f}\n")
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
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

        # Activation-only sparsity controller
        if layer_idx is not None:
            self.activation_sparsity_controller = AdaptiveActivationSparsityController(layer_idx)
            self.enable_adaptive_sparsity = True
        else:
            self.enable_adaptive_sparsity = False

        # Statistics tracking
        self.total_tokens = 0
        self.total_sparsity = 0.0
        self.avg_quality = 1.0

    def forward(self, x, residual=None):
        if x.size(1) == 1 and self.enable_adaptive_sparsity:
            # Activation-only sparsification for generation mode (no input thresholding)

            # Forward pass with full dense input
            gate_output = self.gate_proj(x)
            up_output = self.up_proj(x)
            activated = self.act_fn(gate_output)

            # Apply activation sparsity (on activated gate values).
            # up_output/down_proj/residual let the controller measure output-CONTRIBUTION drift
            # (||y_dense - y_sparse|| / ||residual||) when ACAS_CATS_SIGNAL='contrib'.
            activation_threshold, activation_sparsity, activation_quality = self.activation_sparsity_controller.update(
                activated, up_output=up_output, down_proj=self.down_proj, residual=residual)
            activation_mask = activated.abs() > activation_threshold
            _acas_q1_tap(self.layer_idx, activated, activation_mask)  # ACAS-Q1 instrumentation
            activated_sparse = activated * activation_mask
            
            # Final computation with activation sparsity only
            down_proj = self.down_proj(activated_sparse * up_output)
            
            # Track statistics (activation-only)
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
        hidden_states = self.mlp(hidden_states, residual)
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

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

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
