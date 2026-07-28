# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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

import atexit
import json
import os
import csv
import random
from collections import deque

import numpy as np
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


# =========================
# Shared helpers and config
# =========================

# Activation (teacher) controls
TARGET_QUALITY_ACTIVATION = 0.85

# Predictor (student) controls — edit these directly rather than env vars
PRED_THRESHOLD_INIT = 10.0      # initial predictor threshold
PRED_THRESHOLD_MIN = 0.1         # minimum predictor threshold
PRED_AUDIT_PROB = 0.05          # probability of doing an audit step
TARGET_WRONG_SKIP = 0.05        # target wrong-skip share

# --- contribution-metric swap (two-knob CATS-GP), MIMD-cadence variant ---------
# Identical per-knob signal swap to the PID-contrib CATS-GR file (modeling_acas_catsgp.py);
# the MIMD adaptive check-interval machinery is left untouched (it now adapts on contrib drift,
# since the interval logic below reads each controller's actual signal+target rather than constants).
# On audit tokens (dense already computed) we add a couple of down_proj calls:
#   threshold knob : 1 - ||down(act*up) - down((act*mask)*up)|| / ||residual||   (toward CATSGR_TARGET)
#   predictor knob : ||down(act*up) - down((act with pred-low zeroed)*up)|| / ||residual||
#                    (a drift "badness" driven toward 1-CATSGR_GATE_TARGET via the existing PID path)
# Targets are INDEPENDENT — sweep as a grid like the paper's th1 x th2.
# ACAS_CATSGR_S_VALUES overrides the calibration CSV path (the stock path is CWD-relative).
CATSGR_SIGNAL      = os.environ.get('ACAS_CATSGR_SIGNAL', 'baseline').lower()       # 'baseline' | 'contrib'
CATSGR_TARGET  = float(os.environ.get('ACAS_CATSGR_TARGET',  str(TARGET_QUALITY_ACTIVATION)))
CATSGR_GATE_TARGET = float(os.environ.get('ACAS_CATSGR_GATE_TARGET', '0.95'))
CATSGR_S_VALUES    = os.environ.get('ACAS_CATSGR_S_VALUES', '')
# controller-swap ablation: 'pid' (default, byte-identical) | 'rl' (tabular Q-learning on BOTH knobs)
CATSGR_CONTROLLER  = os.environ.get('ACAS_CATSGR_CONTROLLER', 'pid').lower()

# MIMD interval control constants
MIMD_INCREASE = 1.5
MIMD_DECREASE = 0.5
MIMD_MIN_INTERVAL = int(os.environ.get("ACAS_NMIN", "50"))
MIMD_MAX_INTERVAL = int(os.environ.get("ACAS_NMAX", "500"))
MIMD_INIT_INTERVAL = 200
MIMD_ERROR_THRESHOLD = 0.01


class Sample:
    def __init__(self, error, step):
        self.error = error
        self.step = step


class BufferedPID:
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
        valid = [s for s in self._buffer if current_step - s.step <= self._max_age]
        if not valid:
            return 0.0, 0.0, 0.0, 0.0
        current_error = valid[-1].error
        integral = sum(s.error for s in valid)
        derivative = current_error - valid[-2].error if len(valid) >= 2 else 0.0
        p_term = self._kp * current_error
        i_term = self._ki * integral
        d_term = self._kd * derivative
        adjustment = p_term + i_term + d_term
        return adjustment, p_term, i_term, d_term


def calculate_error(current_value, target_value, sensitivity=5.0):
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


# ==========================================================
# Activation sparsity controller (unchanged logic, minor IO)
# ==========================================================


# === RL controller: tabular Q-learning drop-in for the PID. Used on BOTH CATS-GR knobs
# (activation threshold + predictor threshold) via ACAS_CATSGR_CONTROLLER=rl. Same information as the
# PID (signal vs target) + same knob; reward = -|signal - target|. Validated within 0.3% of the PID
# setpoint (validated offline). Caller scales push: threshold *= (1+push*0.010).
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
    def __init__(self, layer_idx, target_quality=TARGET_QUALITY_ACTIVATION):
        self.layer_idx = layer_idx
        self.threshold = None
        self.pid = BufferedPID(kp=0.5, ki=0.1, kd=0.05, buffer_size=5, max_age=5)
        # contribution-metric swap: in contrib mode the PID drives toward CATSGR_TARGET
        self.target_quality = CATSGR_TARGET if CATSGR_SIGNAL == 'contrib' else target_quality
        self._rl = RLThresholdController(self.target_quality, seed=layer_idx) if CATSGR_CONTROLLER == 'rl' else None
        self.step = 0
        self.sparsity_history = deque(maxlen=100)
        self.quality_history = deque(maxlen=100)
        self.threshold_history = deque(maxlen=100)
        self.pid_history = deque(maxlen=100)
        self.error_history = deque(maxlen=100)

        self.log_dir = "sparsity_logs"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

        # Predictor metrics mirrored here for unified logging
        self.pred_precision_hist = None
        self.pred_recall_hist = None
        self.pred_fpr_hist = None
        self.pred_sparsity_hist = None
        self._pred_threshold = None
        self.predictor_alpha = 0
        self.pred_wrong_skip_hist = None
        self.pred_p_true_hist = None

    def update(self, activated_gate, do_quality_check: bool = True, force_check: bool = False,
               up_output=None, down_proj=None, residual=None):
        # force_check overrides do_quality_check to True
        if force_check:
            do_quality_check = True

        if self.threshold is None:
            activated_abs_float = activated_gate.abs().float()
            self.threshold = torch.quantile(activated_abs_float, 0.20).item()
            return self.threshold, 0.01, 1.0

        mask = activated_gate.abs() > self.threshold
        sparsity = (~mask).float().mean().item()

        quality = 1.0
        if do_quality_check:
            with torch.no_grad():
                activated_sparse = activated_gate * mask
                if CATSGR_SIGNAL == 'contrib' and up_output is not None and down_proj is not None and residual is not None:
                    # threshold knob's STANDALONE output-contribution drift (no predictor): how much
                    # does the activation mask alone move the MLP output, relative to ||residual||.
                    y_dense = down_proj(activated_gate * up_output)
                    y_thr   = down_proj(activated_sparse * up_output)
                    error = (y_dense - y_thr).norm() / (residual.norm() + 1e-8)
                else:
                    # paper signal: L2 drift on the activated gate values
                    error = (activated_gate - activated_sparse).norm() / (activated_gate.norm() + 1e-8)
                quality = float(max(0.0, min(1.0, 1 - error.item())))

                if CATSGR_CONTROLLER == 'rl':
                    push = self._rl.update(quality)
                    adjustment = p_term = i_term = d_term = err = 0.0
                    threshold_change = push * self.threshold * 0.010   # validated RL apply scale
                    self.threshold = max(1e-6, self.threshold + threshold_change)
                else:
                    err = calculate_error(quality, self.target_quality)
                    self.pid.add_sample(-err, self.step)
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
                self.error_history.append(err)
                self.pid_history.append((adjustment, p_term, i_term, d_term))

                self.write_logs()

        self.step += 1
        return self.threshold, sparsity, quality

    def write_logs(self):
        layer_dir = os.path.join(self.log_dir, f"layer_{self.layer_idx}")
        if not os.path.exists(layer_dir):
            os.makedirs(layer_dir, exist_ok=True)
        log_file = os.path.join(layer_dir, "activation.csv")

        if not os.path.exists(log_file):
            with open(log_file, "w") as f:
                f.write(
                    "Step,Sparsity,Quality,Threshold,Error,Adjustment,P_term,I_term,D_term,Avg_Sparsity,Avg_Quality,PredAlpha,PredThr,PredPrec,PredRec,PredFPR,PredWrongSkip,PredSparsity\n"
                )

        current_sparsity = self.sparsity_history[-1] if self.sparsity_history else 0
        current_quality = self.quality_history[-1] if self.quality_history else 1
        avg_sparsity = np.mean(self.sparsity_history) if self.sparsity_history else 0
        avg_quality = np.mean(self.quality_history) if self.quality_history else 1

        pred_precision_hist = getattr(self, 'pred_precision_hist', None)
        pred_recall_hist = getattr(self, 'pred_recall_hist', None)
        pred_fpr_hist = getattr(self, 'pred_fpr_hist', None)
        pred_wrong_skip_hist = getattr(self, 'pred_wrong_skip_hist', None)
        pred_sparsity_hist = getattr(self, 'pred_sparsity_hist', None)

        pred_prec = (pred_precision_hist[-1] if pred_precision_hist else 0.0)
        pred_rec = (pred_recall_hist[-1] if pred_recall_hist else 0.0)
        pred_fpr = (pred_fpr_hist[-1] if pred_fpr_hist else 0.0)
        pred_wrong_skip = (pred_wrong_skip_hist[-1] if pred_wrong_skip_hist else 0.0)
        pred_sparsity = (pred_sparsity_hist[-1] if pred_sparsity_hist else 0.0)
        pred_thr = getattr(self, '_pred_threshold', None)

        with open(log_file, "a") as f:
            last_err = self.error_history[-1] if self.error_history else 0.0
            last_adj, last_p, last_i, last_d = self.pid_history[-1] if self.pid_history else (0.0, 0.0, 0.0, 0.0)
            row = (
                f"{self.step},{current_sparsity:.4f},{current_quality:.4f},{self.threshold:.6f},"
                f"{last_err:.6f},{last_adj:.6f},{last_p:.6f},{last_i:.6f},{last_d:.6f},"
                f"{avg_sparsity:.4f},{avg_quality:.4f},{getattr(self, 'predictor_alpha', 0)},"
                f"{(pred_thr if pred_thr is not None else 0.0):.6f},{pred_prec:.4f},{pred_rec:.4f},{pred_fpr:.4f},{pred_wrong_skip:.4f},{pred_sparsity:.4f}\n"
            )
            f.write(row)


# ============================================================
# Predictor alpha controller (adaptive wrong-skip rate control)
# ============================================================


class PredictorThresholdController:
    """Direct threshold controller matching C++ implementation"""
    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx
        # Direct threshold value
        self.predictor_threshold = float(PRED_THRESHOLD_INIT)
        self.min_threshold = float(PRED_THRESHOLD_MIN)
        self.step = 0
        # Audit probability only (no forced audits)
        self.audit_prob = float(PRED_AUDIT_PROB)

        # Target wrong-skip share. In contrib mode this same
        # PID path is driven by the predictor's standalone output drift toward 1-CATSGR_GATE_TARGET.
        self.target_wrong_skip = (1.0 - CATSGR_GATE_TARGET) if CATSGR_SIGNAL == 'contrib' else float(TARGET_WRONG_SKIP)
        self._rl = RLThresholdController(self.target_wrong_skip, seed=10000 + layer_idx) if CATSGR_CONTROLLER == 'rl' else None
        # last driving signal fed to the PID (wrong_skip in baseline, output drift in contrib) — the
        # MIMD interval logic reads this so cadence adapts on the same signal the controller targets.
        self.last_signal = None
        # PID parameters matching C++ BufferedPID settings
        self.pid = BufferedPID(kp=1.0, ki=0.2, kd=0.15, buffer_size=5, max_age=5)
        self.error_history = deque(maxlen=200)
        self.pid_history = deque(maxlen=200)

        # Histories for logging
        self.p_true_hist = deque(maxlen=200)

        # Last known teacher sparsity (for non-audit steps)
        self.p_true_last = 0.01

        self.log_dir = "sparsity_logs"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

    def should_audit(self, force_check: bool = False) -> bool:
        # force_check bypasses probability gate
        if force_check:
            return True
        # Only probabilistic audits; no forced audits by step gap
        return (random.random() < self.audit_prob)

    def update_threshold(self, wrong_skip: float):
        # `wrong_skip` is the driving signal (true wrong-skip in baseline, output drift in contrib).
        self.last_signal = wrong_skip
        # Scale the wrong-skip error for the PID
        scaled_error = (self.target_wrong_skip - wrong_skip) * 10.0
        if CATSGR_CONTROLLER == 'rl':
            # wrong_skip is the predictor's driving signal (drift in contrib mode); target=target_wrong_skip
            push = self._rl.update(wrong_skip)
            adjustment = p_term = i_term = d_term = 0.0
            threshold_change = push * self.predictor_threshold * 0.010   # validated RL apply scale
            new_threshold = max(self.min_threshold, self.predictor_threshold + threshold_change)
            if abs(threshold_change) > 1e-6:
                self.predictor_threshold = new_threshold
        else:
            self.pid.add_sample(scaled_error, self.step)
            adjustment, p_term, i_term, d_term = self.pid.get_update(self.step)
            # Fixed step size update
            threshold_change = adjustment * self.predictor_threshold * 0.02
            new_threshold = max(self.min_threshold, self.predictor_threshold + threshold_change)
            # Only update if meaningful change
            if abs(threshold_change) > 1e-6:
                self.predictor_threshold = new_threshold

        self.error_history.append(scaled_error)
        self.pid_history.append((adjustment, p_term, i_term, d_term))

    def write_logs(self, pred_thr: float, precision: float, recall: float, fpr: float,
                   wrong_skip: float, pred_sparsity: float):
        layer_dir = os.path.join(self.log_dir, f"layer_{self.layer_idx}")
        if not os.path.exists(layer_dir):
            os.makedirs(layer_dir, exist_ok=True)
        log_file = os.path.join(layer_dir, "predictor.csv")

        if not os.path.exists(log_file):
            with open(log_file, "w") as f:
                f.write(
                    "Step,PredThr,PredPrec,PredRec,PredFPR,WrongSkip,PredSparsity,PTrue,Adj,P,I,D\n"
                )

        last_adj, last_p, last_i, last_d = self.pid_history[-1] if self.pid_history else (0.0, 0.0, 0.0, 0.0)
        with open(log_file, "a") as f:
            last_p_true = self.p_true_hist[-1] if self.p_true_hist else 0.0
            row = (
                f"{self.step},{pred_thr:.6f},{precision:.4f},{recall:.4f},{fpr:.4f},{wrong_skip:.4f},"
                f"{pred_sparsity:.4f},{last_p_true:.4f},{last_adj:.6f},{last_p:.6f},{last_i:.6f},{last_d:.6f}\n"
            )
            f.write(row)


# ====================
# Model blocks (LLAMA)
# ====================


@use_kernel_forward_from_hub("RMSNorm")
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
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
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
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

        self.activation_sparsity_controller = AdaptiveActivationSparsityController(layer_idx)
        self.enable_adaptive_sparsity = True

        # Predictor threshold controller (new)
        self._pred_ctrl = PredictorThresholdController(layer_idx)

        # Calibration S values per layer (3σ scheme): Outlier S, Big S, Middle S
        self._s_values = self._load_s_values()
        # Precomputed weight signs and big-weight mask; lazily initialized
        self._weight_signs = None
        self._gate_big_mask = None

        # Predictor threshold tuning (direct threshold approach)
        self.pred_precision_hist = deque(maxlen=200)
        self.pred_recall_hist = deque(maxlen=200)
        self.pred_fpr_hist = deque(maxlen=200)
        self.pred_wrong_skip_hist = deque(maxlen=200)
        self.pred_sparsity_hist = deque(maxlen=200)

        # Expose for activation logs
        ctrl = self.activation_sparsity_controller
        ctrl.pred_precision_hist = self.pred_precision_hist
        ctrl.pred_recall_hist = self.pred_recall_hist
        ctrl.pred_fpr_hist = self.pred_fpr_hist
        ctrl.pred_wrong_skip_hist = self.pred_wrong_skip_hist
        ctrl.pred_sparsity_hist = self.pred_sparsity_hist
        ctrl._pred_threshold = self._pred_ctrl.predictor_threshold

        # Stats
        self.total_tokens = 0
        self.total_sparsity = 0.0
        self.avg_quality = 1.0

    def _ensure_weight_artifacts(self):
        if self._weight_signs is None or self._gate_big_mask is None:
            with torch.no_grad():
                w = self.gate_proj.weight
                weight_32 = w.to(torch.float32)
                weight_signs = torch.sign(weight_32)
                w_mean = weight_32.mean(dim=-1, keepdim=True)
                w_std = weight_32.std(dim=-1, keepdim=True)
                w_q1 = w_mean - 0.6745 * w_std
                w_q3 = w_mean + 0.6745 * w_std
                big_mask = (weight_32 < w_q1) | (weight_32 > w_q3)
                self._weight_signs = weight_signs.detach()
                self._gate_big_mask = big_mask.detach()

    def _load_s_values(self):
        # stock path is CWD-relative (paper ran from a dir containing gr+t/). Resolve robustly:
        # env override first, then the repo's canonical location, then the stock relative path.
        candidates = [p for p in [
            CATSGR_S_VALUES,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "3std_averages_s_values.csv"),
            os.path.join("gr+t", "calibration", "3std_averages_s_values.csv"),
        ] if p]
        csv_path = next((p for p in candidates if os.path.exists(p)), candidates[-1])
        if self.layer_idx == 0:
            print(f"[ACAS-CATSGR] s-values source: {csv_path} (exists={os.path.exists(csv_path)})")
        default = (30.0, 15.5, 4.0)
        if not os.path.exists(csv_path):
            return default
        try:
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        if int(row.get("Layer", -1)) == self.layer_idx:
                            return (
                                float(row.get("Outlier S", default[0])),
                                float(row.get("Big S", default[1])),
                                float(row.get("Middle S", default[2])),
                            )
                    except Exception:
                        continue
        except Exception:
            return default
        return default

    def _compute_proxy_score(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_weight_artifacts()
        weight_signs = self._weight_signs
        gate_big_mask = self._gate_big_mask

        with torch.no_grad():
            x_32 = x.to(torch.float32)
            x_mean = x_32.mean(dim=2, keepdim=True)
            x_std = x_32.std(dim=2, keepdim=True)
            x_q1 = x_mean - 0.6745 * x_std
            x_q3 = x_mean + 0.6745 * x_std
            x_big = (x_32 < x_q1) | (x_32 > x_q3)
            x_out_pos = x_mean + 3.0 * x_std
            x_out_neg = x_mean - 3.0 * x_std
            x_outlier = (x_32 < x_out_neg) | (x_32 > x_out_pos)

            x_signs = torch.sign(x_32)
            big_w_signs = weight_signs * gate_big_mask
            small_w_signs = weight_signs * (~gate_big_mask)

            big_x_signs = x_signs * x_big
            small_x_signs = x_signs * (~x_big)
            outlier_x_signs = x_signs * x_outlier

            p_big = torch.matmul(big_x_signs, big_w_signs.transpose(0, 1).to(x_32.dtype))
            p_mid = torch.matmul(big_x_signs, small_w_signs.transpose(0, 1).to(x_32.dtype))
            p_mid = p_mid + torch.matmul(small_x_signs, big_w_signs.transpose(0, 1).to(x_32.dtype))
            p_small = torch.matmul(small_x_signs, small_w_signs.transpose(0, 1).to(x_32.dtype))
            p_out = torch.matmul(outlier_x_signs, weight_signs.transpose(0, 1).to(x_32.dtype))

            s_out, s_big, s_mid = self._s_values
            total = p_out * s_out + p_big * s_big + p_mid * s_mid + p_small
            proxy = total.abs()
        return proxy.to(x.dtype)

    def _update_predictor_threshold(self, proxy_mag: torch.Tensor, activated: torch.Tensor, activation_threshold: float,
                                    up_output=None, down_proj=None, residual=None):
        with torch.no_grad():
            act_abs = activated.abs().float()
            true_low = act_abs <= activation_threshold
            p_true = true_low.float().mean().item()

            # Use direct threshold
            pred_low = proxy_mag <= self._pred_ctrl.predictor_threshold
            tp = (pred_low & true_low).sum().item()
            fp = (pred_low & (~true_low)).sum().item()
            fn = ((~pred_low) & true_low).sum().item()
            tn = ((~pred_low) & (~true_low)).sum().item()
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            fpr = fp / (fp + tn + 1e-8)
            pred_sparsity = pred_low.float().mean().item()
            wrong_skip = fpr * (1.0 - p_true)

            # Update histories
            self.pred_precision_hist.append(precision)
            self.pred_recall_hist.append(recall)
            self.pred_fpr_hist.append(fpr)
            self.pred_wrong_skip_hist.append(wrong_skip)
            self.pred_sparsity_hist.append(pred_sparsity)

            # Sync threshold to activation controller for unified logging
            self.activation_sparsity_controller._pred_threshold = self._pred_ctrl.predictor_threshold

            # Update predictor threshold PID and logs.
            # contrib: drive the PID by the predictor's STANDALONE output drift (zero the
            # predicted-low neurons, measure MLP-output movement vs ||residual||) instead of wrong-skip.
            if CATSGR_SIGNAL == 'contrib' and up_output is not None and down_proj is not None and residual is not None:
                act_pred = activated.clone()
                act_pred[pred_low] = 0
                y_dense = down_proj(activated * up_output)
                y_pred  = down_proj(act_pred * up_output)
                pid_signal = ((y_dense - y_pred).norm() / (residual.norm() + 1e-8)).item()  # drift, lower=better
            else:
                pid_signal = wrong_skip
            self._pred_ctrl.p_true_hist.append(p_true)
            self._pred_ctrl.update_threshold(pid_signal)
            self._pred_ctrl.write_logs(
                pred_thr=self._pred_ctrl.predictor_threshold,
                precision=precision,
                recall=recall,
                fpr=fpr,
                wrong_skip=wrong_skip,
                pred_sparsity=pred_sparsity,
            )

    def forward(self, x, force_check: bool = False, residual=None):
        if x.size(1) == 1 and self.enable_adaptive_sparsity:
            # Compute up branch first (unchanged)
            up_output = self.up_proj(x)

            # Compute proxy without loading gate weights
            proxy_mag = self._compute_proxy_score(x)

            # Audit decision for predictor controller — force_check bypasses probability
            audit = self._pred_ctrl.should_audit(force_check=force_check)

            if audit:
                # Full compute for teacher and metrics. up_output/down_proj/residual let each knob
                # measure its STANDALONE output-contribution drift when ACAS_CATSGR_SIGNAL='contrib'.
                gate_output = self.gate_proj(x)
                activated_full = self.act_fn(gate_output)
                activation_threshold, activation_sparsity, activation_quality = self.activation_sparsity_controller.update(
                    activated_full, do_quality_check=True, force_check=force_check,
                    up_output=up_output, down_proj=self.down_proj, residual=residual
                )
                activation_mask = activated_full.abs() > activation_threshold

                # Update predictor threshold (and alpha PID)
                self._update_predictor_threshold(proxy_mag, activated_full, activation_threshold,
                                                 up_output=up_output, down_proj=self.down_proj, residual=residual)
            else:
                # No audit: just use current threshold (no quantile calculation needed)
                # We still need an activation threshold for masking; reuse last known
                activation_threshold = self.activation_sparsity_controller.threshold if self.activation_sparsity_controller.threshold is not None else 0.0
                gate_output = self.gate_proj(x)
                activated_full = self.act_fn(gate_output)
                activation_mask = activated_full.abs() > activation_threshold
                activation_sparsity = (~activation_mask).float().mean().item()
                activation_quality = 1.0

            # Apply predictor: zero out rows predicted-low by proxy threshold
            pred_low = proxy_mag <= self._pred_ctrl.predictor_threshold
            activated = activated_full.clone()
            activated[pred_low] = 0

            # Then apply teacher threshold mask
            activated_sparse = activated * activation_mask

            down_proj = self.down_proj(activated_sparse * up_output)

            # Track statistics
            self.total_tokens += 1
            self.total_sparsity += activation_sparsity
            self.avg_quality = 0.99 * self.avg_quality + 0.01 * activation_quality

            # Update controller steps and last p_true
            self._pred_ctrl.step += 1
            if audit:
                # true_low computed inside _update_predictor_threshold; reuse latest
                p_true_last = self._pred_ctrl.p_true_hist[-1] if self._pred_ctrl.p_true_hist else self._pred_ctrl.p_true_last
                self._pred_ctrl.p_true_last = p_true_last

        else:
            # Training/prefill path unchanged
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
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
        self.mlp = LlamaMLP(config, layer_idx)
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
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        force_check: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
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

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        # pass the pre-MLP residual stream so both contrib knobs can normalize by ||residual||
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

        # MIMD interval state — one entry per layer
        num_layers = config.num_hidden_layers
        self._mimd_interval = [MIMD_INIT_INTERVAL] * num_layers
        self._mimd_tokens_since = [0] * num_layers
        self._global_step = 0

        # Tracking infrastructure
        self._track_records = []
        self._track_seq_summaries = []
        self._track_seq_id = 0
        self._track_token_in_seq = 0
        self._track_seq_layer_buf = {i: [] for i in range(num_layers)}  # per-layer per-token snapshots

        atexit.register(self.dump_tracking)

        self.post_init()

    def dump_tracking(self):
        """Write threshold_trace.csv, seq_summaries.json, and tracking_stats.json."""
        # Flush any open sequence
        if self._track_token_in_seq > 0:
            self._flush_seq_summary()

        out_dir = os.environ.get("ACAS_TRACK_DIR", "acas_tracking")
        os.makedirs(out_dir, exist_ok=True)

        # ---- threshold_trace.csv ----
        num_layers = len(self.layers)
        csv_path = os.path.join(out_dir, "threshold_trace.csv")
        layer_cols = []
        for i in range(num_layers):
            layer_cols += [
                f"threshold_L{i}", f"quality_L{i}", f"interval_L{i}",
                f"pred_threshold_L{i}", f"wrong_skip_L{i}",
            ]
        fieldnames = ["step", "seq_id", "token_in_seq"] + layer_cols
        try:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for rec in self._track_records:
                    writer.writerow(rec)
        except Exception as e:
            logger.warning(f"[ACAS] dump_tracking csv error: {e}")

        # ---- seq_summaries.json ----
        summaries_path = os.path.join(out_dir, "seq_summaries.json")
        try:
            with open(summaries_path, "w") as f:
                json.dump(self._track_seq_summaries, f, indent=2)
        except Exception as e:
            logger.warning(f"[ACAS] dump_tracking seq_summaries error: {e}")

        # ---- tracking_stats.json ----
        stats_path = os.path.join(out_dir, "tracking_stats.json")
        try:
            stats = {
                "global_step": self._global_step,
                "total_sequences": self._track_seq_id,
                "total_records": len(self._track_records),
                "final_mimd_intervals": self._mimd_interval,
                "final_thresholds": [
                    self.layers[i].mlp.activation_sparsity_controller.threshold
                    for i in range(num_layers)
                ],
                "final_pred_thresholds": [
                    self.layers[i].mlp._pred_ctrl.predictor_threshold
                    for i in range(num_layers)
                ],
            }
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)
        except Exception as e:
            logger.warning(f"[ACAS] dump_tracking stats error: {e}")

    def _flush_seq_summary(self):
        """Summarise current sequence and append to _track_seq_summaries."""
        num_layers = len(self.layers)
        summary = {
            "seq_id": self._track_seq_id,
            "length": self._track_token_in_seq,
            "layers": {},
        }
        for i in range(num_layers):
            snaps = self._track_seq_layer_buf[i]
            if snaps:
                summary["layers"][str(i)] = {
                    "avg_threshold": float(np.mean([s["threshold"] for s in snaps])),
                    "avg_quality": float(np.mean([s["quality"] for s in snaps])),
                    "avg_interval": float(np.mean([s["interval"] for s in snaps])),
                    "avg_pred_threshold": float(np.mean([s["pred_threshold"] for s in snaps])),
                    "avg_wrong_skip": float(np.mean([s["wrong_skip"] for s in snaps])),
                }
            else:
                summary["layers"][str(i)] = {}
        self._track_seq_summaries.append(summary)
        # Reset per-seq buffer
        for i in range(num_layers):
            self._track_seq_layer_buf[i] = []

    def _record_tracking_step(self, layer_snapshots):
        """Append one row to _track_records and update per-seq buffers."""
        num_layers = len(self.layers)
        rec = {
            "step": self._global_step,
            "seq_id": self._track_seq_id,
            "token_in_seq": self._track_token_in_seq,
        }
        for i, snap in enumerate(layer_snapshots):
            rec[f"threshold_L{i}"] = snap["threshold"]
            rec[f"quality_L{i}"] = snap["quality"]
            rec[f"interval_L{i}"] = snap["interval"]
            rec[f"pred_threshold_L{i}"] = snap["pred_threshold"]
            rec[f"wrong_skip_L{i}"] = snap["wrong_skip"]
            self._track_seq_layer_buf[i].append(snap)
        self._track_records.append(rec)

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

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds

        # Sequence boundary detection: new sequence when no prior KV cache
        seq_length = inputs_embeds.shape[1]
        is_generation_step = (seq_length == 1)
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        is_new_sequence = (past_len == 0)

        if is_new_sequence and self._track_token_in_seq > 0:
            # Save summary of previous sequence before starting new one
            self._flush_seq_summary()
            self._track_seq_id += 1
            self._track_token_in_seq = 0

        # Per-layer forward with MIMD interval control
        layer_snapshots = []
        for layer_idx, decoder_layer in enumerate(self.layers):
            # Determine force_check for this layer based on MIMD interval
            force_check = False
            if is_generation_step:
                self._mimd_tokens_since[layer_idx] += 1
                if self._mimd_tokens_since[layer_idx] >= self._mimd_interval[layer_idx]:
                    force_check = True
                    self._mimd_tokens_since[layer_idx] = 0

            hidden_states = decoder_layer(
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                force_check=force_check,
                **kwargs,
            )

            # MIMD interval adjustment after a forced check
            if force_check and is_generation_step:
                mlp = decoder_layer.mlp
                act_ctrl = mlp.activation_sparsity_controller
                # Activation quality error vs the controller's ACTUAL target (contrib or baseline).
                act_quality = act_ctrl.quality_history[-1] if act_ctrl.quality_history else act_ctrl.target_quality
                act_err = abs(act_quality - act_ctrl.target_quality)
                # Predictor error: use the controller's last driving signal vs its target so the
                # cadence adapts on the SAME signal the predictor PID optimizes (drift in contrib mode).
                pred_ctrl = mlp._pred_ctrl
                pred_sig = pred_ctrl.last_signal if getattr(pred_ctrl, 'last_signal', None) is not None else pred_ctrl.target_wrong_skip
                pred_err = abs(pred_sig - pred_ctrl.target_wrong_skip)
                # Combined error: conservative max
                combined_err = max(act_err, pred_err)

                if combined_err > MIMD_ERROR_THRESHOLD:
                    new_interval = int(self._mimd_interval[layer_idx] * MIMD_DECREASE)
                else:
                    new_interval = int(self._mimd_interval[layer_idx] * MIMD_INCREASE)
                self._mimd_interval[layer_idx] = max(MIMD_MIN_INTERVAL, min(MIMD_MAX_INTERVAL, new_interval))

            # Snapshot per-layer state for tracking
            if is_generation_step:
                mlp = decoder_layer.mlp
                act_ctrl = mlp.activation_sparsity_controller
                snap = {
                    "threshold": act_ctrl.threshold if act_ctrl.threshold is not None else 0.0,
                    "quality": act_ctrl.quality_history[-1] if act_ctrl.quality_history else 1.0,
                    "interval": self._mimd_interval[layer_idx],
                    "pred_threshold": mlp._pred_ctrl.predictor_threshold,
                    "wrong_skip": mlp.pred_wrong_skip_hist[-1] if mlp.pred_wrong_skip_hist else 0.0,
                }
                layer_snapshots.append(snap)

        # Record tracking after all layers (generation steps only)
        if is_generation_step:
            self._record_tracking_step(layer_snapshots)
            self._track_token_in_seq += 1
            self._global_step += 1

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
    base_model_prefix = "transformer"


class LlamaForTokenClassification(GenericForTokenClassification, LlamaPreTrainedModel): ...


__all__ = [
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaPreTrainedModel",
    "LlamaForSequenceClassification",
    "LlamaForQuestionAnswering",
    "LlamaForTokenClassification",
]
