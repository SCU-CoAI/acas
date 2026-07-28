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
""" PyTorch LLaMA model."""
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import AttentionMaskConverter, _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    logging,
    replace_return_docstrings,
)
from transformers.utils.import_utils import is_torch_fx_available
from .configuration_sparsellama import SparseLlamaConfig

import numpy as np
import csv


if is_flash_attn_2_available():
    try:
        from flash_attn import flash_attn_func, flash_attn_varlen_func
        from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
    except Exception:
        pass


# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"

import os

NUM_LAYERS = 32
MAX_ALPHA = 15000
MIN_ALPHA = 5000

INIT_ALPHA = 10000

TARGET_P = 0.997

proportion_list = [
85.61433623900221,15.523354727672332,3.978918085920551,
54.89820120106032,15.208394805399521,3.933713673260931,
38.48385603041713,15.317835577405475,3.930136544874956,
47.511050333914795,15.250526033480362,3.926523001948811,
56.08373787353046,15.163743496770074,3.916299967874994,
75.6678313078744,15.25108835537739,3.9063718277549175,
72.3729662065994,15.265413329225979,3.905028628195499,
66.8056798635457,15.277871608173811,3.9067184242023014,
68.6940251826593,15.227856423237181,3.900154085656198,
70.54818485143115,15.239834082663904,3.900500863713783,
71.54021684586368,15.215733959421321,3.896830568810752,
71.59332910477399,15.25171622157222,3.8995601787626266,
79.40334678014733,15.286763133819825,3.9024800625012612,
77.91022468029038,15.287899955271804,3.9048852561390013,
68.96848784841666,15.327329844756454,3.9100964988581244,
75.26644584681915,15.303856386404753,3.910150680524277,
53.68015858267448,15.32483949722231,3.9143610296971887,
93.13391803307195,15.345897990975686,3.905787026626225,
110.82023290598704,15.36346124071264,3.9072983106609005,
77.53367115032218,15.359327179285506,3.909032528588298,
91.2089062383907,15.324805980890309,3.8975533845212493,
79.31977256740761,15.37774765629847,3.903711306291901,
93.60031321931224,15.34684644161128,3.8958009607725734,
87.87371245837707,15.345874703526498,3.897102644730032,
82.07775249811819,15.393592377168833,3.8993739147387916,
80.03722164148171,15.456361533734654,3.9067977778527774,
80.90649178476853,15.29571319441261,3.8880207918271132,
83.40322434273706,15.44682671879188,3.907625855460164,
77.20889976849125,15.572933924686586,3.9264753742795873,
68.42784433266162,15.62494562148676,3.937982166883932,
58.35924971869011,15.836915313271207,3.9714310413979503,
85.7118187578991,15.143494897006256,3.9003656587928552,
]

# === ACAS-Grasp quality-signal selection. Defaults = precision ACAS-Grasp. ===
#   ACAS_GRASP_SIGNAL : "precision" (default) | "l2"       ACAS_GRASP_TARGET : metric target
#   ACAS_GRASP_MODE   : "equalize" (default) | "ceiling"   ACAS_GRASP_FREEZE=1 : pin alphas (Grasp baseline)
#   ACAS_GRASP_INIT_ALPHAS : warm-start json    ACAS_GRASP_KP/KI/KD : PID gains
GRASP_SIGNAL = os.environ.get('ACAS_GRASP_SIGNAL', 'precision').lower()
GRASP_TARGET = float(os.environ.get('ACAS_GRASP_TARGET', str(TARGET_P) if GRASP_SIGNAL == 'precision' else '0.95'))
GRASP_MODE = os.environ.get('ACAS_GRASP_MODE', 'equalize').lower()
GRASP_FREEZE = os.environ.get('ACAS_GRASP_FREEZE', '0') == '1'
GRASP_INIT_JSON = os.environ.get('ACAS_GRASP_INIT_ALPHAS', '')
GRASP_KP = float(os.environ.get('ACAS_GRASP_KP', '1.0'))
GRASP_KI = float(os.environ.get('ACAS_GRASP_KI', '0.2'))
GRASP_KD = float(os.environ.get('ACAS_GRASP_KD', '0.15'))
# Depth-weighted target: early layers get a LOWER (more permissive) quality target, late layers
# HIGHER (stricter). Motivation: contrib metric over-penalizes early layers (their drift looks big
# vs the still-small early residual stream) even though those skips are output-safe (Grasp proves it),
# leaving early-layer speed on the table. SLOPE=0 -> flat target (unchanged default).
#   target_layer = GRASP_TARGET + SLOPE * (layer/(L-1) - 0.5)   (mean target stays ~GRASP_TARGET)
GRASP_TARGET_SLOPE = float(os.environ.get('ACAS_GRASP_TARGET_SLOPE', '0.0'))
# controller-swap ablation: 'pid' (default, byte-identical) | 'rl' (tabular Q-learning, same signal)
GRASP_CONTROLLER = os.environ.get('ACAS_GRASP_CONTROLLER', 'pid').lower()

_ALPHA_LOG_DIR = os.environ.get('ACAS_TRACK_DIR', 'alphas')
if not os.path.exists(_ALPHA_LOG_DIR):
    os.makedirs(_ALPHA_LOG_DIR)

# log BOTH metrics every tick (quality/l2_error/skip_ratio appended at end; old columns unchanged)
_LOG_HDR = "step\talpha\tprecision\terror\tadjustment\tp_term\ti_term\td_term\tquality\tl2_error\tskip_ratio\tcontrib_quality\n"
for i in range(NUM_LAYERS):
    with open(os.path.join(_ALPHA_LOG_DIR, f"alpha_{i}.log"), "a") as f:
        f.write(_LOG_HDR)
with open(os.path.join(_ALPHA_LOG_DIR, "alpha_all.log"), "a") as f:
    f.write("step\til\talpha\tprecision\terror\tadjustment\tp_term\ti_term\td_term\tquality\tl2_error\tskip_ratio\tcontrib_quality\n")


def logit(x):
    x = np.clip(x, 1e-10, 1 - 1e-10)
    return np.log(x / (1 - x))

def calculate_error(p, target_p, sensitivity=5.0):
    # ensure probabilities are within a safe range
    if p is None:
        p = target_p
    else:
        p = np.clip(p, 1e-10, 1 - 1e-10)
    target_p = np.clip(target_p, 1e-10, 1 - 1e-10)

    # calculate the logit difference
    diff = logit(target_p) - logit(p)

    # custom sigmoid function to scale the error
    return np.sign(diff) * (2 / (1 + np.exp(-sensitivity * np.abs(diff))) - 1)

import random
from collections import deque

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
        valid_samples = []
        # just collect valid samples without removing old ones from buffer
        for sample in self._buffer:
            if current_step - sample.step <= self._max_age:
                valid_samples.append(sample)

        if len(valid_samples) == 0:
            return 0.0

        # get current error from last valid sample
        current_error = valid_samples[-1].error

        # calculate integral using all valid samples (including current)
        integral = 0.0
        for sample in valid_samples:
            integral += sample.error

        # calculate derivative if we have at least 2 samples
        derivative = 0.0
        if len(valid_samples) >= 2:
            derivative = current_error - valid_samples[len(valid_samples) - 2].error

        p_term = self._kp * current_error
        i_term = self._ki * integral
        d_term = self._kd * derivative

        adjustment = p_term + i_term + d_term

        return adjustment, p_term, i_term, d_term

def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    warnings.warn(
        "Calling `transformers.models.llama.modeling_llama._prepare_4d_attention_mask` is deprecated and will be removed in v4.37. Use `transformers.modeling_attn_mask_utils.AttentionMaskConverter._prepare_4d_attention_mask"
    )
    return AttentionMaskConverter._prepare_4d_attention_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)


def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    warnings.warn(
        "Calling `transformers.models.llama.modeling_llama._make_causal_mask` is deprecated and will be removed in v4.37. Use `transformers.models.llama.modeling_llama.AttentionMaskConverter._make_causal_mask"
    )
    return AttentionMaskConverter._make_causal_mask(
        input_ids_shape=input_ids_shape, dtype=dtype, device=device, past_key_values_length=past_key_values_length
    )


# === RL controller: tabular Q-learning drop-in for the PID alpha update. ============
# Enabled via ACAS_GRASP_CONTROLLER=rl. Same information as the PID (metric vs target) + same knob;
# learns the control law from reward = -|metric - target|. Validated within 0.3% of the PID setpoint
# (validated offline). Caller scales push: alpha += round(push*_alpha_unit), _alpha_unit=10000/(hidden/2).
class RLThresholdController:
    _SIGMA_FALLBACK = 0.01   # used only until the running window has a few samples
    _SIGMA_WINDOW = 40       # checks for the ONLINE per-check noise estimate sigma_q (no hand-set scale)
    _ACTIONS = (-8.0, -3.0, -1.0, 0.0, 1.0, 3.0, 8.0)   # decision-boundary units; apply scales by _alpha_unit
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


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
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
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SparseLlamaMLP(nn.Module):
    def __init__(self, config: SparseLlamaConfig, layer_index):
        super().__init__()
        self.config = config
        self.layer_index=  layer_index
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.gate_proj_s = None
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if config.hidden_act in ACT2FN:
            self.act_fn = ACT2FN[config.hidden_act]
        elif config.hidden_act == "shiftrelu":
            def shifted_relu(x):
                return torch.nn.functional.relu(x - config.hidden_act_param)
            self.act_fn = shifted_relu
        elif config.hidden_act == "fatrelu":
            def fat_relu(x):
                new_x = torch.zeros_like(x)
                mask = torch.ge(x, config.hidden_act_param)
                new_x[mask] = x[mask]
                return new_x
            self.act_fn = fat_relu
        else:
            raise NotImplementedError(f"Unsupported activation function: {config.hidden_act}")
        self.h1 = None
        self.coef = None
        self.mean = None
        self.std = None
        self.threshold = None
        self.proportion = proportion_list[layer_index*3:(layer_index+1)*3]

    def forward(self, x, alpha, update):
        signal = None
        if self.gate_proj_s is None:
            with torch.no_grad():
                gate_32 = self.gate_proj.weight.to(torch.float32)
                gate_mean = torch.mean(gate_32, dim=-1)
                gate_std = torch.std(gate_32, dim=-1)
                gate_Q1 = (gate_mean - 0.6745 * gate_std).unsqueeze(1)
                gate_Q3 = (gate_mean + 0.6745 * gate_std).unsqueeze(1)
                self.gate_proj_s = (gate_32 < gate_Q1) | (gate_32 > gate_Q3)

        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)

        # 큰 값, 중간 값, 작은 값, 이상치로 구분하여 계산 - R
        elif x.size(1) == 1:
            # 크기 비트를 보는 부분
            x_32 = x.to(torch.float32)
            x_mean = torch.mean(x_32, dim=2)
            x_std = torch.std(x_32, dim=2)
            x_Q1 = (x_mean - 0.6745 * x_std).unsqueeze(2)
            x_Q3 = (x_mean + 0.6745 * x_std).unsqueeze(2)
            x_outlier_p = (x_mean + 6 * x_std).unsqueeze(2)
            x_outlier_n = (x_mean - 6 * x_std).unsqueeze(2)
            x_size_mask = (x_32 < x_Q1) | (x_32 > x_Q3)
            x_outlier_mask = (x_32 < x_outlier_n) | (x_32 > x_outlier_p)
            # 끝

            weight_signs = torch.sign(self.gate_proj.weight)

            big_x_signs = torch.sign(x) * x_size_mask
            big_weights_signs = weight_signs * self.gate_proj_s

            small_x_signs = torch.sign(x) * ~x_size_mask
            small_weights_signs = weight_signs * ~self.gate_proj_s

            outlier_x_signs = torch.sign(x) * x_outlier_mask

            big_signs_product = torch.matmul(big_x_signs, big_weights_signs.T).to(dtype=torch.float32)
            middle_signs_product = torch.matmul(big_x_signs, small_weights_signs.T).to(dtype=torch.float32)
            middle_signs_product += torch.matmul(small_x_signs, big_weights_signs.T).to(dtype=torch.float32)
            small_signs_product = torch.matmul(small_x_signs, small_weights_signs.T).to(dtype=torch.float32)
            outlier_signs_product = torch.matmul(outlier_x_signs, weight_signs.T).to(dtype=torch.float32)

            total_product = outlier_signs_product * self.proportion[0] + big_signs_product * self.proportion[1] + middle_signs_product * self.proportion[2] + small_signs_product

            scaling_value = 0
            #if self.layer_index < 20:
            #    scaling_value = -74
            val = 1 - (alpha/10000)
            im = x.size(2) / 2 * val
            if im > 0:
                scaling_value = int(im + 0.5)
            elif im < 0:
                scaling_value = int(im - 0.5)
            mask = torch.ge(total_product, scaling_value)

            actual = self.gate_proj(x)

            # print("actual size:", actual.size(2))
            # print(f"il: {self.layer_index}, alpha: {alpha}, scaling_value: {scaling_value}")

            # On dense-check tokens, compute BOTH quality signals (vectorized; no skip here).
            if update:
                b = torch.nn.functional.relu(actual)            # dense post-activation

                # input-side: Grasp precision = TP/(TP+FP), vectorized
                # (bit-identical to the old per-element Python loop; mask True = keep)
                skip = ~mask
                truly_zero = actual <= 0
                tp = (skip & truly_zero).sum().item()
                fp = (skip & ~truly_zero).sum().item()
                precision = tp / (tp + fp + 1e-6)

                # output-side: normalized L2 drift (same formula as ACAS-CATS)
                l2_error = ((b - b * mask).norm() / (b.norm() + 1e-8)).item()
                quality = max(0.0, min(1.0, 1.0 - l2_error))
                skip_ratio = skip.float().mean().item()

                # output-CONTRIBUTION drift: ||y_dense - y_sparse|| at the MLP OUTPUT (post up/down proj,
                # so it counts only drift that survives the projections). The decoder divides this by
                # ||residual|| -> drift relative to the residual stream the layer feeds (closer to logit impact).
                _up = self.up_proj(x)
                y_delta_norm = (self.down_proj(b * _up) - self.down_proj((b * mask) * _up)).norm().item()

                signal = {"precision": precision, "quality": quality, "l2_error": l2_error,
                          "skip_ratio": skip_ratio, "y_delta_norm": y_delta_norm}
            else:
                a = torch.nn.functional.relu(actual)
                b = torch.zeros_like(a)
                b[mask] = a[mask]

            down_proj = self.down_proj(b * self.up_proj(x))
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj, signal


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


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: SparseLlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaFlashAttention2(LlamaAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # LlamaFlashAttention2 attention does not support output_attentions
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # TODO: llama does not have dropout in the config??
        # It is recommended to use dropout with FA according to the docs
        # when training.
        dropout_rate = 0.0  # if not self.training else self.attn_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            # Handle the case where the model is quantized
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=self.is_causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=self.is_causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: SparseLlamaConfig, layer_index):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = (
            LlamaAttention(config=config)
            if not getattr(config, "_flash_attn_2_enabled", False)
            else LlamaFlashAttention2(config=config)
        )
        self.mlp = SparseLlamaMLP(config, layer_index)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        alpha: Optional[int] = 0,
        update: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, signal = self.mlp(hidden_states, alpha, update)
        if signal is not None:
            # contribution drift relative to the residual stream the MLP output is added to
            # (output-impact proxy: 1 = no impact on the stream). residual = pre-MLP hidden state.
            _rn = residual.norm().item() + 1e-8
            signal["contrib_quality"] = max(0.0, min(1.0, 1.0 - signal["y_delta_norm"] / _rn))
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs, signal


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class SparseLlamaPreTrainedModel(PreTrainedModel):
    config_class = SparseLlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, decoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class SparseLlamaModel(SparseLlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: SparseLlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
        self.pid = [BufferedPID(GRASP_KP, GRASP_KI, GRASP_KD, 5, 5) for _ in range(config.num_hidden_layers)]
        self.rl = [RLThresholdController(GRASP_TARGET, seed=i) for i in range(config.num_hidden_layers)] if GRASP_CONTROLLER == 'rl' else None
        self._alpha_unit = 10000.0 / (config.hidden_size / 2)   # alpha step moving the integer decision boundary by 1
        self.alphas = [0] * config.num_hidden_layers
        for i in range(config.num_hidden_layers):
            self.alphas[i] = INIT_ALPHA
        if GRASP_INIT_JSON:
            import json as _json
            with open(GRASP_INIT_JSON) as _f:
                _wa = _json.load(_f)
            for i in range(config.num_hidden_layers):
                self.alphas[i] = int(_wa[i] if isinstance(_wa, list) else _wa[str(i)])
            print(f"[ACAS-GRASP] warm-started alphas from {GRASP_INIT_JSON} (mean={sum(self.alphas)/len(self.alphas):.0f})")
        if GRASP_FREEZE:
            print("[ACAS-GRASP] FREEZE mode: alphas fixed; metrics measured + logged only")
        self.step = 0
        # NOTE: canonical ACAS-GR/SI use a flat ~0.5% per-token dense-check (no convergence warmup).
        # Convergence-warmup tracking removed 2026-06-01 to match SI and the paper (4090.py).


    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if getattr(self.config, "_flash_attn_2_enabled", False):
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        signals = [None] * self.config.num_hidden_layers
        # Flat ~0.5% per-token dense-check, shared across all layers — canonical ACAS-GR/SI scheme.
        _do_update = random.randint(0, 999) < 5
        updates = [_do_update] * self.config.num_hidden_layers

        # PROBE hook: force every layer dense or sparse (set self._force = 'dense' | 'sparse').
        # 'dense' -> all update=True (no skip); 'sparse' -> all update=False (skip @ current alpha).
        _f = getattr(self, "_force", None)
        if _f == "dense":
            updates = [True] * self.config.num_hidden_layers
        elif _f == "sparse":
            updates = [False] * self.config.num_hidden_layers

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs, signal = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    alpha=self.alphas[idx],
                    update=updates[idx],
                )

                if signal is not None:
                    signals[idx] = signal

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if any(x is not None for x in signals):
            for idx, signal in enumerate(signals):
                # Only process and log if this layer produced a signal this step
                if signal is not None:
                    # select driving metric; sign unchanged (alpha higher = more conservative)
                    if GRASP_SIGNAL == "precision":
                        metric = signal["precision"]
                    elif GRASP_SIGNAL == "contrib":
                        metric = signal["contrib_quality"]   # output-contribution drift (residual-relative)
                    else:
                        metric = signal["quality"]           # local h1 L2 drift
                    # depth-weighted target: looser (lower) early, stricter (higher) late
                    _L = self.config.num_hidden_layers
                    target_l = GRASP_TARGET + GRASP_TARGET_SLOPE * (idx / max(_L - 1, 1) - 0.5)
                    target_l = min(0.999, max(0.5, target_l))
                    error = calculate_error(metric, target_l)
                    if GRASP_MODE == "ceiling" and not GRASP_FREEZE:
                        error = max(0.0, error)   # protect-only: never free safe layers
                    if GRASP_CONTROLLER == 'rl':
                        push = self.rl[idx].update(metric)
                        p_term = i_term = d_term = 0.0
                        adjustment = 0 if GRASP_FREEZE else int(round(push * self._alpha_unit))   # 1 unit = 10000/(hidden/2)
                        self.alphas[idx] += adjustment
                    else:
                        self.pid[idx].add_sample(error, self.step)
                        adjustment, p_term, i_term, d_term = self.pid[idx].get_update(self.step)
                        if adjustment > 0:
                            adjustment += 0.5
                        else:
                            adjustment -= 0.5
                        if GRASP_FREEZE:
                            adjustment = 0   # pin alphas (Grasp baseline / frozen measurement)
                        self.alphas[idx] += int(adjustment)

                    if self.alphas[idx] > MAX_ALPHA:
                        self.alphas[idx] = MAX_ALPHA
                    elif self.alphas[idx] < MIN_ALPHA:
                        self.alphas[idx] = MIN_ALPHA

                    _p, _q, _l2, _sr = signal["precision"], signal["quality"], signal["l2_error"], signal["skip_ratio"]
                    _cq = signal.get("contrib_quality", -1.0)
                    with open(os.path.join(_ALPHA_LOG_DIR, f"alpha_{idx}.log"), "a") as f:
                        f.write(f"{self.step}\t{self.alphas[idx]}\t{_p}\t{error}\t{adjustment}\t{p_term}\t{i_term}\t{d_term}\t{_q}\t{_l2}\t{_sr}\t{_cq}\n")

                    with open(os.path.join(_ALPHA_LOG_DIR, "alpha_all.log"), "a") as f:
                        f.write(f"{self.step}\t{idx}\t{self.alphas[idx]}\t{_p}\t{error}\t{adjustment}\t{p_term}\t{i_term}\t{d_term}\t{_q}\t{_l2}\t{_sr}\t{_cq}\n")
            self.step += 1

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class SparseLlamaForCausalLM(SparseLlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = SparseLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = SparseLlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`SparseLlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class SparseLlamaForSequenceClassification(SparseLlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = SparseLlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1).to(
                    logits.device
                )
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
