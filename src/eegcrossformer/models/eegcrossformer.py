"""EEGCrossformer: dual-band EEG decoding with token-wise scalar fusion and Gain RTUR.

Architecture (``forward``)
    A filtered two-band trial ``[B, 2, C, T]`` is split into a low- and a
    high-band view. Each view passes through its own log-power tokenizer and a
    two-block gated-linear-attention (GLA) encoder, giving ``N`` aligned tokens
    of width 20 per band. A linear router maps each concatenated token pair to a
    scalar gate ``g``; the fused token is ``g * low + (1 - g) * high``, with no
    normalization after fusion. A max-norm temporal convolution spanning all
    ``N`` tokens produces the class logits. Two auxiliary heads classify each
    branch separately during training.

Training objective (``compute_loss``)
    ``CE_main + w_aux * (CE_low + CE_high) + schedule * w_shape * L_gain``.
    For Gain RTUR, a frozen EMA copy of the model searches, for each trial, a
    zero-sum gate perturbation around the trial-mean gate that lowers the
    complete-trial cross-entropy. The verified gain weights a SmoothL1 loss
    between the centered student router logits and the centered candidate
    logits. The router sees detached branch tokens, so this loss trains only
    the router.

Module and parameter names follow the research implementation, so archived
checkpoints, including the EMA teacher, load with ``strict=True``.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# The research runs used deterministic cuDNN convolutions.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


EMBED_DIM = 20
BRANCH_DEPTH = 2
GLA_HEADS = 2
FFN_EXPANSION = 2
# (low, high) dropout inside the tokenizer and attention, and inside the FFN.
BRANCH_DROPOUT = (0.1, 0.3)
FFN_DROPOUT = (0.3, 0.5)
FUSION_DROPOUT = 0.1
AUX_HEAD_DROPOUT = 0.2
HEAD_MAX_NORM = 1.0
BACKENDS = ("gla", "gla_cpu")


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class MaxNormConv2d(nn.Conv2d):
    """Conv2d whose output-channel filters are renormalized to ``max_norm``."""

    def __init__(self, *args, max_norm: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class LogPowerTokenizer(nn.Module):
    """``[B, 1, C, T] -> [B, N, D]`` log-power tokens.

    A 1x1 convolution mixes electrodes into ``D`` features, a depthwise valid
    temporal convolution filters each feature, and the squared response is
    averaged over windows of ``pool_size`` samples with stride ``pool_stride``.
    """

    def __init__(self, n_chan: int, emb_size: int, kernel_size: int, pool_size: int,
                 pool_stride: int, drop_p: float, eps: float = 1e-4):
        super().__init__()
        self.patch_size = pool_size
        self.patch_stride = pool_stride
        self.eps = eps
        self.sconv = nn.Conv1d(n_chan, emb_size, kernel_size=1, bias=False)
        self.tconv = nn.Conv1d(emb_size, emb_size, kernel_size=kernel_size, padding=0,
                               groups=emb_size, bias=False)
        self.bn = nn.BatchNorm1d(emb_size)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(emb_size)
        self.drop = nn.Dropout(drop_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(1)
        if x.shape[-1] < self.patch_size:
            raise ValueError(f"input T={x.shape[-1]} < patch_size={self.patch_size}")
        feat = self.act(self.bn(self.tconv(self.sconv(x))))
        power = F.avg_pool1d(feat.pow(2), kernel_size=self.patch_size, stride=self.patch_stride)
        tokens = torch.log(torch.clamp(power, min=self.eps, max=1e4)).transpose(1, 2)
        return self.drop(self.ln(tokens))


class RMSNormSiLUGate(nn.Module):
    def __init__(self, size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x, gate):
        return x * (x.square().mean(-1, keepdim=True) + self.eps).rsqrt() * self.weight * F.silu(gate)


class CPUGatedLinearAttention(nn.Module):
    """FP32 PyTorch GLA layer for CPU validation, without Triton.

    The recurrence follows fla-org/flash-linear-attention's MIT-licensed
    ``fla/ops/gla/naive.py``; projections, per-head RMS normalization and the
    SiLU output gate follow ``fla/layers/gla.py``, whose parameter names are
    kept. Only the configuration used here (no short convolution, cache or
    mask) is implemented. The upstream notice is in ``FLA_LICENSE.txt``.
    """

    def __init__(self, hidden_size: int, num_heads: int, expand_k=0.5, expand_v=1.0):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        if self.key_dim % num_heads or self.value_dim % num_heads:
            raise ValueError("GLA key/value dimensions must be divisible by head count")
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.gk_proj = nn.Sequential(nn.Linear(hidden_size, 16, bias=False), nn.Linear(16, self.key_dim))
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        self.g_norm_swish_gate = RMSNormSiLUGate(self.head_v_dim)
        self.gate_logit_normalizer = 16

    def forward(self, x):
        batch, length, _ = x.shape
        q = self.q_proj(x).reshape(batch, length, self.num_heads, self.head_k_dim)
        k = self.k_proj(x).reshape_as(q)
        v = self.v_proj(x).reshape(batch, length, self.num_heads, self.head_v_dim)
        decay = (F.logsigmoid(self.gk_proj(x)) / self.gate_logit_normalizer).exp().reshape_as(q)
        state = x.new_zeros(batch, self.num_heads, self.head_k_dim, self.head_v_dim)
        out = torch.empty_like(v)
        q = q * self.head_k_dim ** -0.5
        for index in range(length):
            state = state * decay[:, index, :, :, None] + k[:, index, :, :, None] * v[:, index, :, None, :]
            out[:, index] = (q[:, index, :, :, None] * state).sum(-2)
        gate = self.g_proj(x).reshape_as(out)
        out = self.g_norm_swish_gate(out, gate).reshape(batch, length, self.value_dim)
        return self.o_proj(out), None, None


class GLAAttention(nn.Module):
    """GLA token mixer: ``gla`` uses Flash Linear Attention, ``gla_cpu`` the PyTorch layer."""

    def __init__(self, emb_size: int, drop_p: float, backend: str):
        super().__init__()
        if backend == "gla":
            from fla.layers import GatedLinearAttention
        elif backend == "gla_cpu":
            GatedLinearAttention = CPUGatedLinearAttention
        else:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        # With the default expand_k=0.5, a width of 20 gives key and value
        # dimensions 10 and 20; two heads is the largest count dividing both.
        self.attn = GatedLinearAttention(hidden_size=emb_size, num_heads=GLA_HEADS)
        self.drop = nn.Dropout(drop_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, *_ = self.attn(x)
        return self.drop(out)


class ResidualAdd(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        res = x
        x = self.fn(x)
        x += res
        return x


def encoder_block(emb_size: int, drop_p: float, ffn_drop_p: float, backend: str) -> nn.Sequential:
    """Pre-norm GLA block followed by a pre-norm feed-forward block."""

    return nn.Sequential(
        ResidualAdd(nn.Sequential(nn.LayerNorm(emb_size), GLAAttention(emb_size, drop_p, backend))),
        ResidualAdd(nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Sequential(
                nn.Linear(emb_size, FFN_EXPANSION * emb_size), nn.GELU(), nn.Dropout(ffn_drop_p),
                nn.Linear(FFN_EXPANSION * emb_size, emb_size),
            ),
            nn.Dropout(drop_p),
        )),
    )


class Branch(nn.Module):
    """Tokenizer (``pe``) and GLA encoder (``te``) for one frequency band."""

    def __init__(self, n_chan: int, kernel_size: int, pool_size: int, pool_stride: int,
                 drop_p: float, ffn_drop_p: float, backend: str):
        super().__init__()
        self.pe = LogPowerTokenizer(n_chan, EMBED_DIM, kernel_size, pool_size, pool_stride, drop_p)
        self.te = nn.Sequential(*[
            encoder_block(EMBED_DIM, drop_p, ffn_drop_p, backend) for _ in range(BRANCH_DEPTH)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.te(self.pe(x))


class TokenScalarGatedFusion(nn.Module):
    """One sigmoid gate per token from a linear router over ``[low; high]``.

    ``last_gate_output`` keeps the gate expanded to ``[B, N, D]``.
    """

    def __init__(self, emb_size: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(2 * emb_size, 1), nn.Sigmoid())
        self.dropout = nn.Dropout(FUSION_DROPOUT)
        nn.init.constant_(self.gate[0].bias, 0.0)
        self.last_gate_output = None

    def forward(self, z_l: torch.Tensor, z_h: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat([z_l, z_h], dim=-1))
        self.last_gate_output = gate.expand(-1, -1, z_l.size(-1))
        return self.dropout(gate * z_l + (1.0 - gate) * z_h)


class TemporalReadoutHead(nn.Module):
    """Max-norm convolution over all ``N`` tokens: ``[B, N, D] -> [B, classes]``."""

    def __init__(self, emb_size: int, n_classes: int, n_tokens: int):
        super().__init__()
        self.conv = MaxNormConv2d(emb_size, n_classes, kernel_size=(1, n_tokens), max_norm=HEAD_MAX_NORM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x.transpose(1, 2).unsqueeze(2)).squeeze(-1).squeeze(-1)


class TokenMean(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)


class AuxiliaryHead(nn.Module):
    """Training-only branch classifier on mean-pooled tokens."""

    def __init__(self, emb_size: int, n_classes: int):
        super().__init__()
        self.head = nn.Sequential(
            TokenMean(), nn.LayerNorm(emb_size), nn.Linear(emb_size, emb_size), nn.GELU(),
            nn.Dropout(AUX_HEAD_DROPOUT), nn.Linear(emb_size, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ---------------------------------------------------------------------------
# Gain RTUR objective
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GainRTURLossConfig:
    """Loss weights, RTUR schedule, EMA decay and candidate search settings.

    Progress for the schedule and EMA decay is ``global_step`` divided by
    ``min(planned steps, schedule_horizon_steps)``. RTUR is zero during the
    warm-up fraction and then ramps linearly to full weight.
    """

    auxiliary_ce_weight: float = 0.3
    shape_weight: float = 0.1
    smooth_l1_beta: float = 0.25
    warmup_fraction: float = 0.1
    ramp_fraction: float = 0.1
    schedule_horizon_steps: int | None = 2500
    ema_decay_start: float = 0.99
    ema_decay_end: float = 0.999
    gradient_diagnostic_interval: int = 50
    candidate_steps: int = 3
    candidate_step_size: float = 0.1
    candidate_radius: float = 0.1
    gate_epsilon: float = 1e-4
    gain_tau: float = 1e-3

    def __post_init__(self):
        for name in ("auxiliary_ce_weight", "shape_weight", "warmup_fraction", "ramp_fraction"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("smooth_l1_beta", "candidate_step_size", "candidate_radius", "gain_tau"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.warmup_fraction + self.ramp_fraction > 1.0:
            raise ValueError("warmup_fraction + ramp_fraction must not exceed 1")
        if self.schedule_horizon_steps is not None and self.schedule_horizon_steps < 1:
            raise ValueError("schedule_horizon_steps must be positive or None")
        if not 0.0 <= self.ema_decay_start < 1.0:
            raise ValueError("ema_decay_start must be in [0, 1)")
        if not self.ema_decay_start <= self.ema_decay_end < 1.0:
            raise ValueError("ema_decay_end must be in [ema_decay_start, 1)")
        if self.gradient_diagnostic_interval < 0:
            raise ValueError("gradient_diagnostic_interval must be non-negative")
        if not isinstance(self.candidate_steps, int) or self.candidate_steps < 1:
            raise ValueError("candidate_steps must be a positive integer")
        if not math.isfinite(self.gate_epsilon) or not 0.0 < self.gate_epsilon < 0.5:
            raise ValueError("gate_epsilon must lie strictly between 0 and 0.5")


@dataclass(frozen=True)
class GainRTURTarget:
    anchor_gate: torch.Tensor
    candidate_gate: torch.Tensor
    anchor_ce: torch.Tensor
    candidate_ce: torch.Tensor
    gain: torch.Tensor
    confidence: torch.Tensor
    target_centered: torch.Tensor
    utility_low_over_high: torch.Tensor
    active_trials: torch.Tensor
    mean_gate_error_per_trial: torch.Tensor
    constraint_violation_per_trial: torch.Tensor


@dataclass(frozen=True)
class GainRTURSupervision:
    """The scheduled routing loss and its detached diagnostics."""

    total: torch.Tensor
    shape: torch.Tensor
    ranking: torch.Tensor
    schedule_scale: torch.Tensor
    utility_low_over_high: torch.Tensor
    target_centered: torch.Tensor
    router_logits_centered: torch.Tensor
    active_trials: torch.Tensor
    active_trial_fraction: torch.Tensor
    utility_spread_mean: torch.Tensor
    effective_pair_count: torch.Tensor
    pair_order_accuracy: torch.Tensor
    gate_utility_spearman: torch.Tensor
    gate_per_token: torch.Tensor
    gate_saturation_fraction: torch.Tensor
    dynamic_over_constant_ce_advantage: torch.Tensor
    dynamic_over_shuffled_ce_advantage: torch.Tensor
    dynamic_over_constant_accuracy_delta: torch.Tensor
    dynamic_over_shuffled_accuracy_delta: torch.Tensor
    used_full_ema_teacher: bool
    anchor_gate: torch.Tensor
    candidate_gate: torch.Tensor
    anchor_ce: torch.Tensor
    candidate_ce: torch.Tensor
    gain: torch.Tensor
    confidence: torch.Tensor
    shape_per_trial: torch.Tensor
    mean_gate_error_per_trial: torch.Tensor
    constraint_violation_per_trial: torch.Tensor


@dataclass(frozen=True)
class GainRTURLoss:
    """Complete CE + auxiliary CE + scheduled Gain RTUR objective."""

    total: torch.Tensor
    ce_main: torch.Tensor
    ce_low: torch.Tensor
    ce_high: torch.Tensor
    rtur: GainRTURSupervision

    def detached_components(self) -> dict[str, torch.Tensor]:
        rtur = self.rtur
        return {
            "ce_main": self.ce_main.detach(),
            "ce_low": self.ce_low.detach(),
            "ce_high": self.ce_high.detach(),
            "loss_rtur": rtur.total.detach(),
            "loss_rtur_shape": rtur.shape.detach(),
            "loss_rtur_ranking": rtur.ranking.detach(),
            "rtur_schedule_scale": rtur.schedule_scale.detach(),
            "rtur_active_trial_fraction": rtur.active_trial_fraction.detach(),
            "rtur_utility_abs_mean": rtur.utility_low_over_high.abs().mean().detach(),
            "rtur_utility_spread_mean": rtur.utility_spread_mean.detach(),
            "rtur_effective_pair_count": rtur.effective_pair_count.detach(),
            "rtur_pair_order_accuracy": rtur.pair_order_accuracy.detach(),
            "rtur_gate_utility_spearman": rtur.gate_utility_spearman.detach(),
            "gate_mean": rtur.gate_per_token.mean().detach(),
            "gate_token_var": rtur.gate_per_token.var(dim=1, unbiased=False).mean().detach(),
            "gate_saturation_fraction": rtur.gate_saturation_fraction.detach(),
            "rtur_dynamic_over_constant_ce_advantage": rtur.dynamic_over_constant_ce_advantage.detach(),
            "rtur_dynamic_over_shuffled_ce_advantage": rtur.dynamic_over_shuffled_ce_advantage.detach(),
            "rtur_dynamic_over_constant_accuracy_delta": rtur.dynamic_over_constant_accuracy_delta.detach(),
            "rtur_dynamic_over_shuffled_accuracy_delta": rtur.dynamic_over_shuffled_accuracy_delta.detach(),
            "rtur_used_full_ema_teacher": self.ce_main.new_tensor(float(rtur.used_full_ema_teacher)).detach(),
            "rtur_gain_mean": rtur.gain.mean().detach(),
            "rtur_gain_confidence_mean": rtur.confidence.mean().detach(),
            "rtur_gain_positive_fraction": rtur.active_trials.float().mean().detach(),
            "rtur_gain_anchor_ce": rtur.anchor_ce.mean().detach(),
            "rtur_gain_candidate_ce": rtur.candidate_ce.mean().detach(),
        }


def _feature_outputs(outputs):
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 8:
        raise ValueError("Gain RTUR requires the 8-item return_feat=True output contract")
    return outputs


def _validate_labels(labels: torch.Tensor, batch_size: int) -> torch.Tensor:
    labels = labels.reshape(-1).long()
    if labels.numel() != batch_size:
        raise ValueError(f"expected {batch_size} labels, got {labels.numel()}")
    return labels


def rtur_schedule_scale(progress: float, *, warmup_fraction: float = 0.1, ramp_fraction: float = 0.1) -> float:
    """Zero during warm-up, then a linear ramp to one."""

    if not math.isfinite(progress):
        raise ValueError("progress must be finite")
    if warmup_fraction < 0.0 or ramp_fraction < 0.0:
        raise ValueError("warmup_fraction and ramp_fraction must be non-negative")
    if warmup_fraction + ramp_fraction > 1.0:
        raise ValueError("warmup_fraction + ramp_fraction must not exceed 1")
    progress = min(max(float(progress), 0.0), 1.0)
    if progress <= warmup_fraction:
        return 0.0
    if ramp_fraction == 0.0:
        return 1.0
    return min((progress - warmup_fraction) / ramp_fraction, 1.0)


@torch.no_grad()
def project_zero_sum_box(values: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Euclidean projection onto a per-trial box intersected with ``sum = 0``.

    Bounds must admit zero. The multiplier is solved on its piecewise-linear
    breakpoints in one batch; clipping a centered vector would lose zero-sum.
    """

    with torch.autocast(device_type=values.device.type, enabled=False):
        values, lower, upper = values.float(), lower.float(), upper.float()
        knots = torch.cat([values - upper, values - lower], dim=1).sort(dim=1).values
        sums = (values[:, None, :] - knots[:, :, None]).clamp(
            min=lower[:, None, :], max=upper[:, None, :]
        ).sum(dim=-1)
        right = (sums > 0.0).sum(dim=1, keepdim=True).clamp(1, knots.size(1) - 1)
        left = right - 1
        left_knot, right_knot = knots.gather(1, left), knots.gather(1, right)
        left_sum, right_sum = sums.gather(1, left), sums.gather(1, right)
        fraction = left_sum / (left_sum - right_sum).clamp_min(torch.finfo(values.dtype).tiny)
        multiplier = left_knot + fraction * (right_knot - left_knot)
        return (values - multiplier).clamp(min=lower, max=upper)


def _readout_logits(tokens_low, tokens_high, gate, head_weight, head_bias) -> torch.Tensor:
    """FP32 main-head logits for a given gate, without fusion dropout."""

    expanded_gate = gate.float().unsqueeze(-1)
    fused = expanded_gate * tokens_low.float() + (1.0 - expanded_gate) * tokens_high.float()
    logits = torch.einsum("bne,cen->bc", fused, head_weight.detach().float())
    if head_bias is not None:
        logits = logits + head_bias.detach().float()
    return logits


@torch.no_grad()
def gate_utility(tokens_low, tokens_high, gate, labels, *, head_weight, head_bias=None):
    """Return fixed-head logits and ``-d CE_trial / d gate_token`` in FP32."""

    with torch.autocast(device_type=tokens_low.device.type, enabled=False):
        low, high = tokens_low.detach().float(), tokens_high.detach().float()
        difference = low - high
        fused = high + gate.detach().float().unsqueeze(-1) * difference
        weight = head_weight.detach().float()
        logits = torch.einsum("bne,cen->bc", fused, weight)
        if head_bias is not None:
            logits = logits + head_bias.detach().float()
        class_direction = F.one_hot(labels.long(), num_classes=logits.size(-1)).float()
        class_direction = class_direction - logits.softmax(dim=-1)
        token_class_derivative = torch.einsum("bne,cen->bnc", difference, weight)
        utility = torch.einsum("bc,bnc->bn", class_direction, token_class_derivative)
        return logits, utility


@torch.no_grad()
def gain_rtur_candidate_target(tokens_low, tokens_high, teacher_gate, labels, *, head_weight,
                               head_bias=None, config: GainRTURLossConfig | None = None) -> GainRTURTarget:
    """Search and verify a bounded temporal reallocation at a fixed mean gate.

    The anchor is the clipped trial-mean gate. Each step moves along the gate
    utility, projects onto the zero-sum box within ``candidate_radius``, and is
    accepted only if the complete-trial CE decreases, so the gain is
    non-negative.
    """

    config = config or GainRTURLossConfig()
    if tokens_low.ndim != 3 or tokens_low.shape != tokens_high.shape:
        raise ValueError("low/high tokens must share shape [B,N,E]")
    labels = _validate_labels(labels, tokens_low.size(0))
    if teacher_gate.ndim == 3:
        teacher_gate = teacher_gate.mean(dim=-1)
    if teacher_gate.shape != tokens_low.shape[:2]:
        raise ValueError("teacher_gate must have shape [B,N] or [B,N,E]")
    with torch.autocast(device_type=tokens_low.device.type, enabled=False):
        low, high = tokens_low.detach().float(), tokens_high.detach().float()
        mean_gate = teacher_gate.detach().float().mean(dim=1, keepdim=True).clamp(
            config.gate_epsilon, 1.0 - config.gate_epsilon
        )
        anchor = mean_gate.expand(-1, tokens_low.size(1))
        lower = torch.maximum(mean_gate.new_full(mean_gate.shape, -config.candidate_radius),
                              config.gate_epsilon - mean_gate)
        upper = torch.minimum(mean_gate.new_full(mean_gate.shape, config.candidate_radius),
                              1.0 - config.gate_epsilon - mean_gate)
        anchor_logits, anchor_utility = gate_utility(
            low, high, anchor, labels, head_weight=head_weight, head_bias=head_bias)
        anchor_ce = F.cross_entropy(anchor_logits, labels, reduction="none")
        best_gate, best_ce = anchor, anchor_ce
        utility = anchor_utility
        for _ in range(config.candidate_steps):
            proposal_delta = project_zero_sum_box(
                best_gate - anchor + config.candidate_step_size * utility, lower, upper)
            proposal_gate = anchor + proposal_delta
            proposal_logits, proposal_utility = gate_utility(
                low, high, proposal_gate, labels, head_weight=head_weight, head_bias=head_bias)
            proposal_ce = F.cross_entropy(proposal_logits, labels, reduction="none")
            accepted = proposal_ce < best_ce
            best_gate = torch.where(accepted[:, None], proposal_gate, best_gate)
            best_ce = torch.where(accepted, proposal_ce, best_ce)
            utility = torch.where(accepted[:, None], proposal_utility, utility)
        gain = (anchor_ce - best_ce).clamp_min(0.0)
        confidence = gain / (gain + config.gain_tau)
        candidate_logits = torch.logit(best_gate)
        target_centered = candidate_logits - candidate_logits.mean(dim=1, keepdim=True)
        delta = best_gate - anchor
        mean_error = delta.mean(dim=1).abs()
        gate_box_violation = torch.maximum(
            (config.gate_epsilon - best_gate).clamp_min(0.0),
            (best_gate - (1.0 - config.gate_epsilon)).clamp_min(0.0),
        ).amax(dim=1)
        radius_violation = (delta.abs() - config.candidate_radius).clamp_min(0.0).amax(dim=1)
        constraint_violation = torch.maximum(torch.maximum(gate_box_violation, radius_violation), mean_error)
        return GainRTURTarget(
            anchor_gate=anchor, candidate_gate=best_gate, anchor_ce=anchor_ce, candidate_ce=best_ce,
            gain=gain, confidence=confidence, target_centered=target_centered,
            utility_low_over_high=anchor_utility, active_trials=gain > 0.0,
            mean_gate_error_per_trial=mean_error, constraint_violation_per_trial=constraint_violation,
        )


def gain_weighted_centered_shape_loss(router_logits, target_centered, confidence, *, beta: float = 0.25):
    """Confidence-weighted SmoothL1 between centered router and target logits.

    The full batch is the denominator, so trials without a verified gain
    contribute zero.
    """

    centered = router_logits.float() - router_logits.float().mean(dim=1, keepdim=True)
    per_trial = F.smooth_l1_loss(
        centered, target_centered.detach().float(), beta=beta, reduction="none"
    ).mean(dim=1)
    return (confidence.detach().float() * per_trial).mean(), centered, per_trial


@torch.no_grad()
def _pairwise_ranking_diagnostic(router_logits, target: GainRTURTarget):
    """Confidence-weighted RankNet loss and pair-order accuracy (not optimized)."""

    n_tokens = router_logits.size(1)
    upper = torch.triu(torch.ones(n_tokens, n_tokens, device=router_logits.device, dtype=torch.bool), diagonal=1)
    target_delta = target.target_centered.unsqueeze(2) - target.target_centered.unsqueeze(1)
    logit_delta = router_logits.unsqueeze(2) - router_logits.unsqueeze(1)
    selected = upper & target.active_trials[:, None, None] & (target_delta.abs() > 0.0)
    weight = target_delta.abs() * selected
    pair_loss = F.softplus(-target_delta.sign() * logit_delta)
    per_trial = (pair_loss * weight).sum(dim=(1, 2)) / weight.sum(dim=(1, 2)).clamp_min(1e-12)
    loss = (target.confidence * per_trial).mean()
    ordered = (target_delta.sign() * logit_delta > 0.0).float()
    accuracy = (ordered * weight).sum() / weight.sum().clamp_min(1e-12)
    return loss, selected.sum().float(), accuracy.detach()


@torch.no_grad()
def _timing_diagnostics(model, features, labels, target: GainRTURTarget):
    """Gate/utility rank correlation and CE/accuracy of dynamic vs. constant/shifted gates."""

    with torch.autocast(device_type=features[3].device.type, enabled=False):
        gate = features[7].detach().float().mean(dim=-1)
        weight, bias = model.head.conv.weight.squeeze(2), model.head.conv.bias
        logits = [
            _readout_logits(features[3].detach(), features[4].detach(), intervention, weight, bias)
            for intervention in (gate, gate.mean(dim=1, keepdim=True).expand_as(gate), gate.roll(1, dims=1))
        ]
        ce = [F.cross_entropy(value, labels, reduction="none") for value in logits]
        correct = [value.argmax(dim=1).eq(labels).float() for value in logits]
        gate_rank = gate.argsort(dim=1).argsort(dim=1).float()
        target_rank = target.target_centered.argsort(dim=1).argsort(dim=1).float()
        gate_rank = gate_rank - gate_rank.mean(dim=1, keepdim=True)
        target_rank = target_rank - target_rank.mean(dim=1, keepdim=True)
        correlation = (gate_rank * target_rank).sum(dim=1) / (
            gate_rank.norm(dim=1) * target_rank.norm(dim=1)).clamp_min(1e-12)
        active = target.active_trials.float()
        spearman = (correlation * active).sum() / active.sum().clamp_min(1.0)
        return (spearman, (ce[1] - ce[0]).mean(), (ce[2] - ce[0]).mean(),
                (correct[0] - correct[1]).mean(), (correct[0] - correct[2]).mean())


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _probe_token_count(tokenizer: LogPowerTokenizer, n_chan: int, input_time: int) -> int:
    with torch.no_grad():
        return tokenizer(torch.rand(2, 1, n_chan, input_time)).shape[-2]


class EEGCrossformer(nn.Module):
    """Dual-band EEG model with token-wise fusion and a model-owned EMA teacher.

    ``kernel_size``, ``pool_size`` and ``pool_stride`` (in samples) set the
    shared valid-convolution token grid of both branches. ``forward`` returns
    ``(main_logits, low_logits, high_logits, low_tokens, high_tokens,
    fused_tokens, fused_tokens, gate)`` when ``return_feat`` is true and only
    the main logits otherwise; Gain RTUR needs the full tuple. The sixth and
    seventh items are the same tensor, kept for the archived output layout.
    """

    def __init__(self, n_chan: int, input_time: int, n_class: int, *, kernel_size: int,
                 pool_size: int, pool_stride: int, backend: str = "gla", return_feat: bool = True):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd to keep token centers aligned, got {kernel_size}")
        self.return_feat = return_feat
        branch = dict(n_chan=n_chan, kernel_size=kernel_size, pool_size=pool_size,
                      pool_stride=pool_stride, backend=backend)
        self.l_branch = Branch(drop_p=BRANCH_DROPOUT[0], ffn_drop_p=FFN_DROPOUT[0], **branch)
        self.h_branch = Branch(drop_p=BRANCH_DROPOUT[1], ffn_drop_p=FFN_DROPOUT[1], **branch)

        # The steps marked "seed compatibility" reproduce the global-RNG use of
        # the research implementation, which probed the low tokenizer in
        # training mode and built a mean-pooling main head before replacing it.
        # They keep a given seed's initial weights identical to the archived runs.
        n_tokens = _probe_token_count(self.l_branch.pe, n_chan, input_time)  # seed compatibility
        self.l_branch.pe.bn.reset_running_stats()
        self.head = nn.Identity()  # replaced below; registered here to keep the parameter order
        nn.Linear(EMBED_DIM, EMBED_DIM), nn.Linear(EMBED_DIM, n_class)  # seed compatibility
        self.fusion = TokenScalarGatedFusion(EMBED_DIM)
        self.head_l = AuxiliaryHead(EMBED_DIM, n_class)
        self.head_h = AuxiliaryHead(EMBED_DIM, n_class)
        self.h_branch.pe.eval()
        if _probe_token_count(self.h_branch.pe, n_chan, input_time) != n_tokens:  # seed compatibility
            raise ValueError("low and high branches must produce the same token count")
        self.h_branch.pe.train()
        self.head = TemporalReadoutHead(EMBED_DIM, n_class, n_tokens)

        # Copy before assigning so the teacher does not contain itself.
        self._ema_teacher = None
        teacher = copy.deepcopy(self)
        self._ema_teacher = teacher
        teacher.requires_grad_(False)
        teacher.eval()
        self.register_buffer("ema_update_count", torch.zeros((), dtype=torch.long))
        self._last_ema_global_step: int | None = None

    def forward(self, x: torch.Tensor):
        views = torch.split(x, 1, dim=1)
        if len(views) != 2:
            raise ValueError(f"expected two filter-bank views [B,2,C,T], got shape {tuple(x.shape)}")
        x_l = self.l_branch(views[0])
        x_h = self.h_branch(views[1])
        fused = self.fusion(x_l, x_h)
        out_main = self.head(fused)
        out_l = self.head_l(x_l)
        out_h = self.head_h(x_h)
        if self.return_feat:
            return out_main, out_l, out_h, x_l, x_h, fused, fused, self.fusion.last_gate_output
        return out_main

    def get_branch_grad_norms(self) -> dict:
        """Gradient L2 norms of the two branches; call after ``backward``."""

        def norm(module):
            return sum(p.grad.data.norm(2).item() ** 2 for p in module.parameters() if p.grad is not None) ** 0.5

        return {"grad_norm_l_branch": norm(self.l_branch), "grad_norm_h_branch": norm(self.h_branch)}

    # EMA teacher -------------------------------------------------------------

    @property
    def ema_teacher(self) -> "EEGCrossformer":
        if self._ema_teacher is None:
            raise RuntimeError("EMA teacher is not initialized")
        return self._ema_teacher

    def train(self, mode: bool = True):
        result = super().train(mode)
        teacher = getattr(self, "_ema_teacher", None)
        if teacher is not None:
            teacher.eval()
        return result

    @torch.no_grad()
    def update_ema_teacher(self, decay: float) -> None:
        """Move every teacher parameter and floating buffer towards the student."""

        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        teacher_parameters = dict(self.ema_teacher.named_parameters())
        for name, student in self.named_parameters():
            if not name.startswith("_ema_teacher."):
                teacher_parameters[name].lerp_(student.detach(), 1.0 - decay)
        teacher_buffers = dict(self.ema_teacher.named_buffers())
        for name, student in self.named_buffers():
            if name.startswith("_ema_teacher.") or name == "ema_update_count":
                continue
            teacher_buffer = teacher_buffers.get(name)
            if teacher_buffer is None:
                continue
            if torch.is_floating_point(teacher_buffer):
                teacher_buffer.lerp_(student.detach(), 1.0 - decay)
            else:
                teacher_buffer.copy_(student.detach())
        self.ema_update_count.add_(1)
        self.ema_teacher.eval()

    @staticmethod
    def ema_decay_at_progress(progress: float, config: GainRTURLossConfig) -> float:
        progress = min(max(float(progress), 0.0), 1.0)
        return config.ema_decay_start + progress * (config.ema_decay_end - config.ema_decay_start)

    def maybe_update_ema_teacher(self, *, global_step: int | None, progress: float,
                                 config: GainRTURLossConfig) -> None:
        """Update once per optimizer step, before that step's loss."""

        if global_step is None:
            return
        global_step = int(global_step)
        if global_step <= 0 or global_step == self._last_ema_global_step:
            return
        self.update_ema_teacher(self.ema_decay_at_progress(progress, config))
        self._last_ema_global_step = global_step

    # Objective ---------------------------------------------------------------

    def _gain_target(self, outputs, labels, inputs, config):
        teacher = self.ema_teacher
        teacher.eval()
        with torch.no_grad():
            features = _feature_outputs(teacher(inputs) if inputs is not None else outputs)
            target = gain_rtur_candidate_target(
                features[3], features[4], features[7], labels,
                head_weight=teacher.head.conv.weight.squeeze(2), head_bias=teacher.head.conv.bias,
                config=config,
            )
        return target, inputs is not None

    def compute_loss(self, outputs, labels: torch.Tensor, config: GainRTURLossConfig | None = None, *,
                     inputs=None, progress: float = 1.0, update_teacher: bool = False,
                     global_step: int | None = None) -> GainRTURLoss:
        """Main CE, auxiliary branch CE and scheduled Gain RTUR for one batch.

        ``inputs`` is the transformed batch; the teacher re-encodes it. Without
        it, the teacher reuses the student's branch tokens.
        """

        config = config or GainRTURLossConfig()
        if update_teacher:
            self.maybe_update_ema_teacher(global_step=global_step, progress=progress, config=config)
        features = _feature_outputs(outputs)
        logits, logits_low, logits_high = features[:3]
        labels = _validate_labels(labels, logits.size(0))
        ce_main = F.cross_entropy(logits.float(), labels)
        ce_low = F.cross_entropy(logits_low.float(), labels)
        ce_high = F.cross_entropy(logits_high.float(), labels)
        supervision = self._gain_supervision(features, labels, inputs=inputs, progress=progress, config=config)
        return GainRTURLoss(
            total=ce_main + config.auxiliary_ce_weight * (ce_low + ce_high) + supervision.total,
            ce_main=ce_main, ce_low=ce_low, ce_high=ce_high, rtur=supervision,
        )

    def _gain_supervision(self, features, labels, *, inputs, progress, config) -> GainRTURSupervision:
        with torch.autocast(device_type=features[3].device.type, enabled=False):
            # Detached tokens: the routing loss reaches only the router.
            router_input = torch.cat([features[3].detach().float(), features[4].detach().float()], dim=-1)
            router_logits = self.fusion.gate[0](router_input).squeeze(-1).float()
        target, used_full_teacher = self._gain_target(features, labels, inputs, config)
        shape, centered, shape_per_trial = gain_weighted_centered_shape_loss(
            router_logits, target.target_centered, target.confidence, beta=config.smooth_l1_beta)
        ranking, pairs, pair_accuracy = _pairwise_ranking_diagnostic(router_logits, target)
        diagnostics = _timing_diagnostics(self, features, labels, target)
        schedule = router_logits.new_tensor(rtur_schedule_scale(
            progress, warmup_fraction=config.warmup_fraction, ramp_fraction=config.ramp_fraction))
        utility = target.utility_low_over_high
        spread = torch.quantile(utility, 0.9, dim=1) - torch.quantile(utility, 0.1, dim=1)
        gate = features[7].detach().float().mean(dim=-1)
        return GainRTURSupervision(
            total=schedule * (config.shape_weight * shape),
            shape=shape, ranking=ranking, schedule_scale=schedule,
            utility_low_over_high=utility, target_centered=target.target_centered,
            router_logits_centered=centered, active_trials=target.active_trials,
            active_trial_fraction=target.active_trials.float().mean(),
            utility_spread_mean=spread.mean(), effective_pair_count=pairs,
            pair_order_accuracy=pair_accuracy, gate_utility_spearman=diagnostics[0],
            gate_per_token=gate,
            gate_saturation_fraction=((gate < 0.05) | (gate > 0.95)).float().mean(),
            dynamic_over_constant_ce_advantage=diagnostics[1],
            dynamic_over_shuffled_ce_advantage=diagnostics[2],
            dynamic_over_constant_accuracy_delta=diagnostics[3],
            dynamic_over_shuffled_accuracy_delta=diagnostics[4],
            used_full_ema_teacher=used_full_teacher,
            anchor_gate=target.anchor_gate, candidate_gate=target.candidate_gate,
            anchor_ce=target.anchor_ce, candidate_ce=target.candidate_ce,
            gain=target.gain, confidence=target.confidence, shape_per_trial=shape_per_trial,
            mean_gate_error_per_trial=target.mean_gate_error_per_trial,
            constraint_violation_per_trial=target.constraint_violation_per_trial,
        )

    def compute_relative_token_utility(self, outputs, labels, *, inputs=None,
                                       config: GainRTURLossConfig | None = None) -> torch.Tensor:
        """Teacher ``-d CE / d gate`` per token at the trial-mean anchor gate."""

        features = _feature_outputs(outputs)
        labels = _validate_labels(labels, features[0].size(0))
        target, _ = self._gain_target(features, labels, inputs, config or GainRTURLossConfig())
        return target.utility_low_over_high


__all__ = ["EEGCrossformer", "GainRTURLossConfig", "GainRTURLoss", "GainRTURSupervision",
           "gain_rtur_candidate_target", "rtur_schedule_scale"]
