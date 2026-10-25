"""Detached RTUR diagnostics and their trial/step-weighted epoch reductions."""

from dataclasses import dataclass, field

import numpy as np
import torch


TRIAL_METRICS = (
  "main/ce", "aux/ce_low", "aux/ce_high", "gate/trial_mean",
  "gate/temporal_std", "gate/near_endpoint_fraction",
)
ACTIVE_METRICS = (
  "rtur/shape_raw", "rtur/logit_temporal_std", "rtur/target_temporal_std",
)
STEP_METRICS = ("rtur/weighted", "rtur/schedule")
RANK_METRICS = ("rtur/rank_raw", "rtur/pair_order_strict")
GRADIENT_METRICS = (
  "router_grad/main_ce_norm", "router_grad/rtur_norm", "router_grad/ratio",
  "router_grad/cosine",
)
GAIN_METRICS = (
  "gain/anchor_ce", "gain/candidate_ce", "gain/ce_advantage",
  "gain/confidence", "gain/accepted_fraction", "gain/shape_weighted",
  "gain/candidate_temporal_std", "gain/mean_gate_error",
  "gain/max_adjustment", "gain/constraint_violation",
)
METRIC_COLUMNS = (
  *TRIAL_METRICS, *ACTIVE_METRICS, *RANK_METRICS, *STEP_METRICS,
  "rtur/active_count", "rtur/trial_count", "rtur/rank_valid_count",
  "rtur/active_fraction", "rtur/utility_spread_median",
  "main/accuracy", "main/macro_f1", *GRADIENT_METRICS,
)


@dataclass
class RTURBatchMetrics:
  values: dict[str, float]
  weights: dict[str, int]
  spread: np.ndarray
  confusion: np.ndarray
  token_values: dict[str, np.ndarray] | None = None


@torch.no_grad()
def collect_rtur_metrics(
  losses, outputs, labels, gradient=None, *, token_limit=0,
) -> RTURBatchMetrics:
  """Read existing loss tensors; never change the optimized objective."""
  supervision = losses.rtur
  active = supervision.active_trials.detach()
  target = supervision.target_centered.detach().float()
  logits = supervision.router_logits_centered.detach().float()
  utility = supervision.utility_low_over_high.detach().float()
  gate = outputs[-1].detach()[..., 0].float()
  trial_count = target.shape[0]
  active_count = int(active.sum())
  n_tokens = target.shape[1]
  upper = torch.triu(torch.ones(n_tokens, n_tokens, device=target.device), 1).bool()
  target_delta = target[:, :, None] - target[:, None, :]
  logit_delta = logits[:, :, None] - logits[:, None, :]
  weight = target_delta.abs() * upper
  denominators = weight.sum((1, 2))
  rank_valid = active & (denominators > 0)
  rank_count = int(rank_valid.sum())
  ordered = (target_delta * logit_delta > 0).float()
  order_per_trial = (ordered * weight).sum((1, 2)) / denominators.clamp_min(
    torch.finfo(weight.dtype).eps
  )

  def mean_selected(values, mask):
    return (values * mask).sum() / mask.sum().clamp_min(1)

  spread = (torch.quantile(utility, .9, dim=1) - torch.quantile(utility, .1, dim=1)).cpu().numpy()
  scalars = {
    "main/ce": losses.ce_main.detach(),
    "aux/ce_low": losses.ce_low.detach(),
    "aux/ce_high": losses.ce_high.detach(),
    "rtur/shape_raw": supervision.shape.detach(),
    "rtur/rank_raw": supervision.ranking.detach(),
    "rtur/weighted": supervision.total.detach(),
    "rtur/schedule": supervision.schedule_scale.detach(),
    "rtur/logit_temporal_std": mean_selected(logits.std(1, unbiased=False), active),
    "rtur/target_temporal_std": mean_selected(target.std(1, unbiased=False), active),
    "rtur/pair_order_strict": mean_selected(order_per_trial, rank_valid),
    "gate/trial_mean": gate.mean(1).mean(),
    "gate/temporal_std": gate.std(1, unbiased=False).mean(),
    "gate/near_endpoint_fraction": ((gate < .05) | (gate > .95)).float().mean(1).mean(),
  }
  is_gain = hasattr(supervision, "confidence")
  if is_gain:
    candidate = supervision.candidate_gate.detach().float()
    anchor = supervision.anchor_gate.detach().float()
    scalars["rtur/shape_raw"] = mean_selected(
      supervision.shape_per_trial.detach(), active,
    )
    raw_pair_loss = torch.nn.functional.softplus(-target_delta.sign() * logit_delta)
    raw_rank_per_trial = (raw_pair_loss * weight).sum((1, 2)) / denominators.clamp_min(
      torch.finfo(weight.dtype).eps
    )
    scalars["rtur/rank_raw"] = mean_selected(raw_rank_per_trial, rank_valid)
    scalars.update({
      "gain/anchor_ce": supervision.anchor_ce.detach().mean(),
      "gain/candidate_ce": supervision.candidate_ce.detach().mean(),
      "gain/ce_advantage": supervision.gain.detach().mean(),
      "gain/confidence": supervision.confidence.detach().mean(),
      "gain/accepted_fraction": active.float().mean(),
      "gain/shape_weighted": supervision.shape.detach(),
      "gain/candidate_temporal_std": candidate.std(1, unbiased=False).mean(),
      "gain/mean_gate_error": supervision.mean_gate_error_per_trial.detach().mean(),
      "gain/max_adjustment": (candidate - anchor).abs().amax(1).mean(),
      "gain/constraint_violation": supervision.constraint_violation_per_trial.detach().mean(),
    })
  # Transfer scalars together rather than synchronize CUDA once per field.
  numbers = torch.stack([v.float() for v in scalars.values()]).cpu().tolist()
  values = dict(zip(scalars, numbers))
  values.update({
    "rtur/active_count": active_count,
    "rtur/trial_count": trial_count,
    "rtur/rank_valid_count": rank_count,
    "rtur/active_fraction": active_count / trial_count,
    "rtur/utility_spread_median": float(np.median(spread)),
  })
  if gradient:
    ce = float(gradient["router_grad_norm_main_ce"])
    rtur = float(gradient["router_grad_norm_rtur"])
    values.update({
      "router_grad/main_ce_norm": ce,
      "router_grad/rtur_norm": rtur,
      "router_grad/ratio": rtur / (ce + 1e-12),
      "router_grad/cosine": float(gradient["router_grad_cosine_main_rtur"]),
    })
  weights = {key: trial_count for key in TRIAL_METRICS}
  weights.update({key: active_count for key in ACTIVE_METRICS})
  weights.update({key: rank_count for key in RANK_METRICS})
  weights.update({key: 1 for key in STEP_METRICS})
  weights.update({key: 1 for key in GRADIENT_METRICS if key in values})
  if is_gain:
    weights.update({key: trial_count for key in GAIN_METRICS})
  labels = labels.detach().reshape(-1).long()
  classes = outputs[0].shape[-1]
  predicted = outputs[0].detach().argmax(-1)
  confusion = torch.bincount(labels * classes + predicted, minlength=classes**2)
  confusion = confusion.reshape(classes, classes).cpu().numpy()
  values.update(classification_metrics(confusion))
  token_values = None
  if is_gain and token_limit:
    count = min(token_limit, trial_count)
    tensors = {
      "utility": utility[:count], "target_centered": target[:count],
      "router_logit_centered": logits[:count], "gate": gate[:count],
      "candidate_gate": candidate[:count],
      "anchor_gate": anchor.expand_as(candidate)[:count],
      "ce_advantage": supervision.gain.detach()[:count, None].expand(-1, n_tokens),
      "confidence": supervision.confidence.detach()[:count, None].expand(-1, n_tokens),
    }
    token_values = {key: value.cpu().numpy() for key, value in tensors.items()}
  return RTURBatchMetrics(values, weights, spread, confusion, token_values)


def classification_metrics(confusion):
  true_positive = np.diag(confusion)
  denominator = confusion.sum(0) + confusion.sum(1)
  f1 = np.divide(2 * true_positive, denominator, out=np.zeros_like(denominator, dtype=float), where=denominator > 0)
  return {
    "main/accuracy": float(true_positive.sum() / max(confusion.sum(), 1)),
    # Match the task's TorchMetrics macro F1: omit classes absent from both
    # labels and predictions, while retaining predicted-only classes.
    "main/macro_f1": float(f1[denominator > 0].mean()) if (denominator > 0).any() else 0.,
  }


@dataclass
class RTUREpochAccumulator:
  sums: dict[str, float] = field(default_factory=dict)
  counts: dict[str, int] = field(default_factory=dict)
  spreads: list[np.ndarray] = field(default_factory=list)
  confusion: np.ndarray | None = None
  active_count: int = 0
  trial_count: int = 0
  rank_count: int = 0
  batch_count: int = 0

  def update(self, batch: RTURBatchMetrics):
    for key, count in batch.weights.items():
      self.sums[key] = self.sums.get(key, 0.) + batch.values[key] * count
      self.counts[key] = self.counts.get(key, 0) + count
    self.spreads.append(batch.spread)
    self.active_count += int(batch.values["rtur/active_count"])
    self.trial_count += int(batch.values["rtur/trial_count"])
    self.rank_count += int(batch.values["rtur/rank_valid_count"])
    self.batch_count += 1
    self.confusion = batch.confusion.copy() if self.confusion is None else self.confusion + batch.confusion

  def compute(self):
    values = {key: self.sums[key] / count if count else 0. for key, count in self.counts.items()}
    values.update({
      "rtur/active_count": self.active_count,
      "rtur/trial_count": self.trial_count,
      "rtur/rank_valid_count": self.rank_count,
      "rtur/active_fraction": self.active_count / max(self.trial_count, 1),
      "rtur/utility_spread_median": float(np.median(np.concatenate(self.spreads))) if self.spreads else 0.,
    })
    if self.confusion is not None:
      values.update(classification_metrics(self.confusion))
    return values
