"""SDBA: synchronized composite augmentation for paired frequency bands.

The input contract is ``[batch, band, channel, time]``.  Every stochastic
decision is sampled at trial level and shared across the band dimension, so a
paired low/high representation never loses its temporal or channel alignment.
"""

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F


SDBA_CODENAME = "SDBA"
CROSS_SESSION_SDBA_CODENAME = "CS-SDBA"


@dataclass(frozen=True)
class SDBAConfig:
  """Configuration for :class:`SynchronizedDualBandAugmentation`."""

  max_shift_points: int = 25
  channel_drop_probability: float = 0.05
  gain_min: float = 0.9
  gain_max: float = 1.1
  mixup_probability: float = 0.5
  mixup_min_lambda: float = 0.6

  def __post_init__(self):
    if self.max_shift_points < 0:
      raise ValueError("max_shift_points must be non-negative")
    if not 0.0 <= self.channel_drop_probability < 1.0:
      raise ValueError("channel_drop_probability must be in [0, 1)")
    if self.gain_min <= 0.0 or self.gain_max < self.gain_min:
      raise ValueError("gain range must be positive and ordered")
    if not 0.0 <= self.mixup_probability <= 1.0:
      raise ValueError("mixup_probability must be in [0, 1]")
    if not 0.0 <= self.mixup_min_lambda <= 1.0:
      raise ValueError("mixup_min_lambda must be in [0, 1]")

  def to_dict(self) -> dict:
    return asdict(self)


class SynchronizedDualBandAugmentation:
  """Apply SDBA without changing batch size or hard class labels.

  The implementation supports one or more aligned bands, although the named
  use case is the paired low/high filter-bank input used by Crossformer.
  """

  codename = SDBA_CODENAME

  def __init__(self, config: SDBAConfig | None = None):
    self.config = config or SDBAConfig()

  def __call__(self, data: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    self._validate_input(data, labels)
    augmented = self._same_class_mixup(data, labels)
    augmented = self._temporal_shift(augmented)
    augmented = self._amplitude_gain(augmented)
    return self._channel_dropout(augmented)

  def _validate_input(self, data: torch.Tensor, labels: torch.Tensor) -> None:
    if data.ndim != 4:
      raise ValueError(
        "SDBA expects [batch, band, channel, time], "
        f"got {tuple(data.shape)}"
      )
    if data.shape[0] != labels.numel():
      raise ValueError("batch size and label count must match")
    if data.shape[1] < 1 or data.shape[2] < 1 or data.shape[3] < 1:
      raise ValueError("band, channel, and time dimensions must be non-empty")
    if not data.is_floating_point():
      raise ValueError("SDBA expects floating-point input")
    if self.config.max_shift_points >= data.shape[-1]:
      raise ValueError("max_shift_points must be smaller than the time dimension")

  def _same_class_mixup(
    self,
    data: torch.Tensor,
    labels: torch.Tensor,
  ) -> torch.Tensor:
    probability = self.config.mixup_probability
    if probability == 0.0:
      return data.clone()

    source = data
    mixed = data.clone()
    flat_labels = labels.reshape(-1).to(device=data.device)
    for class_id in torch.unique(flat_labels):
      class_indices = torch.where(flat_labels == class_id)[0]
      if class_indices.numel() < 2:
        continue

      order = class_indices[
        torch.randperm(class_indices.numel(), device=data.device)
      ]
      partners = torch.roll(order, shifts=1)
      apply_mask = torch.rand(order.numel(), device=data.device) < probability
      target_indices = order[apply_mask]
      partner_indices = partners[apply_mask]
      if target_indices.numel() == 0:
        continue

      minimum = self.config.mixup_min_lambda
      lambdas = minimum + (1.0 - minimum) * torch.rand(
        target_indices.numel(),
        1,
        1,
        1,
        device=data.device,
        dtype=data.dtype,
      )
      mixed[target_indices] = (
        lambdas * source[target_indices]
        + (1.0 - lambdas) * source[partner_indices]
      )
    return mixed

  def _temporal_shift(self, data: torch.Tensor) -> torch.Tensor:
    maximum = self.config.max_shift_points
    if maximum == 0:
      return data

    batch_size, bands, channels, time_points = data.shape
    flattened = data.reshape(batch_size * bands * channels, 1, time_points)
    padded = F.pad(flattened, (maximum, maximum), mode="reflect").reshape(
      batch_size,
      bands,
      channels,
      time_points + 2 * maximum,
    )
    starts = torch.randint(
      0,
      2 * maximum + 1,
      (batch_size, 1),
      device=data.device,
    )
    time_indices = torch.arange(time_points, device=data.device).reshape(1, -1)
    gather_indices = (starts + time_indices).reshape(
      batch_size,
      1,
      1,
      time_points,
    )
    gather_indices = gather_indices.expand(
      batch_size,
      bands,
      channels,
      time_points,
    )
    return torch.gather(padded, dim=-1, index=gather_indices)

  def _amplitude_gain(self, data: torch.Tensor) -> torch.Tensor:
    minimum = self.config.gain_min
    maximum = self.config.gain_max
    if minimum == 1.0 and maximum == 1.0:
      return data

    gains = minimum + (maximum - minimum) * torch.rand(
      data.shape[0],
      1,
      1,
      1,
      device=data.device,
      dtype=data.dtype,
    )
    return data * gains

  def _channel_dropout(self, data: torch.Tensor) -> torch.Tensor:
    probability = self.config.channel_drop_probability
    if probability == 0.0:
      return data

    dropped = (
      torch.rand(
        data.shape[0],
        data.shape[2],
        device=data.device,
      )
      < probability
    )
    all_dropped = dropped.all(dim=1)
    dropped[all_dropped, 0] = False
    return data.masked_fill(dropped[:, None, :, None], 0.0)


class CrossSessionSynchronizedDualBandAugmentation(
  SynchronizedDualBandAugmentation
):
  """SDBA whose same-class Mixup partner must come from another session.

  Temporal shift, gain, and channel dropout retain the ordinary SDBA
  semantics.  Mixup is skipped for a target trial when its batch contains no
  same-class trial from a different session.  Hard labels and batch size are
  unchanged, and every stochastic decision remains synchronized across bands.
  """

  codename = CROSS_SESSION_SDBA_CODENAME

  def __call__(
    self,
    data: torch.Tensor,
    labels: torch.Tensor,
    session_ids: torch.Tensor,
  ) -> torch.Tensor:
    self._validate_input(data, labels)
    flat_sessions = session_ids.reshape(-1).to(device=data.device)
    if flat_sessions.numel() != data.shape[0]:
      raise ValueError("batch size and session ID count must match")
    augmented = self._cross_session_same_class_mixup(
      data,
      labels,
      flat_sessions,
    )
    augmented = self._temporal_shift(augmented)
    augmented = self._amplitude_gain(augmented)
    return self._channel_dropout(augmented)

  def _cross_session_same_class_mixup(
    self,
    data: torch.Tensor,
    labels: torch.Tensor,
    session_ids: torch.Tensor,
  ) -> torch.Tensor:
    probability = self.config.mixup_probability
    if probability == 0.0:
      return data.clone()

    flat_labels = labels.reshape(-1).to(device=data.device)
    mixed = data.clone()
    target_indices = torch.where(
      torch.rand(data.shape[0], device=data.device) < probability
    )[0]
    if target_indices.numel() == 0:
      return mixed

    partners = []
    eligible_targets = []
    for target_index in target_indices.tolist():
      candidates = torch.where(
        (flat_labels == flat_labels[target_index])
        & (session_ids != session_ids[target_index])
      )[0]
      if candidates.numel() == 0:
        continue
      partner_offset = torch.randint(
        candidates.numel(),
        (),
        device=data.device,
      )
      eligible_targets.append(target_index)
      partners.append(candidates[partner_offset])

    if not eligible_targets:
      return mixed

    target_tensor = torch.tensor(
      eligible_targets,
      device=data.device,
      dtype=torch.long,
    )
    partner_tensor = torch.stack(partners).long()
    minimum = self.config.mixup_min_lambda
    lambdas = minimum + (1.0 - minimum) * torch.rand(
      target_tensor.numel(),
      1,
      1,
      1,
      device=data.device,
      dtype=data.dtype,
    )
    mixed[target_tensor] = (
      lambdas * data[target_tensor]
      + (1.0 - lambdas) * data[partner_tensor]
    )
    return mixed
