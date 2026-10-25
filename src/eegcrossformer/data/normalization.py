"""Z-score normalization strategies from Apicella et al., EAAI 2023.

Four strategies (Z0/Z1/Z2/Z3) differ in which (μ, σ) is used for training-set
subjects vs the held-out (test) subject in a LOSO setup. We compute per-feature
stats where the "feature" axes are everything between the sample axis and the
time axis. For raw EEG of shape (N, C, T) that's (C,); for filter-bank EEG of
shape (N, V, C, T) it's (V, C); for view-stacked EEG of shape (N, V*C, T) it's
(V*C,). Stats are pooled over the N and T axes.

This matches the paper's "per-feature" interpretation when X is a flattened
feature vector and stays statistically stable for raw time-series by treating
time as additional samples within each feature.

Validation subjects share the "train" role (they belong to the source-domain
pool used for pooled stats and follow the train-side rule for Z1 / Z3).

Pipeline order
--------------
Stats are applied at __getitem__ on top of the cached, *post*-DataModule-
transform tensor. Concretely:

  raw EEG -> resample -> DataModule transform (e.g. MeanStdNormalize,
  FilterBank, MultiViewStacking) [cached]
          -> __getitem__: apply ChannelStats (z0/z1/z2/z3)
          -> DataLoader batch -> batch augmentations (RandomCrop, CutMix,
             RandomErasing, ...)
          -> model

That is, batch augmentations see *already-normalized* samples. This differs
from the ImageNet convention (augment-then-normalize), and is deliberate:

- Stats are computed in the *model-input* feature space. If we filter-bank
  first then z-score, the stats' shape is (V, C) and each (band, channel)
  ends up unit-variance — which is what the model actually wants. Doing
  z-score on raw (C,) before the filter bank would leave the post-filter
  bands at very different scales.
- For the current augmentation set this ordering is benign or beneficial:
  RandomCrop is invariant; CutMix linearly mixes two normalized samples
  (still ~unit variance when both come from the same stats pool); for
  RandomErasing, "fill with zero" on normalized data corresponds to "fill
  with the per-feature mean", which is a sensible neutral value.
- Caveat: amplitude-scaling / gain-perturbation augmentations must go
  *before* normalization, or the std absorbs the perturbation. Likewise,
  cross-domain CutMix under Z1/Z3 mixes samples normalized by different
  stats and will produce inconsistent scale — keep mixing within the
  source-domain pool.
"""

from typing import Literal, Optional

import numpy as np
import torch


NormalizationStrategy = Literal["none", "z0", "z1", "z2", "z3"]
SubjectRole = Literal["train", "test"]


class ChannelStats:
  """Per-feature mean/std for either time-series or pure-feature data.

  Two shape conventions are supported, distinguished by `has_time`:

  - `has_time=True`  (raw / filter-bank EEG): per-sample shape is
    `(*feature_shape, T)`. Stats pool over `T*N` time-sample points.
  - `has_time=False` (DE / band-power / any non-temporal feature vector):
    per-sample shape is `(*feature_shape,)`. Stats pool over `N` samples
    only.

  In both cases `mean.shape == std.shape == feature_shape`.
  """

  def __init__(
    self,
    mean: np.ndarray,
    std: np.ndarray,
    has_time: bool = True,
  ):
    self.mean = np.asarray(mean, dtype=np.float32)
    self.std = np.asarray(std, dtype=np.float32) + 1e-6
    self.has_time = has_time
    if self.mean.shape != self.std.shape:
      raise ValueError(
        f"mean/std shape mismatch: {self.mean.shape} vs {self.std.shape}"
      )

  @property
  def feature_shape(self) -> tuple:
    return tuple(self.mean.shape)

  def _broadcast_shape(self, x_shape: tuple) -> tuple:
    """Compute the shape to reshape mean/std to so they broadcast onto x.

    For `has_time=True`: returns `(*leading_ones, *feature_shape, 1)`
    matching either `(*feature_shape, T)` or `(N, *feature_shape, T)`.

    For `has_time=False`: returns `(*leading_ones, *feature_shape)`
    matching either `(*feature_shape,)` or `(N, *feature_shape)`.
    """
    feat = self.feature_shape
    n_feat = len(feat)

    if self.has_time:
      min_ndim = n_feat + 1  # at least (*feat, T)
      if len(x_shape) < min_ndim:
        raise ValueError(
          f"Cannot apply has_time=True stats of feature_shape={feat} "
          f"to array of shape={x_shape}"
        )
      if tuple(x_shape[-n_feat - 1 : -1]) != feat:
        raise ValueError(
          f"Feature dims mismatch: stats expect {feat} at "
          f"axes [-{n_feat + 1}:-1], got {x_shape}"
        )
      leading = (1,) * (len(x_shape) - n_feat - 1)
      return leading + feat + (1,)

    # has_time=False: last axes are feature_shape, no trailing T.
    min_ndim = n_feat
    if len(x_shape) < min_ndim:
      raise ValueError(
        f"Cannot apply has_time=False stats of feature_shape={feat} "
        f"to array of shape={x_shape}"
      )
    if tuple(x_shape[-n_feat:]) != feat:
      raise ValueError(
        f"Feature dims mismatch: stats expect {feat} as the trailing "
        f"axes, got {x_shape}"
      )
    leading = (1,) * (len(x_shape) - n_feat)
    return leading + feat

  def apply(self, X: np.ndarray) -> np.ndarray:
    target_shape = self._broadcast_shape(X.shape)
    m = self.mean.reshape(target_shape)
    s = self.std.reshape(target_shape)
    return (X - m) / s

  def apply_torch(self, x: torch.Tensor) -> torch.Tensor:
    target_shape = self._broadcast_shape(tuple(x.shape))
    mean_t = torch.as_tensor(self.mean, device=x.device, dtype=x.dtype).reshape(
      target_shape
    )
    std_t = torch.as_tensor(self.std, device=x.device, dtype=x.dtype).reshape(
      target_shape
    )
    return (x - mean_t) / std_t


def compute_channel_stats(
  data: np.ndarray, time_axis: Optional[int] = -1
) -> ChannelStats:
  """Per-feature mean/std from a stack of samples.

  Args:
    data: numpy array shaped `(N, *feature_shape[, T])`. The leading axis
      is samples (N).
    time_axis: which axis is "time" — pooled together with N when computing
      stats. Default `-1` (last axis) matches raw / filter-bank EEG. Pass
      `None` to treat the entire non-sample shape as features (no time
      pooling). Appropriate for DE / band-power / any non-temporal
      feature vector. Negative ints are normalised; the sample axis (0)
      cannot be the time axis.

  Returns:
    ChannelStats whose `mean`/`std` shape equals `feature_shape`.
  """
  if data.ndim < 2:
    raise ValueError(f"Expected at least 2D (N, ...), got shape {data.shape}")

  if time_axis is None:
    # All non-sample axes are features. Pool only over N.
    N = data.shape[0]
    feat = data.shape[1:]
    if not feat:
      raise ValueError("Need at least one feature axis after N")
    flat = data.reshape(N, -1)  # (N, prod(feat))
    mean = flat.mean(axis=0).reshape(feat)
    std = flat.std(axis=0).reshape(feat)
    return ChannelStats(mean=mean, std=std, has_time=False)

  # Normalise negative axis index. After resolution, time_axis must be in
  # [1, data.ndim - 1] (the 0-th axis is reserved for samples).
  ax = time_axis if time_axis >= 0 else data.ndim + time_axis
  if ax == 0 or ax < 0 or ax >= data.ndim:
    raise ValueError(
      f"time_axis={time_axis} out of range for data shape {data.shape}"
    )
  # Move the chosen time axis to the last position, then run the original
  # (N, *feat, T) -> stats path.
  data = np.moveaxis(data, ax, -1)
  if data.ndim < 3:
    raise ValueError(
      f"Need at least one feature axis between N and T, got shape {data.shape}"
    )
  N, *feat, T = data.shape
  reordered = np.moveaxis(data, 0, -1)  # (*feat, T, N) — T stays before N
  flat = reordered.reshape(*feat, T * N) if feat else reordered.reshape(T * N)
  if not feat:
    raise ValueError("Need at least one feature axis between N and T")
  return ChannelStats(
    mean=flat.mean(axis=-1), std=flat.std(axis=-1), has_time=True
  )


def pool_channel_stats(
  data_list: list[np.ndarray], time_axis: Optional[int] = -1
) -> ChannelStats:
  """Concatenate per-subject arrays along axis 0 and compute pooled stats.

  `time_axis` has the same semantics as in `compute_channel_stats`.
  """
  if not data_list:
    raise ValueError("Cannot pool stats from empty list")
  concat = np.concatenate(data_list, axis=0)
  return compute_channel_stats(concat, time_axis=time_axis)


def resolve_subject_stats(
  strategy: NormalizationStrategy,
  role: SubjectRole,
  own: Optional[ChannelStats],
  pooled_train: Optional[ChannelStats],
) -> Optional[ChannelStats]:
  """Pick the (μ, σ) to apply to a subject under the given strategy.

  role="train" covers actual training subjects AND validation subjects (both
  are source-domain data). role="test" is the leftout target subject.
  """
  if strategy == "none":
    return None
  if strategy == "z0":
    return pooled_train
  if strategy == "z1":
    return own if role == "train" else pooled_train
  if strategy == "z2":
    return own
  if strategy == "z3":
    return pooled_train if role == "train" else own
  raise ValueError(f"Unknown normalization strategy: {strategy}")
