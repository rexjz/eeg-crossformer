"""SEEDV Session-Dependent K-Fold DataModule — hash-keyed cache + z0~z3 normalization.

A single class supports any `offline_transform` (filter-bank, DE, raw, etc.):

  * `io_path` is derived from a deterministic hash of (offline_transform,
    chunk_size, dataset_version) — different transform configs automatically
    use different on-disk caches without collision.
  * The hashing strategy mirrors `OpenBMI/mi/cache_utils.py::_generate_cache_key`
    (via `data/cache_common.py::serialize_transform`).
  * torcheeg's `BaseDataset` performs the actual offline caching at the
    derived `io_path`; we just route it.
  * z-score normalization is applied at `__getitem__` via `ChannelStats`
    (none / z0 / z1 / z2 / z3), so a single offline cache serves all five
    strategies.

z0~z3 semantics in this protocol:
Session-Dependent K-Fold splits a single (subject, session) into K folds, so
the "subject" axis in Apicella's framework collapses to one entity:
  - source pool = train + val trials (current fold's complement)
  - target pool = test fold trials
With one source entity, z0 ≡ z1 (target-blind) and z2 ≡ z3 (target uses
its own pool stats). All four names are still accepted for API parity with
`OpenBMIMILOSODataModule`.
"""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from typing import Literal, Optional

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torcheeg import transforms
from torcheeg.datasets import SEEDVDataset
from torcheeg.model_selection import Subcategory

from eegcrossformer.data.cache_common import serialize_transform
from eegcrossformer.data.normalization import (
  ChannelStats,
  NormalizationStrategy,
  compute_channel_stats,
  resolve_subject_stats,
)
from eegcrossformer.data.transforms.fb_transform import TrailFilteringBankTransform
from eegcrossformer.project_config import seedV_eeg_path


DATASET_VERSION = "v1"
CHUNK_SIZE = 200
# Use 'pickle' io_mode instead of torcheeg's default 'lmdb' — LMDB raises
# "environment already open in this process" when Subcategory.split() deep-
# copies the dataset (it tries to open the same LMDB path a second time
# while the original still holds it). PickleEEGSignalIO has a shallow
# __copy__ and is unaffected.
IO_MODE = "pickle"
raw_eeg_folder = os.path.join(seedV_eeg_path, "EEG_raw")


# ---------------------------------------------------------------------
# Cache key derivation (mirrors OpenBMI's _generate_cache_key pattern)
# ---------------------------------------------------------------------


def _default_offline_transform() -> transforms.EEGTransform:
  """FilterBank-only — z-score is applied later in __getitem__ via stats."""
  return transforms.Compose(
    [
      TrailFilteringBankTransform(freq_bands=[[1, 16], [16, 40]], sfreq=200),
    ]
  )


def derive_cache_dir(offline_transform: Optional[transforms.EEGTransform]) -> str:
  """Hash (offline_transform, chunk_size, dataset_version) -> stable cache dir.

  Same offline_transform configuration always maps to the same directory, so
  multiple DataModule instances share the on-disk cache. Different configs
  map to different directories, never colliding.
  """
  config = {
    "offline_transform": serialize_transform(offline_transform),
    "chunk_size": CHUNK_SIZE,
    "io_mode": IO_MODE,
    "dataset_version": DATASET_VERSION,
  }
  config_str = json.dumps(config, sort_keys=True)
  digest = hashlib.md5(config_str.encode()).hexdigest()[:12]
  return os.path.join(seedV_eeg_path, ".torcheeg", f"seedv_{digest}")


# ---------------------------------------------------------------------
# Dataset assembly helpers (cache-dir aware)
# ---------------------------------------------------------------------


def _get_dataset(cache_dir, offline_transform, online_transform):
  return SEEDVDataset(
    io_path=os.path.join(cache_dir, "dataset"),
    root_path=raw_eeg_folder,
    chunk_size=CHUNK_SIZE,
    num_worker=8,
    io_mode=IO_MODE,
    offline_transform=offline_transform,
    online_transform=online_transform,
    label_transform=transforms.Compose(
      [
        transforms.Select("emotion"),
      ]
    ),
  )


def _date_column_element_transform(x):
  if pd.isna(x):
    raise ValueError("SEED-V subject metadata contains a missing date")
  if isinstance(x, (int, np.integer)):
    return int(x)
  if isinstance(x, (float, np.floating)):
    if not np.isfinite(x) or not float(x).is_integer():
      raise ValueError(f"Invalid numeric SEED-V date: {x!r}")
    return int(x)
  if isinstance(x, str):
    value = x.removesuffix(".cnt")
    try:
      return int(value)
    except ValueError as exc:
      raise ValueError(f"Invalid string SEED-V date: {x!r}") from exc
  raise TypeError(f"Unsupported SEED-V date value: {x!r}")


@contextmanager
def _subject_metadata_lock(cache_dir: str, subject_code: str):
  """Serialize shared subject/session metadata reads and repairs."""
  os.makedirs(cache_dir, exist_ok=True)
  lock_path = os.path.join(cache_dir, f".subject_{subject_code}.metadata.lock")
  with open(lock_path, "a+", encoding="utf-8") as lock_file:
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
    try:
      yield
    finally:
      fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _normalize_date_column_unlocked(abs_path: str) -> None:
  if not os.path.exists(abs_path):
    return

  df = pd.read_csv(abs_path, low_memory=False)
  stale_index_columns = [
    column for column in df.columns if str(column).startswith("Unnamed:")
  ]
  if stale_index_columns:
    df = df.drop(columns=stale_index_columns)

  changed = bool(stale_index_columns)
  if "date" in df.columns:
    normalized_dates = df["date"].map(_date_column_element_transform)
    changed = changed or not normalized_dates.equals(df["date"])
    df["date"] = normalized_dates

  if not changed:
    return

  output_dir = os.path.dirname(abs_path)
  fd, temp_path = tempfile.mkstemp(
    dir=output_dir,
    prefix=f".{os.path.basename(abs_path)}.",
    suffix=".tmp",
  )
  try:
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as temp_file:
      df.to_csv(temp_file, index=False)
      temp_file.flush()
      os.fsync(temp_file.fileno())
    os.replace(temp_path, abs_path)
  finally:
    if os.path.exists(temp_path):
      os.unlink(temp_path)


def _ensure_date_column_fixed(cache_dir: str, subject_code: str) -> None:
  """Atomically normalize `date` in subject_split/{code}.csv.

  torcheeg writes the date column in a mix of int/float/str-with-extension
  formats; downstream date-based Subcategory expects clean ints. A per-subject
  process lock and atomic replacement protect concurrent experiment workers.
  """
  abs_path = os.path.join(cache_dir, "subject_split", f"{subject_code}.csv")
  with _subject_metadata_lock(cache_dir, subject_code):
    _normalize_date_column_unlocked(abs_path)


def _select_subject_dataset(seed_dataset, cache_dir, subject_code):
  subject_split_path = os.path.join(cache_dir, "subject_split")
  cv = Subcategory(criteria="subject_id", split_path=subject_split_path)
  subject_dataset = None
  target = int(subject_code) - 1
  for i, ds in enumerate(cv.split(seed_dataset)):
    if target == i:
      subject_dataset = ds
      break
  return subject_dataset


def _get_subject_dataset(
  cache_dir, subject_code, offline_transform, online_transform
):
  seed_dataset = _get_dataset(cache_dir, offline_transform, online_transform)
  return _select_subject_dataset(seed_dataset, cache_dir, subject_code)


def _get_session_datasets(
  cache_dir, subject_code, offline_transform, online_transform
):
  # Cache construction can create worker processes, so finish it before
  # acquiring a metadata lock that child processes could otherwise inherit.
  seed_dataset = _get_dataset(cache_dir, offline_transform, online_transform)
  subject_csv = os.path.join(
    cache_dir, "subject_split", f"{subject_code}.csv"
  )
  with _subject_metadata_lock(cache_dir, subject_code):
    # Repair an existing CSV before Subcategory can read it. Keep the lock
    # through session split discovery because torcheeg may create split files.
    _normalize_date_column_unlocked(subject_csv)
    subject_dataset = _select_subject_dataset(
      seed_dataset, cache_dir, subject_code
    )
    # The first Subcategory call may have created the subject CSV.
    _normalize_date_column_unlocked(subject_csv)
    if "date" in subject_dataset.info.columns:
      subject_dataset.info["date"] = subject_dataset.info["date"].map(
        _date_column_element_transform
      )
    split_path = os.path.join(
      cache_dir, f"subject{subject_code.zfill(2)}_session_split"
    )
    # Stale '<date>.cnt.csv' files from prior failed runs would otherwise be
    # reused as-is — wipe and let Subcategory regenerate from the cleaned info.
    if os.path.exists(split_path):
      if any(".cnt" in f for f in os.listdir(split_path)):
        shutil.rmtree(split_path)
    session_cv = Subcategory(criteria="date", split_path=split_path)
    return list(session_cv.split(subject_dataset))


def _collect_subset_for_stats(subset) -> np.ndarray:
  """Iterate a Subset and stack data into (N, *feat, T) float32 ndarray."""
  arrs = []
  for i in range(len(subset)):
    item = subset[i]
    data = item[0] if isinstance(item, tuple) else item
    if isinstance(data, torch.Tensor):
      data = data.detach().cpu().numpy()
    elif not isinstance(data, np.ndarray):
      data = np.asarray(data)
    arrs.append(data)
  return np.stack(arrs, axis=0).astype(np.float32)


# ---------------------------------------------------------------------
# Stats-aware wrapper
# ---------------------------------------------------------------------


class SEEDDatasetWrapWithStats(torch.utils.data.Dataset):
  """Wraps a Subset and applies a ChannelStats at __getitem__.

  Mirrors `OpenBMI_MIDataset_FromCache.stats`: stats may be None
  (passthrough) or a `ChannelStats` instance broadcast across the feature
  axes of each sample. Supports `dict` and `tuple` return styles.

  `extra_info` (default None) is propagated as the third tuple element
  (or under the `"extra_info"` key in dict mode). Used to carry e.g.
  `{"domain_id": <int>}` for WithinDomainAugmentation in LOSO setups.
  """

  def __init__(
    self,
    dataset,
    data_key: str = "data",
    get_item_style: Literal["dict", "tuple"] = "dict",
    stats: Optional[ChannelStats] = None,
    extra_info: Optional[dict] = None,
  ):
    super().__init__()
    self.dataset = dataset
    self.data_key = data_key
    self.get_item_style = get_item_style
    self.stats = stats
    self.extra_info = extra_info

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    item = self.dataset[idx]
    data, target = item if isinstance(item, tuple) else (item, None)

    if self.stats is not None:
      if isinstance(data, torch.Tensor):
        data = self.stats.apply_torch(data)
      else:
        data = self.stats.apply(np.asarray(data, dtype=np.float32)).astype(
          np.float32
        )

    if self.get_item_style == "dict":
      out = {self.data_key: data}
      if target is not None:
        out["target"] = target
      if self.extra_info is not None:
        out["extra_info"] = self.extra_info
      return out
    else:
      return (data, target, dict(self.extra_info) if self.extra_info else {})


# ---------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------


class SEEDVSessionDependentDataModule(L.LightningDataModule):
  """Unified SEEDV Session-Dependent K-Fold DataModule.

  Cache dir is auto-derived from the offline_transform's hash. z0~z3
  normalization is applied at __getitem__ on top of the cached output.

  Args:
    offline_transform: torcheeg transform applied once at cache build time.
      Defaults to FilterBank [[1,16], [16,40]] @ sfreq=200. Pass any
      `transforms.Compose([...])` to override.
    online_transform: torcheeg transform applied at every __getitem__
      *before* z-score (e.g. `transforms.ToTensor()`).
    normalization: one of 'none', 'z0', 'z1', 'z2', 'z3'. See module
      docstring for in-session semantics.
  """

  def __init__(
    self,
    subject_code="1",
    session_id="1",
    data_dir: str = raw_eeg_folder,
    batch_size: int = 32,
    train_test_seed=42,
    train_val_seed=42,
    num_workers=1,
    kfold: int = 5,
    test_fold: int = 1,
    kfold_seed: int = None,
    train_shuffle=True,
    offline_transform: Optional[transforms.EEGTransform] = None,
    online_transform: Optional[transforms.EEGTransform] = None,
    get_item_style: Literal["dict", "tuple"] = "dict",
    normalization: NormalizationStrategy = "none",
    stats_time_axis: Optional[int] = -1,
    use_default_offline_transform: bool = True,
  ):
    """Args:
      stats_time_axis: which axis of the per-sample feature tensor is
        "time" when computing channel stats. Default `-1` (raw / FB EEG
        `(V, C, T)`-style data). Pass `None` for non-temporal feature
        vectors like DE (`(C, B)`) so band/feature axes aren't
        incorrectly pooled as if they were time samples.
    """
    super().__init__()
    self.data_dir = data_dir
    self.batch_size = batch_size
    self.train_test_seed = train_test_seed
    self.train_val_seed = train_val_seed
    self.num_workers = num_workers
    self.subject_code = subject_code
    self.session_id = session_id
    self.kfold = kfold
    self.test_fold = test_fold
    self.kfold_seed = kfold_seed if kfold_seed is not None else train_test_seed
    self.train_shuffle = train_shuffle
    if self.test_fold > self.kfold or self.test_fold < 1:
      raise ValueError(f"test_fold must be between 1 and {self.kfold}")

    self.offline_transform = (
      offline_transform
      if offline_transform is not None or not use_default_offline_transform
      else _default_offline_transform()
    )
    self.online_transform = online_transform
    self.cache_dir = derive_cache_dir(self.offline_transform)
    self.get_item_style = get_item_style
    self.normalization: NormalizationStrategy = normalization
    self.stats_time_axis = stats_time_axis

    self._pooled_train_stats: Optional[ChannelStats] = None
    self._test_stats: Optional[ChannelStats] = None

  # ---------------------------------------------------------------------
  # Split + stats helpers
  # ---------------------------------------------------------------------

  def _split_indices(self, n_trials: int):
    """Reproducible (train, val, test) indices for the current fold."""
    fold_size = n_trials // self.kfold
    indices = torch.randperm(
      n_trials, generator=torch.Generator().manual_seed(self.kfold_seed)
    ).tolist()
    test_indices = indices[
      (self.test_fold - 1) * fold_size : self.test_fold * fold_size
    ]
    train_indices = [i for i in indices if i not in test_indices]
    split_point = int(len(train_indices) * 0.8)
    val_indices = train_indices[split_point:]
    train_indices = train_indices[:split_point]
    return train_indices, val_indices, test_indices, indices

  def _resolve_stats_for_role(
    self, role: Literal["train", "test"]
  ) -> Optional[ChannelStats]:
    """Pick the ChannelStats to apply for this role under self.normalization.

    In single-(subject,session) k-fold, "own" for a source-domain trial is
    the source pool itself, so we pass `_pooled_train_stats` as `own` for
    the train role.
    """
    if self.normalization == "none":
      return None
    own = self._pooled_train_stats if role == "train" else self._test_stats
    return resolve_subject_stats(
      strategy=self.normalization,
      role=role,
      own=own,
      pooled_train=self._pooled_train_stats,
    )

  # ---------------------------------------------------------------------
  # Lightning hook
  # ---------------------------------------------------------------------

  def setup(self, stage: str):
    session_datasets = _get_session_datasets(
      self.cache_dir,
      self.subject_code,
      offline_transform=self.offline_transform,
      online_transform=self.online_transform,
    )
    dataset = session_datasets[int(self.session_id) - 1]

    train_indices, val_indices, test_indices, indices = self._split_indices(
      len(dataset)
    )

    train_set = torch.utils.data.Subset(dataset, train_indices)
    validation_set = torch.utils.data.Subset(dataset, val_indices)
    test_set = torch.utils.data.Subset(dataset, test_indices)

    needs_pooled = self.normalization in ("z0", "z1", "z3")
    needs_test_own = self.normalization in ("z2", "z3")

    if stage == "fit":
      if self.normalization != "none" and self._pooled_train_stats is None:
        source_subset = torch.utils.data.ConcatDataset(
          [train_set, validation_set]
        )
        source_data = _collect_subset_for_stats(source_subset)
        self._pooled_train_stats = compute_channel_stats(
          source_data, time_axis=self.stats_time_axis
        )

      train_stats = self._resolve_stats_for_role("train")
      val_stats = self._resolve_stats_for_role("train")

      self.train_loader = torch.utils.data.DataLoader(
        SEEDDatasetWrapWithStats(
          train_set,
          get_item_style=self.get_item_style,
          stats=train_stats,
        ),
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        shuffle=self.train_shuffle,
      )
      self.validation_loader = torch.utils.data.DataLoader(
        SEEDDatasetWrapWithStats(
          validation_set,
          get_item_style=self.get_item_style,
          stats=val_stats,
        ),
        batch_size=self.batch_size,
        num_workers=self.num_workers,
      )

    if stage == "test":
      if needs_pooled and self._pooled_train_stats is None:
        raise RuntimeError(
          f"normalization='{self.normalization}' needs pooled training stats — "
          "call setup('fit') before setup('test')."
        )
      if needs_test_own and self._test_stats is None:
        test_data = _collect_subset_for_stats(test_set)
        self._test_stats = compute_channel_stats(
          test_data, time_axis=self.stats_time_axis
        )

      test_stats = self._resolve_stats_for_role("test")

      self.test_loader = torch.utils.data.DataLoader(
        SEEDDatasetWrapWithStats(
          test_set,
          get_item_style=self.get_item_style,
          stats=test_stats,
        ),
        batch_size=self.batch_size,
        num_workers=self.num_workers,
      )
    return train_indices, val_indices, test_indices, indices

  def train_dataloader(self):
    return self.train_loader

  def val_dataloader(self):
    return self.validation_loader

  def test_dataloader(self):
    return self.test_loader


if __name__ == "__main__":
  # Smoke: hash determinism
  t1 = _default_offline_transform()
  t2 = _default_offline_transform()
  assert derive_cache_dir(t1) == derive_cache_dir(t2), "hash must be deterministic"
  t3 = transforms.Compose(
    [
      transforms.MeanStdNormalize(axis=0),
      TrailFilteringBankTransform(freq_bands=[[1, 16], [16, 40]], sfreq=100),
    ]
  )
  assert derive_cache_dir(t1) != derive_cache_dir(t3), "different transforms must hash differently"
  print(f"default cache dir: {derive_cache_dir(t1).split(os.sep)[-1]}")
  print(f"legacy cache dir:  {derive_cache_dir(t3).split(os.sep)[-1]}")
  print("hash determinism OK")
