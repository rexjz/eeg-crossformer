"""SEED-V Leave-One-Subject-Out DataModule with Apicella-2023 z0~z3 normalization.

Cross-subject evaluation on SEED-V: hold out one subject's three sessions
as test, randomly sample K validation subjects from the remaining 15
(all sessions), pool the rest as training.

Shares torcheeg's offline cache with `SEEDVSessionDependentDataModule` —
both use `derive_cache_dir(offline_transform)` so a single on-disk cache
serves both protocols.

z0~z3 framework (Apicella et al., EAAI 2023) now has non-degenerate
semantics on this protocol because we have *multiple* source subjects:
  - z0: pooled training stats applied to everyone
  - z1: per-subject (own) for train/val; pooled training for test
  - z2: per-subject (own) for everyone
  - z3: pooled training for train/val; own for test
Validation subjects share the "train" role (source-domain).
"""

from typing import Literal, Optional
import os
import random

import lightning as L
import numpy as np
import torch
import torch.utils.data
from torcheeg import transforms
from torcheeg.model_selection import Subcategory

from eegcrossformer.data.dataset.seedV.SEEDVSessionDependentDataModule import (
  SEEDDatasetWrapWithStats,
  _collect_subset_for_stats,
  _get_dataset,
  _default_offline_transform,
  derive_cache_dir,
)
from eegcrossformer.data.normalization import (
  ChannelStats,
  NormalizationStrategy,
  compute_channel_stats,
  pool_channel_stats,
  resolve_subject_stats,
)
from eegcrossformer.project_config import seedV_eeg_path, seedv_subjects_code


raw_eeg_folder = os.path.join(seedV_eeg_path, "EEG_raw")


def _get_all_subject_datasets(
  cache_dir, offline_transform, online_transform
):
  """Build the full SEED-V dataset and split it by subject_id.

  Returns: dict mapping subject_code (1..16) -> torcheeg Subset of that
  subject's chunks (all sessions concatenated).
  """
  seed_dataset = _get_dataset(cache_dir, offline_transform, online_transform)
  subject_split_path = os.path.join(cache_dir, "subject_split")
  cv = Subcategory(criteria="subject_id", split_path=subject_split_path)
  # cv.split returns one Subset per unique subject_id, in sorted order.
  # SEED-V has subject_ids 1..16 as strings; the first Subset corresponds
  # to subject "1", etc. We re-key by subject_code so callers can index
  # cleanly.
  subsets = list(cv.split(seed_dataset))
  return {str(i + 1): subsets[i] for i in range(len(subsets))}


class SEEDVLOSODataModule(L.LightningDataModule):
  """SEED-V LOSO DataModule with z0~z3 normalization.

  - Test set:        leftout subject's three sessions
  - Validation set:  K randomly selected non-leftout subjects (all sessions)
  - Training set:    remaining non-leftout subjects (all sessions)

  Args:
    leftout_subject_code: subject id ("1"..."16") held out for test.
    num_val_subjects: how many of the remaining subjects to use for val.
    offline_transform: torcheeg transform applied once at cache build time
      (cache is keyed on this; shared with session-dependent DataModule).
    online_transform: applied at every __getitem__ before z-score.
    normalization: 'none' / 'z0' / 'z1' / 'z2' / 'z3'.
    get_item_style: 'tuple' (default; matches OpenBMI LOSO) or 'dict'.
    use_default_offline_transform: when false, ``offline_transform=None``
      means raw 200-point EEG instead of the default two-band transform.
  """

  def __init__(
    self,
    leftout_subject_code: str,
    num_val_subjects: int = 3,
    all_subjects: Optional[list] = None,
    batch_size: int = 256,
    random_seed: int = 42,
    num_workers: int = 4,
    offline_transform: Optional[transforms.EEGTransform] = None,
    online_transform: Optional[transforms.EEGTransform] = None,
    get_item_style: Literal["dict", "tuple"] = "tuple",
    normalization: NormalizationStrategy = "none",
    stats_time_axis: Optional[int] = -1,
    train_shuffle: bool = True,
    use_default_offline_transform: bool = True,
  ):
    """Args:
      stats_time_axis: which axis of the per-sample feature tensor is
        "time" when computing channel stats. Default `-1` (raw / FB EEG
        `(V, C, T)`). Pass `None` for non-temporal feature vectors like
        DE (`(C, B)`) so band/feature axes are treated as features
        rather than being incorrectly pooled as time samples.
    """
    super().__init__()
    if all_subjects is None:
      all_subjects = list(seedv_subjects_code)  # ["1", ..., "16"]
    if leftout_subject_code not in all_subjects:
      raise ValueError(
        f"Invalid leftout_subject_code: {leftout_subject_code}. "
        f"Must be one of {all_subjects}"
      )
    if num_val_subjects < 1 or num_val_subjects >= len(all_subjects):
      raise ValueError(
        f"num_val_subjects ({num_val_subjects}) must be in [1, {len(all_subjects) - 1}]"
      )

    self.leftout_subject_code = leftout_subject_code
    self.num_val_subjects = num_val_subjects
    self.all_subjects = all_subjects
    self.batch_size = batch_size
    self.random_seed = random_seed
    self.num_workers = num_workers
    self.train_shuffle = train_shuffle

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
    self._per_subject_stats: dict[str, ChannelStats] = {}
    self.training_subjects: list[str] = []
    self.validation_subjects: list[str] = []

  def _split_subjects(self):
    rng = random.Random(self.random_seed)
    remaining = [s for s in self.all_subjects if s != self.leftout_subject_code]
    rng.shuffle(remaining)
    val = remaining[: self.num_val_subjects]
    train = remaining[self.num_val_subjects :]
    return train, val

  def _print_split_info(self, train, val):
    print("\n" + "=" * 70)
    print("SEED-V Leave-One-Subject-Out (LOSO) Cross Validation Setup")
    print("=" * 70)
    print(f"Total subjects: {len(self.all_subjects)}")
    print(f"Leftout subject (test): {self.leftout_subject_code}")
    print(f"Number of validation subjects: {self.num_val_subjects}")
    print(f"Random seed: {self.random_seed}")
    print(f"Normalization: {self.normalization}")
    print()
    print("Subject Split Results:")
    print(f"  Training subjects ({len(train)}): {train}")
    print(f"  Validation subjects ({len(val)}): {val}")
    print(f"  Test subject (1): [{self.leftout_subject_code}]")
    print("=" * 70 + "\n")

  # ---------------------------------------------------------------------
  # Stats orchestration
  # ---------------------------------------------------------------------

  def _compute_fit_stats(
    self, subject_codes: list[str], subject_to_subset: dict
  ) -> None:
    if self.normalization == "none":
      return

    needs_per_subject = self.normalization in ("z1", "z2")
    needs_pooled = self.normalization in ("z0", "z1", "z3")

    # Collect each subject's data array (iterating __getitem__ — slow but
    # only done once at setup('fit') time).
    per_subject_data = {}
    for sid in subject_codes:
      print(f"  Collecting stats data for subject {sid}...")
      per_subject_data[sid] = _collect_subset_for_stats(subject_to_subset[sid])

    if needs_per_subject:
      for sid, arr in per_subject_data.items():
        self._per_subject_stats[sid] = compute_channel_stats(
          arr, time_axis=self.stats_time_axis
        )

    if needs_pooled:
      flat = list(per_subject_data.values())
      self._pooled_train_stats = pool_channel_stats(
        flat, time_axis=self.stats_time_axis
      )

  def _compute_test_stats_if_needed(
    self, leftout: str, subject_to_subset: dict
  ) -> None:
    if self.normalization in ("z2", "z3") and leftout not in self._per_subject_stats:
      print(f"  Collecting stats data for test subject {leftout}...")
      arr = _collect_subset_for_stats(subject_to_subset[leftout])
      self._per_subject_stats[leftout] = compute_channel_stats(
        arr, time_axis=self.stats_time_axis
      )

    if self.normalization in ("z0", "z1") and self._pooled_train_stats is None:
      raise RuntimeError(
        f"normalization='{self.normalization}' needs pooled training stats — "
        "call setup('fit') before setup('test')."
      )

  def _resolve_stats(
    self, subject_code: str, role: Literal["train", "test"]
  ) -> Optional[ChannelStats]:
    if self.normalization == "none":
      return None
    own = self._per_subject_stats.get(subject_code)
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
    train, val = self._split_subjects()
    self.training_subjects = list(train)
    self.validation_subjects = list(val)
    self._print_split_info(train, val)

    subject_to_subset = _get_all_subject_datasets(
      self.cache_dir, self.offline_transform, self.online_transform
    )

    def wrap(sid: str, subset, role: Literal["train", "test"]):
      return SEEDDatasetWrapWithStats(
        subset,
        get_item_style=self.get_item_style,
        stats=self._resolve_stats(sid, role),
        extra_info={"domain_id": int(sid)},
      )

    if stage == "fit":
      if self.normalization != "none" and self._pooled_train_stats is None:
        self._compute_fit_stats(train + val, subject_to_subset)

      train_wraps = [wrap(sid, subject_to_subset[sid], "train") for sid in train]
      val_wraps = [wrap(sid, subject_to_subset[sid], "train") for sid in val]

      self.train_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(train_wraps),
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        shuffle=self.train_shuffle,
        persistent_workers=True if self.num_workers > 0 else False,
        pin_memory=True,
      )
      self.validation_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(val_wraps),
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        shuffle=False,
        persistent_workers=True if self.num_workers > 0 else False,
        pin_memory=True,
      )

    if stage == "test":
      self._compute_test_stats_if_needed(
        self.leftout_subject_code, subject_to_subset
      )
      test_wrap = wrap(
        self.leftout_subject_code,
        subject_to_subset[self.leftout_subject_code],
        "test",
      )
      self.test_loader = torch.utils.data.DataLoader(
        test_wrap,
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        shuffle=False,
        persistent_workers=True if self.num_workers > 0 else False,
        pin_memory=True,
      )

  def train_dataloader(self):
    return self.train_loader

  def val_dataloader(self):
    return self.validation_loader

  def test_dataloader(self):
    return self.test_loader


if __name__ == "__main__":
  dm = SEEDVLOSODataModule(
    leftout_subject_code="1",
    num_val_subjects=3,
    batch_size=256,
    normalization="z2",
  )
  print(f"cache dir: {os.path.basename(dm.cache_dir)}")
  dm.setup("fit")
  print(f"\nTrain loader batch count: {len(dm.train_dataloader())}")
  print(f"Val loader batch count:   {len(dm.val_dataloader())}")
  dm.setup("test")
  print(f"Test loader batch count:  {len(dm.test_dataloader())}")
