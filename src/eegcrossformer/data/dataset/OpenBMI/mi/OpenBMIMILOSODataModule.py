import torch
import lightning as L
import random
import multiprocessing as mp
import os
from multiprocessing import Pool
from typing import Literal, Optional
from torcheeg import transforms
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from eegcrossformer.data.dataset.OpenBMI.mi.OpenBMIMIDataset import (
  OpenBMI_MIDataset,
  OpenBMI_MIPaperMOABBSubjectDataset,
  paper_moabb_batch_dataloader,
)
from eegcrossformer.data.dataset.OpenBMI.mi.paper_moabb_transform_cache import (
  DEFAULT_EVENT_WINDOW_SECONDS,
  normalize_paper_moabb_event_window,
)
from eegcrossformer.data.dataset.OpenBMI.mi.cache_utils import (
  _load_and_cache_subject_safe,
  _generate_cache_key,
  OpenBMI_MIDataset_FromCache,
)
from eegcrossformer.data.normalization import (
  ChannelStats,
  NormalizationStrategy,
  compute_channel_stats,
  pool_channel_stats,
  resolve_subject_stats,
)
from eegcrossformer.project_config import openbmi_path


def load_subject_datasets(args):
  subject_id, root_path, transform, get_item_style, event_window_seconds = args
  return [
    OpenBMI_MIDataset(
      subject_id=subject_id,
      session="1",
      root_path=root_path,
      transform=transform,
      get_item_style=get_item_style,
      extra_info={"domain_id": int(subject_id)},
      event_window_seconds=event_window_seconds,
    ),
    OpenBMI_MIDataset(
      subject_id=subject_id,
      session="2",
      root_path=root_path,
      transform=transform,
      get_item_style=get_item_style,
      extra_info={"domain_id": int(subject_id)},
      event_window_seconds=event_window_seconds,
    ),
  ]


class OpenBMIMILOSODataModule(L.LightningDataModule):
  """OpenBMI Motor Imagery LOSO DataModule with Apicella-2023 Z-score variants.

  - Test set:        leftout subject's two sessions
  - Validation set:  K randomly selected non-leftout subjects (both sessions)
  - Training set:    remaining non-leftout subjects (both sessions)

  Normalization (`normalization` arg, per Apicella et al. 2023, EAAI 123:106205):
  - "none": no z-score
  - "z0":   pooled training stats applied to everyone
  - "z1":   per-subject for train/val, pooled training stats for test
  - "z2":   per-subject for everyone
  - "z3":   pooled training stats for train/val, own stats for test

  Validation subjects share the "train" role (they're source-domain data).
  Per-subject stats combine both sessions of that subject.
  Granularity: per-feature μ/σ over concatenated epochs (feature axes are
  everything between sample and time, so post-transform shapes like
  (N, V, C, T) get (V, C)-shaped stats automatically).
  """

  def __init__(
    self,
    leftout_subject_id: str,
    num_val_subjects: int = 5,
    root_path: str = openbmi_path,
    batch_size: int = 500,
    random_seed: int = 42,
    num_workers: int = 4,
    num_processes: Optional[int] = None,
    transform: Optional[transforms.EEGTransform] = None,
    get_item_style: Literal["dict", "tuple"] = "dict",
    normalization: NormalizationStrategy = "none",
    cache_dir: Optional[str] = None,
    use_cache: bool = True,
    force_refresh: bool = False,
    event_window_seconds: tuple[float, float] | None = None,
  ):
    super().__init__()

    if num_val_subjects < 1:
      raise ValueError(f"num_val_subjects must be at least 1, got {num_val_subjects}")
    if num_val_subjects >= 53:
      raise ValueError(
        f"num_val_subjects ({num_val_subjects}) must be less than 53 "
        f"(total subjects - 1 test subject). Recommended range: 1-20."
      )

    self.leftout_subject_id = leftout_subject_id
    self.num_val_subjects = num_val_subjects
    self.root_path = root_path
    self.batch_size = batch_size
    self.random_seed = random_seed
    self.num_workers = num_workers
    self.num_processes = num_processes
    self.transform = transform
    self.get_item_style = get_item_style
    self.normalization: NormalizationStrategy = normalization
    self.event_window_seconds = (
      normalize_paper_moabb_event_window(event_window_seconds)
      if event_window_seconds is not None
      else None
    )

    self.use_cache = use_cache
    self.force_refresh = force_refresh

    if cache_dir is None:
      self.cache_dir = os.path.join(root_path, ".eegcrossformer", "OpenBMI_MI_LOSO_Cache")
    else:
      self.cache_dir = cache_dir

    if self.use_cache:
      os.makedirs(self.cache_dir, exist_ok=True)

    self._pooled_train_stats: Optional[ChannelStats] = None
    self._per_subject_stats: dict[str, ChannelStats] = {}
    self.training_subjects: list[str] = []
    self.validation_subjects: list[str] = []

  def setup(self, stage: str):
    all_subjects = [str(i) for i in range(1, 55)]

    if self.leftout_subject_id not in all_subjects:
      raise ValueError(
        f"Invalid leftout_subject_id: {self.leftout_subject_id}. "
        f"Must be one of {all_subjects}"
      )

    rng = random.Random(self.random_seed)
    remaining_subjects = [s for s in all_subjects if s != self.leftout_subject_id]
    rng.shuffle(remaining_subjects)

    validation_subjects = remaining_subjects[: self.num_val_subjects]
    training_subjects = remaining_subjects[self.num_val_subjects :]
    self.training_subjects = list(training_subjects)
    self.validation_subjects = list(validation_subjects)

    self._print_split_info(all_subjects, training_subjects, validation_subjects)

    if stage == "fit":
      training_datasets = self._load_subjects_parallel(training_subjects, "training")
      validation_datasets = self._load_subjects_parallel(
        validation_subjects, "validation"
      )

      self._compute_fit_stats(
        training_subjects + validation_subjects,
        training_datasets + validation_datasets,
      )
      self._assign_stats(training_datasets, role="train")
      self._assign_stats(validation_datasets, role="train")

      self.train_loader = torch.utils.data.DataLoader(
        ConcatDataset(training_datasets),
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        shuffle=True,
        persistent_workers=True if self.num_workers > 0 else False,
        pin_memory=True,
        prefetch_factor=2 if self.num_workers > 0 else None,
      )
      self.validation_loader = torch.utils.data.DataLoader(
        ConcatDataset(validation_datasets),
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        shuffle=False,
        persistent_workers=True if self.num_workers > 0 else False,
        pin_memory=True,
        prefetch_factor=2 if self.num_workers > 0 else None,
      )

    if stage == "test":
      test_datasets = self._load_subjects_parallel([self.leftout_subject_id], "test")

      self._compute_subject_stats_if_needed(
        [self.leftout_subject_id], test_datasets, role="test"
      )
      self._assign_stats(test_datasets, role="test")

      self.test_loader = torch.utils.data.DataLoader(
        ConcatDataset(test_datasets),
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        shuffle=False,
        persistent_workers=True if self.num_workers > 0 else False,
        pin_memory=True,
        prefetch_factor=2 if self.num_workers > 0 else None,
      )

  # ---------------------------------------------------------------------
  # Normalization stats orchestration
  # ---------------------------------------------------------------------

  def _compute_fit_stats(self, subject_ids: list[str], datasets: list) -> None:
    """Compute per-subject + pooled stats over source-domain (train+val) data."""
    if self.normalization == "none":
      return

    by_subject: dict[str, list] = {}
    for ds in datasets:
      by_subject.setdefault(ds.subject_id, []).append(ds.data)

    needs_per_subject = self.normalization in ("z1", "z2")
    needs_pooled = self.normalization in ("z0", "z1", "z3")

    if needs_per_subject:
      for sid, arrs in by_subject.items():
        merged = arrs[0] if len(arrs) == 1 else _concat_along_first_axis(arrs)
        self._per_subject_stats[sid] = compute_channel_stats(merged)

    if needs_pooled:
      flat = [arr for arrs in by_subject.values() for arr in arrs]
      self._pooled_train_stats = pool_channel_stats(flat)

  def _compute_subject_stats_if_needed(
    self,
    subject_ids: list[str],
    datasets: list,
    role: Literal["train", "test"],
  ) -> None:
    if self.normalization in ("z2", "z3") and role == "test":
      by_subject: dict[str, list] = {}
      for ds in datasets:
        by_subject.setdefault(ds.subject_id, []).append(ds.data)
      for sid, arrs in by_subject.items():
        merged = arrs[0] if len(arrs) == 1 else _concat_along_first_axis(arrs)
        self._per_subject_stats[sid] = compute_channel_stats(merged)

    if role == "test" and self.normalization in ("z0", "z1") and self._pooled_train_stats is None:
      raise RuntimeError(
        f"normalization='{self.normalization}' needs pooled training stats — "
        "call setup('fit') before setup('test')."
      )

  def _assign_stats(self, datasets: list, role: Literal["train", "test"]) -> None:
    if self.normalization == "none":
      return
    for ds in datasets:
      own = self._per_subject_stats.get(ds.subject_id)
      ds.stats = resolve_subject_stats(
        strategy=self.normalization,
        role=role,
        own=own,
        pooled_train=self._pooled_train_stats,
      )

  # ---------------------------------------------------------------------
  # Loading (cache + no-cache paths)
  # ---------------------------------------------------------------------

  def _load_subjects_parallel(self, subject_ids: list, desc: str) -> list:
    if not self.use_cache:
      return self._load_subjects_no_cache(subject_ids, desc)

    load_args = []
    for subject_id in subject_ids:
      for session in ["1", "2"]:
        load_args.append(
          (
            subject_id,
            session,
            self.root_path,
            self.transform,
            self.cache_dir,
            self.force_refresh,
            self.event_window_seconds,
          )
        )

    cached_count = self._count_cached(load_args)
    total_count = len(load_args)

    if cached_count == total_count and not self.force_refresh:
      print(f"✓ All {total_count} subject-sessions found in cache")
    elif cached_count > 0 and not self.force_refresh:
      print(
        f"✓ Found {cached_count}/{total_count} cached, preprocessing {total_count - cached_count} remaining..."
      )
    else:
      print(
        f"⚙ Preprocessing and caching {total_count} subject-sessions (one-time cost)..."
      )

    ctx = mp.get_context("spawn")
    num_procs = min(self.num_processes or 8, 8)

    with ctx.Pool(processes=num_procs) as pool:
      results = list(
        tqdm(
          pool.imap(_load_and_cache_subject_safe, load_args),
          total=len(load_args),
          desc=f"Loading {desc} data",
        )
      )

    datasets = []
    for result in results:
      dataset = OpenBMI_MIDataset_FromCache(
        data=result["data"],
        labels=result["labels"],
        subject_id=result["subject_id"],
        session=result["session"],
        transform=self.transform,
        get_item_style=self.get_item_style,
        extra_info={"domain_id": int(result["subject_id"])},
        transform_applied=True,
      )
      datasets.append(dataset)

    return datasets

  def _count_cached(self, load_args: list) -> int:
    cached_count = 0
    for args in load_args:
      (
        subject_id,
        session,
        _,
        transform,
        cache_dir,
        force_refresh,
        event_window_seconds,
      ) = args
      if force_refresh:
        continue

      cache_key = _generate_cache_key(
        subject_id, session, transform, event_window_seconds
      )
      cache_file = os.path.join(cache_dir, f"{cache_key}.npz")
      if os.path.exists(cache_file):
        cached_count += 1
    return cached_count

  def _load_subjects_no_cache(self, subject_ids: list, desc: str) -> list:
    with Pool(processes=self.num_processes) as pool:
      results = list(
        tqdm(
          pool.imap(
            load_subject_datasets,
            [
              (
                subject_id,
                self.root_path,
                self.transform,
                self.get_item_style,
                self.event_window_seconds,
              )
              for subject_id in subject_ids
            ],
          ),
          total=len(subject_ids),
          desc=f"Loading {desc} data",
        )
      )
    return [dataset for subject_datasets in results for dataset in subject_datasets]

  def _print_split_info(self, all_subjects, training_subjects, validation_subjects):
    print("\n" + "=" * 70)
    print("Leave-One-Subject-Out (LOSO) Cross Validation Setup")
    print("=" * 70)
    print("Dataset: OpenBMI Motor Imagery")
    print(f"Total subjects: {len(all_subjects)}")
    print(f"Leftout subject (test): {self.leftout_subject_id}")
    print(f"Number of validation subjects: {self.num_val_subjects}")
    print(f"Random seed: {self.random_seed}")
    print(f"Normalization: {self.normalization}")
    print(f"Event window: {self.event_window_seconds or 'full legacy trial'}")
    print("Sessions per subject: 2 (both loaded)")
    print()
    print("Subject Split Results:")
    print(f"  Training subjects ({len(training_subjects)}): {training_subjects}")
    print(f"  Validation subjects ({len(validation_subjects)}): {validation_subjects}")
    print(f"  Test subject (1): [{self.leftout_subject_id}]")
    print("=" * 70 + "\n")

  def train_dataloader(self):
    return self.train_loader

  def val_dataloader(self):
    return self.validation_loader

  def test_dataloader(self):
    return self.test_loader


class OpenBMI_MIPaperMOABBLOSODataModule(L.LightningDataModule):
  """OpenBMI LOSO using the strict ADFCNN paper MOABB preprocessing path.

  Test set is the left-out subject's selected sessions. Validation subjects are
  sampled from the non-leftout subjects with the same deterministic split used
  by the standard OpenBMI LOSO module.
  """

  def __init__(
    self,
    leftout_subject_id: str,
    num_val_subjects: int = 5,
    batch_size: int = 128,
    random_seed: int = 42,
    num_workers: int = 0,
    sessions: tuple[str, ...] = ("1", "2"),
    channels: Literal["motor20", "all"] = "motor20",
    get_item_style: Literal["dict", "tuple"] = "tuple",
    source_subjects: Optional[list[str]] = None,
    offline_transform_profile: str | None = None,
    event_window_seconds: tuple[float, float] = DEFAULT_EVENT_WINDOW_SECONDS,
  ):
    super().__init__()

    if num_val_subjects < 1:
      raise ValueError(f"num_val_subjects must be at least 1, got {num_val_subjects}")
    if num_val_subjects >= 53:
      raise ValueError(
        f"num_val_subjects ({num_val_subjects}) must be less than 53 "
        f"(total subjects - 1 test subject). Recommended range: 1-20."
      )

    self.leftout_subject_id = leftout_subject_id
    self.num_val_subjects = num_val_subjects
    self.batch_size = batch_size
    self.random_seed = random_seed
    self.num_workers = num_workers
    self.sessions = sessions
    self.channels = channels
    self.get_item_style = get_item_style
    self.source_subjects = source_subjects
    self.offline_transform_profile = offline_transform_profile
    self.event_window_seconds = normalize_paper_moabb_event_window(
      event_window_seconds
    )
    self.training_subjects: list[str] = []
    self.validation_subjects: list[str] = []

  def setup(self, stage: str):
    all_subjects = [str(i) for i in range(1, 55)]

    if self.leftout_subject_id not in all_subjects:
      raise ValueError(
        f"Invalid leftout_subject_id: {self.leftout_subject_id}. "
        f"Must be one of {all_subjects}"
      )

    rng = random.Random(self.random_seed)
    if self.source_subjects is None:
      remaining_subjects = [s for s in all_subjects if s != self.leftout_subject_id]
    else:
      source_subjects = [str(subject) for subject in self.source_subjects]
      invalid_subjects = [
        subject for subject in source_subjects if subject not in all_subjects
      ]
      if invalid_subjects:
        raise ValueError(f"Invalid source_subjects: {invalid_subjects}")
      if self.leftout_subject_id in source_subjects:
        raise ValueError("source_subjects must not include the leftout subject")
      if len(source_subjects) <= self.num_val_subjects:
        raise ValueError(
          "source_subjects must contain more subjects than num_val_subjects"
        )
      remaining_subjects = source_subjects
    rng.shuffle(remaining_subjects)

    validation_subjects = remaining_subjects[: self.num_val_subjects]
    training_subjects = remaining_subjects[self.num_val_subjects :]
    self.training_subjects = list(training_subjects)
    self.validation_subjects = list(validation_subjects)

    self._print_split_info(all_subjects, training_subjects, validation_subjects)

    if stage == "fit":
      training_datasets = self._load_subjects(training_subjects, "training")
      validation_datasets = self._load_subjects(validation_subjects, "validation")

      self.train_loader = paper_moabb_batch_dataloader(
        training_datasets,
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        shuffle=True,
      )
      self.validation_loader = paper_moabb_batch_dataloader(
        validation_datasets,
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        shuffle=False,
      )

    if stage == "test":
      test_datasets = self._load_subjects([self.leftout_subject_id], "test")
      self.test_loader = paper_moabb_batch_dataloader(
        test_datasets,
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        shuffle=False,
      )

  def _load_subjects(self, subject_ids: list[str], desc: str) -> list:
    return [
      OpenBMI_MIPaperMOABBSubjectDataset(
        subject_id=subject_id,
        sessions=self.sessions,
        channels=self.channels,
        get_item_style=self.get_item_style,
        extra_info={"domain_id": int(subject_id)},
        offline_transform_profile=self.offline_transform_profile,
        event_window_seconds=self.event_window_seconds,
      )
      for subject_id in tqdm(subject_ids, desc=f"Loading {desc} paper-MOABB data")
    ]

  def _print_split_info(self, all_subjects, training_subjects, validation_subjects):
    print("\n" + "=" * 70)
    print("Leave-One-Subject-Out (LOSO) Cross Validation Setup")
    print("=" * 70)
    print("Dataset: OpenBMI Motor Imagery")
    print("Preprocessing: ADFCNN paper MOABB raw-continuous protocol")
    print(f"Total subjects: {len(all_subjects)}")
    print(f"Leftout subject (test): {self.leftout_subject_id}")
    print(f"Number of validation subjects: {self.num_val_subjects}")
    print(f"Random seed: {self.random_seed}")
    print(f"Sessions per subject: {list(self.sessions)}")
    n_channels = 62 if self.channels == "all" else 20
    print(f"Channels: {self.channels}")
    start, stop = self.event_window_seconds
    n_times = round((stop - start) * 250)
    print(f"Raw epoch shape: [1, {n_channels}, {n_times}]")
    print(f"Cue-aligned event window: {self.event_window_seconds}")
    print(f"Offline transform profile: {self.offline_transform_profile}")
    print()
    print("Subject Split Results:")
    print(f"  Training subjects ({len(training_subjects)}): {training_subjects}")
    print(f"  Validation subjects ({len(validation_subjects)}): {validation_subjects}")
    print(f"  Test subject (1): [{self.leftout_subject_id}]")
    print("=" * 70 + "\n")

  def train_dataloader(self):
    return self.train_loader

  def val_dataloader(self):
    return self.validation_loader

  def test_dataloader(self):
    return self.test_loader


def _concat_along_first_axis(arrs: list) -> "np.ndarray":  # noqa: F821
  import numpy as np

  return np.concatenate(arrs, axis=0)


if __name__ == "__main__":
  print("Testing OpenBMIMILOSODataModule with normalization='z2'...")
  data_module = OpenBMIMILOSODataModule(
    leftout_subject_id="1",
    num_val_subjects=5,
    num_processes=4,
    use_cache=True,
    normalization="z2",
  )
  data_module.setup("fit")
  print(f"\nTrain loader batch count: {len(data_module.train_dataloader())}")
  print(f"Val loader batch count: {len(data_module.val_dataloader())}")

  data_module.setup("test")
  print(f"Test loader batch count: {len(data_module.test_dataloader())}")
