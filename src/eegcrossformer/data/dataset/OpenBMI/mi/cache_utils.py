"""Caching utilities for OpenBMI dataset.

Cache stores POST-transform epochs keyed by ``(subject, session, transform,
resample_factor)`` so per-fetch filter banks / view stacking are skipped at
training time. Z-score normalization is applied at __getitem__ via
``self.stats`` on top of cached data, so a single cache serves all five
strategies (none / z0 / z1 / z2 / z3).
"""

import hashlib
import json
import os
from typing import Literal, Optional

import numpy as np
from scipy.signal import resample_poly
from torch.utils.data import Dataset
from torcheeg import transforms

from eegcrossformer.data.cache_common import serialize_transform, validate_npz_cache
from eegcrossformer.data.dataset.OpenBMI.mi.paper_moabb_transform_cache import (
  normalize_paper_moabb_event_window,
)
from eegcrossformer.data.normalization import ChannelStats


DATASET_VERSION = "v2.0"
OPENBMI_RAW_SFREQ = 1000


def _crop_openbmi_event_window(
  data: np.ndarray,
  event_window_seconds: tuple[float, float] | None,
) -> np.ndarray:
  if event_window_seconds is None:
    return data
  start, stop = normalize_paper_moabb_event_window(event_window_seconds)
  start_sample = round(start * OPENBMI_RAW_SFREQ)
  stop_sample = round(stop * OPENBMI_RAW_SFREQ)
  if stop_sample > data.shape[-1]:
    raise ValueError(
      f"Event window {event_window_seconds} exceeds the OpenBMI trial "
      f"with {data.shape[-1]} samples at {OPENBMI_RAW_SFREQ} Hz"
    )
  return data[..., start_sample:stop_sample]


def _generate_cache_key(
  subject_id: str,
  session: str,
  transform: Optional[transforms.EEGTransform],
  event_window_seconds: tuple[float, float] | None = None,
) -> str:
  """Cache key based on subject, session, transform, and resample factor."""
  config = {
    "subject_id": subject_id,
    "session": session,
    "transform": serialize_transform(transform),
    "resample_factor": 4,
    "dataset_version": DATASET_VERSION,
  }
  if event_window_seconds is not None:
    config["event_window_seconds"] = list(
      normalize_paper_moabb_event_window(event_window_seconds)
    )

  config_str = json.dumps(config, sort_keys=True)
  hash_digest = hashlib.md5(config_str.encode()).hexdigest()[:16]

  return f"s{subject_id}_sess{session}_{hash_digest}"


def _validate_cache_file(cache_file: str) -> bool:
  return validate_npz_cache(cache_file, required_keys=("data", "labels"))


def _load_and_cache_subject_safe(args: tuple) -> dict:
  """Load one subject-session, cache to disk, return raw (un-normalized) data.

  Args:
      args: Tuple of (subject_id, session, root_path, transform, cache_dir,
        force_refresh, event_window_seconds)
  """
  (
    subject_id,
    session,
    root_path,
    transform,
    cache_dir,
    force_refresh,
    event_window_seconds,
  ) = args

  from eegcrossformer.data.dataset.OpenBMI.io import read_openbmi_subject_mat

  cache_key = _generate_cache_key(
    subject_id, session, transform, event_window_seconds
  )
  cache_file = os.path.join(cache_dir, f"{cache_key}.npz")

  if os.path.exists(cache_file) and not force_refresh:
    if _validate_cache_file(cache_file):
      cached = np.load(cache_file)
      return {
        "data": cached["data"],
        "labels": cached["labels"],
        "subject_id": subject_id,
        "session": session,
      }
    else:
      try:
        os.remove(cache_file)
      except Exception:
        pass

  subject_mi_file = os.path.join(
    root_path,
    f"session{session}",
    f"s{subject_id}",
    f"sess{session.zfill(2)}_subj{subject_id.zfill(2)}_EEG_MI.mat",
  )

  data, labels = read_openbmi_subject_mat(subject_mi_file, data_key="EEG_MI_train")

  data = _crop_openbmi_event_window(data, event_window_seconds)
  data = resample_poly(data, up=1, down=4, axis=-1)  # 1000Hz -> 250Hz
  data = data.astype(np.float32)

  if transform is not None:
    transformed = []
    for i in range(data.shape[0]):
      sample = transform(eeg=data[i])["eeg"]
      if isinstance(sample, np.ndarray):
        transformed.append(sample)
      else:
        transformed.append(
          sample.numpy() if hasattr(sample, "numpy") else np.array(sample)
        )
    data = np.stack(transformed, axis=0).astype(np.float32)

  np.savez_compressed(cache_file, data=data, labels=labels)

  return {"data": data, "labels": labels, "subject_id": subject_id, "session": session}


class OpenBMI_MIDataset_FromCache(Dataset):
  """Dataset wrapper for pre-processed cached data.

  Normalization is applied at __getitem__ via `self.stats` (settable after
  construction). Set to None for "no normalization".
  """

  def __init__(
    self,
    data: np.ndarray,
    labels: np.ndarray,
    subject_id: str,
    session: str,
    transform: Optional[transforms.EEGTransform] = None,
    get_item_style: Literal["dict", "tuple"] = "dict",
    extra_info: Optional[dict] = None,
    data_key: str = "data",
    transform_applied: bool = False,
    stats: Optional[ChannelStats] = None,
  ):
    super().__init__()
    self.data = data
    self.labels = labels
    self.subject_id = subject_id
    self.session = session
    self.transform = transform
    self.get_item_style = get_item_style
    self.extra_info = extra_info
    self.data_key = data_key
    self.transform_applied = transform_applied
    self.stats = stats

  def __len__(self):
    return self.data.shape[0]

  def __getitem__(self, index):
    eeg_data = self.data[index].astype(np.float32)

    if self.stats is not None:
      eeg_data = self.stats.apply(eeg_data).astype(np.float32)

    if self.transform is not None and not self.transform_applied:
      eeg_data = self.transform(eeg=eeg_data)["eeg"]

    if self.get_item_style == "dict":
      return_dict = {
        self.data_key: eeg_data,
        "target": self.labels[index],
      }
      if self.extra_info is not None:
        return_dict["extra_info"] = self.extra_info
      return return_dict
    else:
      return (
        eeg_data,
        self.labels[index],
        {} if self.extra_info is None else self.extra_info,
      )
