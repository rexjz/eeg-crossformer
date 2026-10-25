"""Content-addressed cache for BCI Competition III Dataset IVa trials.

The cache stores deterministic per-trial outputs keyed by subject and offline
transform. Random batch augmentation and train/test splitting stay outside the
cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from torch.utils.data import Dataset
from torcheeg import transforms

from eegcrossformer.data.cache_common import serialize_transform
from eegcrossformer.project_config import bcic3_dataset4a_path


BCIC34A_CACHE_SCHEMA_VERSION = "v1"
BCIC34A_SOURCE_PROTOCOL_VERSION = "raw_car_trials_118ch_350_v1"
BCIC34A_CACHE_DIRNAME = "BCIC34a_Subject_Cache"


@dataclass(frozen=True)
class BCIC34aCacheEntry:
  data: np.memmap
  labels: np.memmap
  directory: Path
  manifest: dict
  status: Literal["hit", "build"]


def _source_metadata(root_path: str | os.PathLike, subject_code: str) -> dict:
  root = Path(root_path)
  files = {
    "data": root / f"data_set_IVa_{subject_code}.mat",
    "labels": root / f"true_labels_{subject_code}.mat",
  }
  return {
    key: {
      "name": path.name,
      "size": path.stat().st_size,
      "mtime_ns": path.stat().st_mtime_ns,
    }
    for key, path in files.items()
  }


def bcic34a_cache_key(
  subject_code: str,
  transform: Optional[transforms.EEGTransform],
  *,
  root_path: str | os.PathLike = bcic3_dataset4a_path,
) -> tuple[str, dict]:
  payload = {
    "schema_version": BCIC34A_CACHE_SCHEMA_VERSION,
    "source_protocol_version": BCIC34A_SOURCE_PROTOCOL_VERSION,
    "subject_code": subject_code,
    "source_files": _source_metadata(root_path, subject_code),
    "transform": serialize_transform(transform),
  }
  canonical = json.dumps(
    payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
  ).encode("utf-8")
  return hashlib.sha256(canonical).hexdigest(), payload


def bcic34a_cache_root(
  root_path: str | os.PathLike = bcic3_dataset4a_path,
) -> Path:
  return (
    Path(root_path)
    / ".eegcrossformer"
    / BCIC34A_CACHE_DIRNAME
    / BCIC34A_CACHE_SCHEMA_VERSION
  )


def bcic34a_cache_directory(
  subject_code: str,
  transform: Optional[transforms.EEGTransform],
  *,
  root_path: str | os.PathLike = bcic3_dataset4a_path,
) -> Path:
  digest, _ = bcic34a_cache_key(subject_code, transform, root_path=root_path)
  return bcic34a_cache_root(root_path) / digest / f"subject_{subject_code}"


def bcic34a_cache_settings(
  transform: Optional[transforms.EEGTransform],
  *,
  root_path: str | os.PathLike = bcic3_dataset4a_path,
) -> dict:
  transform_payload = serialize_transform(transform)
  digest = hashlib.sha256(
    json.dumps(transform_payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
  ).hexdigest()
  return {
    "cache_enabled": True,
    "cache_schema_version": BCIC34A_CACHE_SCHEMA_VERSION,
    "source_protocol_version": BCIC34A_SOURCE_PROTOCOL_VERSION,
    "offline_transform": transform_payload,
    "offline_transform_hash_template": digest,
    "cache_root": str(bcic34a_cache_root(root_path)),
  }


def _npy_file_size(path: Path) -> int:
  return path.stat().st_size


def _load_valid_entry(
  directory: Path, *, digest: str, payload: dict
) -> tuple[np.memmap, np.memmap, dict] | None:
  try:
    manifest_path = directory / "manifest.json"
    data_path = directory / "data.npy"
    labels_path = directory / "labels.npy"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
      "schema_version": BCIC34A_CACHE_SCHEMA_VERSION,
      "source_protocol_version": BCIC34A_SOURCE_PROTOCOL_VERSION,
      "transform_hash": digest,
      "subject_code": payload["subject_code"],
      "source_files": payload["source_files"],
      "transform": payload["transform"],
      "dtype": "float32",
      "label_dtype": "int64",
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
      return None
    if manifest.get("data_file_size") != _npy_file_size(data_path):
      return None
    if manifest.get("labels_file_size") != _npy_file_size(labels_path):
      return None

    data = np.load(data_path, mmap_mode="r", allow_pickle=False)
    labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
    trial_count = manifest.get("trial_count")
    sample_shape = tuple(manifest.get("sample_shape", ()))
    if data.dtype != np.float32 or labels.dtype != np.int64:
      return None
    if data.shape != (trial_count, *sample_shape):
      return None
    if labels.shape != (trial_count,):
      return None
    return data, labels, manifest
  except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
    return None


def _acquire_lock(lock_path: Path, stale_after_seconds: float = 6 * 60 * 60) -> int:
  while True:
    try:
      return os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
      try:
        if time.time() - lock_path.stat().st_mtime > stale_after_seconds:
          lock_path.unlink(missing_ok=True)
          continue
      except OSError:
        pass
      time.sleep(0.1)


def _build_arrays(root_path: str | os.PathLike, subject_code: str, transform):
  from eegcrossformer.data.dataset.bci_competition.BCIC3Dataset4a import (
    BcicompetitionIIIDatasetIVaRawTrailDataset,
  )

  raw_dataset = BcicompetitionIIIDatasetIVaRawTrailDataset(
    root_path=str(root_path),
    subject_code=subject_code,
    transform=None,
    get_item_style="dict",
  )
  data = []
  labels = []
  for index in range(len(raw_dataset)):
    sample = raw_dataset[index]
    eeg = np.asarray(sample["data"], dtype=np.float32)
    if transform is not None:
      eeg = np.asarray(transform(eeg=eeg)["eeg"], dtype=np.float32)
    data.append(eeg)
    labels.append(int(sample["target"]))
  return np.stack(data, axis=0).astype(np.float32), np.asarray(labels, dtype=np.int64)


def load_or_build_bcic34a_subject_cache(
  subject_code: str,
  transform: Optional[transforms.EEGTransform],
  *,
  root_path: str | os.PathLike = bcic3_dataset4a_path,
  force_refresh: bool = False,
) -> BCIC34aCacheEntry:
  digest, payload = bcic34a_cache_key(subject_code, transform, root_path=root_path)
  directory = bcic34a_cache_directory(subject_code, transform, root_path=root_path)

  if not force_refresh:
    valid = _load_valid_entry(directory, digest=digest, payload=payload)
    if valid is not None:
      data, labels, manifest = valid
      return BCIC34aCacheEntry(data, labels, directory, manifest, "hit")

  directory.parent.mkdir(parents=True, exist_ok=True)
  lock_path = directory.parent / f"{directory.name}.lock"
  lock_fd = _acquire_lock(lock_path)
  try:
    with os.fdopen(lock_fd, "w", encoding="utf-8") as lock_handle:
      lock_handle.write(f"{os.getpid()}\n")

    if not force_refresh:
      valid = _load_valid_entry(directory, digest=digest, payload=payload)
      if valid is not None:
        data, labels, manifest = valid
        return BCIC34aCacheEntry(data, labels, directory, manifest, "hit")

    for stale_tmp in directory.parent.glob(f"{directory.name}.tmp-*"):
      shutil.rmtree(stale_tmp, ignore_errors=True)
    if directory.exists():
      shutil.rmtree(directory)

    data, labels = _build_arrays(root_path, subject_code, transform)
    tmp_dir = Path(
      tempfile.mkdtemp(prefix=f"{directory.name}.tmp-", dir=directory.parent)
    )
    try:
      data_path = tmp_dir / "data.npy"
      labels_path = tmp_dir / "labels.npy"
      np.save(data_path, data, allow_pickle=False)
      np.save(labels_path, labels, allow_pickle=False)
      manifest = {
        "schema_version": BCIC34A_CACHE_SCHEMA_VERSION,
        "source_protocol_version": BCIC34A_SOURCE_PROTOCOL_VERSION,
        "transform_hash": digest,
        "subject_code": subject_code,
        "source_files": payload["source_files"],
        "transform": payload["transform"],
        "trial_count": int(data.shape[0]),
        "sample_shape": list(data.shape[1:]),
        "dtype": "float32",
        "label_dtype": "int64",
        "data_file_size": _npy_file_size(data_path),
        "labels_file_size": _npy_file_size(labels_path),
      }
      (tmp_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
      )
      os.replace(tmp_dir, directory)
    except BaseException:
      shutil.rmtree(tmp_dir, ignore_errors=True)
      raise

    valid = _load_valid_entry(directory, digest=digest, payload=payload)
    if valid is None:
      raise RuntimeError(f"Published BCI34a cache failed validation: {directory}")
    data_mmap, labels_mmap, manifest = valid
    return BCIC34aCacheEntry(data_mmap, labels_mmap, directory, manifest, "build")
  finally:
    lock_path.unlink(missing_ok=True)


class BCIC34aSubjectCachedDataset(Dataset):
  def __init__(
    self,
    data: np.ndarray,
    labels: np.ndarray,
    *,
    get_item_style: Literal["dict", "tuple"] = "dict",
    extra_info: Optional[dict] = None,
  ):
    self.data = data
    self.labels = labels
    self.get_item_style = get_item_style
    self.extra_info = extra_info

  def __len__(self):
    return int(self.labels.shape[0])

  def __getitem__(self, index):
    eeg_data = np.array(self.data[index], dtype=np.float32, copy=True)
    label = int(self.labels[index])
    if self.get_item_style == "dict":
      item = {"data": eeg_data, "target": label}
      if self.extra_info is not None:
        item["extra_info"] = self.extra_info
      return item
    return eeg_data, label, {} if self.extra_info is None else self.extra_info
