"""Content-addressed offline transform cache for paper-MOABB OpenBMI trials."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from torcheeg import transforms

from eegcrossformer.data.cache_common import serialize_transform
from eegcrossformer.data.transforms.fb_transform import (
  MultiViewStackingTransform,
  PaddingTransform,
  TrailFilteringBankTransform,
)
from eegcrossformer.project_config import openbmi_path, openbmi_subjects


TRANSFORM_CACHE_SCHEMA_VERSION = "v2"
PAPER_MOABB_SOURCE_PROTOCOL_VERSION = "paper_moabb_raw_continuous_v1"
TRANSFORM_CACHE_DIRNAME = "OpenBMI_MI_PaperMOABB_Transform_Cache"
DEFAULT_EVENT_WINDOW_SECONDS = (0.0, 3.0)


def normalize_paper_moabb_event_window(
  event_window_seconds: tuple[float, float],
) -> tuple[float, float]:
  start, stop = (float(value) for value in event_window_seconds)
  if not all(math.isfinite(value) for value in (start, stop)):
    raise ValueError("event_window_seconds must contain finite values")
  if start < 0 or stop > 4 or stop <= start:
    raise ValueError(
      "event_window_seconds must be a nonempty window within the cue-aligned [0, 4] s epoch"
    )
  if not all(math.isclose(value * 250, round(value * 250), abs_tol=1e-8) for value in (start, stop)):
    raise ValueError("event_window_seconds boundaries must align with the 250 Hz sample grid")
  return start, stop

FBCNET_BANDS = (
  (4, 8),
  (8, 12),
  (12, 16),
  (16, 20),
  (20, 24),
  (24, 28),
  (28, 32),
  (32, 36),
  (36, 40),
)
IFNET_BANDS = ((1, 16), (16, 40))
CROSSFORMER_V7_BANDS = ((1, 16), (16, 40))
CROSSFORMER_AC7_BANDS_0_20_16_40 = ((0, 20), (16, 40))
CROSSFORMER_AC7_BANDS_0_24_16_40 = ((0, 24), (16, 40))
DGCNN_BANDS = {
  "delta": [1, 4],
  "theta": [4, 8],
  "alpha": [8, 14],
  "beta": [14, 31],
  "gamma": [31, 40],
}


@dataclass(frozen=True)
class TransformProfile:
  name: str
  factory: Callable[[], transforms.EEGTransform]
  shape_factory: Callable[[int], tuple[int, ...]]
  squeeze_single_view: bool = False
  dtype: str = "float32"
  temporal_axis: int | None = -1

  def expected_shape(
    self,
    channels: Literal["motor20", "all"],
    event_window_seconds: tuple[float, float] = DEFAULT_EVENT_WINDOW_SECONDS,
  ) -> tuple[int, ...]:
    start, stop = normalize_paper_moabb_event_window(event_window_seconds)
    shape = list(self.shape_factory(62 if channels == "all" else 20))
    if self.temporal_axis is not None:
      # Factories describe the historical 750-point window, including padding.
      shape[self.temporal_axis] += round((stop - start) * 250) - 750
    return tuple(shape)


def _fbcnet9_transform() -> transforms.Compose:
  return transforms.Compose(
    [
      TrailFilteringBankTransform(freq_bands=FBCNET_BANDS, sfreq=250),
      PaddingTransform(padding=2),
    ]
  )


def _ifnet2_transform() -> transforms.Compose:
  return transforms.Compose(
    [
      TrailFilteringBankTransform(freq_bands=IFNET_BANDS, sfreq=250),
      MultiViewStackingTransform(freq_bands=IFNET_BANDS),
    ]
  )


def _fbcnet9_pad3_transform() -> transforms.Compose:
  """Right-pad a 625-point late window to a multiple of four (628)."""
  return transforms.Compose(
    [
      TrailFilteringBankTransform(freq_bands=FBCNET_BANDS, sfreq=250),
      PaddingTransform(padding=3),
    ]
  )


def _crossformer_v7_transform() -> transforms.Compose:
  return transforms.Compose(
    [
      TrailFilteringBankTransform(
        freq_bands=CROSSFORMER_V7_BANDS,
        sfreq=250,
      )
    ]
  )


def _crossformer_ac7_0_20_16_40_transform() -> transforms.Compose:
  return transforms.Compose(
    [
      TrailFilteringBankTransform(
        freq_bands=CROSSFORMER_AC7_BANDS_0_20_16_40,
        sfreq=250,
      )
    ]
  )


def _crossformer_ac7_0_24_16_40_transform() -> transforms.Compose:
  return transforms.Compose(
    [
      TrailFilteringBankTransform(
        freq_bands=CROSSFORMER_AC7_BANDS_0_24_16_40,
        sfreq=250,
      )
    ]
  )


def _dgcnn_de_transform() -> transforms.BandDifferentialEntropy:
  return transforms.BandDifferentialEntropy(
    sampling_rate=250,
    band_dict=DGCNN_BANDS,
  )


TRANSFORM_PROFILES: dict[str, TransformProfile] = {
  "fbcnet9_pad3": TransformProfile(
    name="fbcnet9_pad3",
    factory=_fbcnet9_pad3_transform,
    shape_factory=lambda channels: (9, channels, 753),
  ),
  "fbcnet9": TransformProfile(
    name="fbcnet9",
    factory=_fbcnet9_transform,
    shape_factory=lambda channels: (9, channels, 752),
  ),
  "ifnet2": TransformProfile(
    name="ifnet2",
    factory=_ifnet2_transform,
    shape_factory=lambda channels: (2 * channels, 750),
  ),
  "crossformer_v7_2band": TransformProfile(
    name="crossformer_v7_2band",
    factory=_crossformer_v7_transform,
    shape_factory=lambda channels: (2, channels, 750),
  ),
  "crossformer_ac7_2band_0_20_16_40": TransformProfile(
    name="crossformer_ac7_2band_0_20_16_40",
    factory=_crossformer_ac7_0_20_16_40_transform,
    shape_factory=lambda channels: (2, channels, 750),
  ),
  "crossformer_ac7_2band_0_24_16_40": TransformProfile(
    name="crossformer_ac7_2band_0_24_16_40",
    factory=_crossformer_ac7_0_24_16_40_transform,
    shape_factory=lambda channels: (2, channels, 750),
  ),
  "dgcnn_de": TransformProfile(
    name="dgcnn_de",
    factory=_dgcnn_de_transform,
    shape_factory=lambda channels: (channels, 5),
    squeeze_single_view=True,
    temporal_axis=None,
  ),
}


@dataclass(frozen=True)
class TransformCacheEntry:
  data: np.memmap
  labels: np.memmap
  directory: Path
  manifest: dict
  status: Literal["hit", "build"]


def get_transform_profile(name: str) -> TransformProfile:
  try:
    return TRANSFORM_PROFILES[name]
  except KeyError as exc:
    raise ValueError(
      f"Unknown offline transform profile {name!r}; choose from {sorted(TRANSFORM_PROFILES)}"
    ) from exc


def transform_cache_key(
  profile_name: str,
  channels: Literal["motor20", "all"] = "all",
  *,
  event_window_seconds: tuple[float, float] = DEFAULT_EVENT_WINDOW_SECONDS,
  schema_version: str = TRANSFORM_CACHE_SCHEMA_VERSION,
  source_protocol_version: str = PAPER_MOABB_SOURCE_PROTOCOL_VERSION,
) -> tuple[str, dict]:
  profile = get_transform_profile(profile_name)
  event_window_seconds = normalize_paper_moabb_event_window(event_window_seconds)
  payload = {
    "schema_version": schema_version,
    "source_protocol_version": source_protocol_version,
    "channels": channels,
    "dtype": profile.dtype,
    "expected_shape": list(profile.expected_shape(channels, event_window_seconds)),
    "transform": serialize_transform(profile.factory()),
  }
  if event_window_seconds != DEFAULT_EVENT_WINDOW_SECONDS:
    payload["event_window_seconds"] = list(event_window_seconds)
  canonical = json.dumps(
    payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
  ).encode("utf-8")
  return hashlib.sha256(canonical).hexdigest(), payload


def transform_cache_root(root_path: str | os.PathLike = openbmi_path) -> Path:
  return (
    Path(root_path)
    / ".eegcrossformer"
    / TRANSFORM_CACHE_DIRNAME
    / TRANSFORM_CACHE_SCHEMA_VERSION
  )


def transform_cache_directory(
  profile_name: str,
  subject_id: str,
  session_index: int,
  channels: Literal["motor20", "all"] = "all",
  *,
  event_window_seconds: tuple[float, float] = DEFAULT_EVENT_WINDOW_SECONDS,
  root_path: str | os.PathLike = openbmi_path,
) -> Path:
  digest, _ = transform_cache_key(
    profile_name,
    channels,
    event_window_seconds=event_window_seconds,
  )
  return (
    transform_cache_root(root_path)
    / digest
    / f"s{str(subject_id).zfill(2)}_session{int(session_index)}_{channels}"
  )


def transform_cache_settings(
  profile_name: str | None,
  channels: Literal["motor20", "all"] = "all",
  *,
  event_window_seconds: tuple[float, float] = DEFAULT_EVENT_WINDOW_SECONDS,
  root_path: str | os.PathLike = openbmi_path,
) -> dict:
  if profile_name is None:
    return {
      "offline_transform_profile": None,
      "offline_transform": None,
      "transform_hash": None,
      "transform_cache_schema_version": TRANSFORM_CACHE_SCHEMA_VERSION,
      "transform_cache_directory": None,
    }
  digest, payload = transform_cache_key(
    profile_name,
    channels,
    event_window_seconds=event_window_seconds,
  )
  return {
    "offline_transform_profile": profile_name,
    "offline_transform": payload["transform"],
    "transform_hash": digest,
    "transform_cache_schema_version": TRANSFORM_CACHE_SCHEMA_VERSION,
    "transform_cache_directory": str(transform_cache_root(root_path) / digest),
  }


def _npy_file_size(path: Path) -> int:
  return path.stat().st_size


def _load_valid_entry(
  directory: Path,
  *,
  digest: str,
  payload: dict,
  subject_id: str,
  session_index: int,
  channels: Literal["motor20", "all"],
) -> tuple[np.memmap, np.memmap, dict] | None:
  try:
    manifest_path = directory / "manifest.json"
    data_path = directory / "data.npy"
    labels_path = directory / "labels.npy"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
      "schema_version": TRANSFORM_CACHE_SCHEMA_VERSION,
      "transform_hash": digest,
      "source_protocol_version": payload["source_protocol_version"],
      "subject_id": str(subject_id),
      "session_index": int(session_index),
      "channels": channels,
      "dtype": "float32",
      "label_dtype": "int64",
      "sample_shape": payload["expected_shape"],
    }
    if "event_window_seconds" in payload:
      expected["event_window_seconds"] = payload["event_window_seconds"]
    if any(manifest.get(key) != value for key, value in expected.items()):
      return None
    if manifest.get("transform") != payload["transform"]:
      return None
    if manifest.get("data_file_size") != _npy_file_size(data_path):
      return None
    if manifest.get("labels_file_size") != _npy_file_size(labels_path):
      return None

    data = np.load(data_path, mmap_mode="r", allow_pickle=False)
    labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
    trial_count = manifest.get("trial_count")
    if data.dtype != np.float32 or labels.dtype != np.int64:
      return None
    if data.shape != (trial_count, *tuple(payload["expected_shape"])):
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


def _apply_transform(profile: TransformProfile, transform, trial: np.ndarray) -> np.ndarray:
  eeg = trial
  if profile.squeeze_single_view and eeg.ndim == 3 and eeg.shape[0] == 1:
    eeg = eeg[0]
  output = transform(eeg=eeg)["eeg"]
  return np.asarray(output, dtype=np.float32)


def _default_source_loader(
  subject_id: str,
  session_index: int,
  channels: Literal["motor20", "all"],
  event_window_seconds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
  # Lazy import avoids a cycle with the dataset classes that consume this cache.
  from eegcrossformer.data.dataset.OpenBMI.mi.OpenBMIMIDataset import (
    _load_openbmi_paper_moabb_subject_session,
  )

  return _load_openbmi_paper_moabb_subject_session(
    str(subject_id), int(session_index), channels, event_window_seconds
  )


def load_or_build_transform_cache(
  profile_name: str,
  subject_id: str,
  session_index: int,
  channels: Literal["motor20", "all"] = "all",
  *,
  event_window_seconds: tuple[float, float] = DEFAULT_EVENT_WINDOW_SECONDS,
  root_path: str | os.PathLike = openbmi_path,
  source_loader: Callable[[str, int, Literal["motor20", "all"]], tuple[np.ndarray, np.ndarray]]
  | None = None,
) -> TransformCacheEntry:
  """Open a valid read-only mmap cache, lazily building it when absent/invalid."""
  profile = get_transform_profile(profile_name)
  event_window_seconds = normalize_paper_moabb_event_window(event_window_seconds)
  digest, payload = transform_cache_key(
    profile_name,
    channels,
    event_window_seconds=event_window_seconds,
  )
  directory = transform_cache_directory(
    profile_name,
    subject_id,
    session_index,
    channels,
    event_window_seconds=event_window_seconds,
    root_path=root_path,
  )
  valid = _load_valid_entry(
    directory,
    digest=digest,
    payload=payload,
    subject_id=subject_id,
    session_index=session_index,
    channels=channels,
  )
  if valid is not None:
    data, labels, manifest = valid
    return TransformCacheEntry(data, labels, directory, manifest, "hit")

  directory.parent.mkdir(parents=True, exist_ok=True)
  lock_path = directory.parent / f"{directory.name}.lock"
  lock_fd = _acquire_lock(lock_path)
  try:
    with os.fdopen(lock_fd, "w", encoding="utf-8") as lock_handle:
      lock_handle.write(f"{os.getpid()}\n")

    valid = _load_valid_entry(
      directory,
      digest=digest,
      payload=payload,
      subject_id=subject_id,
      session_index=session_index,
      channels=channels,
    )
    if valid is not None:
      data, labels, manifest = valid
      return TransformCacheEntry(data, labels, directory, manifest, "hit")

    for stale_tmp in directory.parent.glob(f"{directory.name}.tmp-*"):
      shutil.rmtree(stale_tmp, ignore_errors=True)
    if directory.exists():
      shutil.rmtree(directory)

    if source_loader is None:
      raw_data, raw_labels = _default_source_loader(
        str(subject_id), int(session_index), channels, event_window_seconds
      )
    else:
      raw_data, raw_labels = source_loader(
        str(subject_id), int(session_index), channels
      )
    raw_data = np.asarray(raw_data)
    raw_labels = np.asarray(raw_labels, dtype=np.int64)
    if raw_data.ndim != 4 or raw_data.shape[0] != raw_labels.shape[0]:
      raise ValueError(
        f"Invalid raw paper-MOABB arrays: data={raw_data.shape}, labels={raw_labels.shape}"
      )

    tmp_dir = Path(
      tempfile.mkdtemp(prefix=f"{directory.name}.tmp-", dir=directory.parent)
    )
    try:
      data_path = tmp_dir / "data.npy"
      labels_path = tmp_dir / "labels.npy"
      expected_shape = tuple(payload["expected_shape"])
      data_mmap = np.lib.format.open_memmap(
        data_path,
        mode="w+",
        dtype=np.float32,
        shape=(raw_data.shape[0], *expected_shape),
      )
      transform = profile.factory()
      for trial_index in range(raw_data.shape[0]):
        transformed = _apply_transform(profile, transform, raw_data[trial_index])
        if transformed.shape != expected_shape:
          raise ValueError(
            f"Profile {profile_name} produced {transformed.shape}; expected {expected_shape}"
          )
        data_mmap[trial_index] = transformed
      data_mmap.flush()
      del data_mmap
      np.save(labels_path, raw_labels.astype(np.int64, copy=False), allow_pickle=False)

      # Reopen before publication so truncated or malformed files never become hits.
      checked_data = np.load(data_path, mmap_mode="r", allow_pickle=False)
      checked_labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
      if checked_data.shape != (raw_data.shape[0], *expected_shape):
        raise ValueError("Transform cache data shape changed while building")
      if checked_data.dtype != np.float32 or checked_labels.dtype != np.int64:
        raise ValueError("Transform cache dtype validation failed")
      if checked_labels.shape != (raw_data.shape[0],):
        raise ValueError("Transform cache label count validation failed")
      del checked_data, checked_labels

      manifest = {
        "schema_version": TRANSFORM_CACHE_SCHEMA_VERSION,
        "source_protocol_version": payload["source_protocol_version"],
        "profile": profile_name,
        "transform_hash": digest,
        "transform": payload["transform"],
        "subject_id": str(subject_id),
        "session_index": int(session_index),
        "channels": channels,
        "trial_count": int(raw_data.shape[0]),
        "sample_shape": list(expected_shape),
        "dtype": "float32",
        "label_dtype": "int64",
        "data_file_size": _npy_file_size(data_path),
        "labels_file_size": _npy_file_size(labels_path),
      }
      if "event_window_seconds" in payload:
        manifest["event_window_seconds"] = payload["event_window_seconds"]
      (tmp_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
      )
      os.replace(tmp_dir, directory)
    except BaseException:
      shutil.rmtree(tmp_dir, ignore_errors=True)
      raise

    valid = _load_valid_entry(
      directory,
      digest=digest,
      payload=payload,
      subject_id=subject_id,
      session_index=session_index,
      channels=channels,
    )
    if valid is None:
      raise RuntimeError(f"Published transform cache failed validation: {directory}")
    data, labels, manifest = valid
    return TransformCacheEntry(data, labels, directory, manifest, "build")
  finally:
    lock_path.unlink(missing_ok=True)


def _prewarm_one(task) -> tuple[str, str, str]:
  profile, subject, session, channels, window = task
  try:
    entry = load_or_build_transform_cache(
      profile, subject, session, channels, event_window_seconds=window
    )
    return entry.status, str(entry.directory), ""
  except Exception as exc:
    return "fail", transform_cache_directory(
      profile, subject, session, channels, event_window_seconds=window
    ).as_posix(), f"{type(exc).__name__}: {exc}"


def _parse_csv_or_all(value: str, choices: list[str]) -> list[str]:
  if value.strip().lower() == "all":
    return choices
  values = [item.strip() for item in value.split(",") if item.strip()]
  invalid = sorted(set(values) - set(choices))
  if invalid:
    raise ValueError(f"Unsupported values {invalid}; choose from {choices} or all")
  return values


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--profiles", default="all")
  parser.add_argument("--subjects", default="all")
  parser.add_argument("--sessions", default="both")
  parser.add_argument("--channels", default="all")
  parser.add_argument("--workers", type=int, default=3)
  parser.add_argument("--event_window_seconds", nargs=2, type=float, default=DEFAULT_EVENT_WINDOW_SECONDS)
  args = parser.parse_args(argv)
  window = normalize_paper_moabb_event_window(tuple(args.event_window_seconds))

  profiles = _parse_csv_or_all(args.profiles, list(TRANSFORM_PROFILES))
  subjects = _parse_csv_or_all(args.subjects, list(openbmi_subjects))
  sessions = [1, 2] if args.sessions.lower() in {"both", "all"} else [
    int(item) for item in args.sessions.split(",")
  ]
  if any(session not in (1, 2) for session in sessions):
    parser.error("--sessions must be 1, 2, both, or 1,2")
  channels = _parse_csv_or_all(args.channels, ["all", "motor20"])
  # `--channels all` denotes the all-channel protocol, not both channel modes.
  if args.channels.strip().lower() == "all":
    channels = ["all"]

  tasks = [
    (profile, subject, session, channel_mode, window)
    for profile in profiles
    for subject in subjects
    for session in sessions
    for channel_mode in channels
  ]
  counts = {"hit": 0, "build": 0, "fail": 0}
  with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
    futures = [executor.submit(_prewarm_one, task) for task in tasks]
    for future in as_completed(futures):
      status, path, error = future.result()
      counts[status] += 1
      suffix = f" ({error})" if error else ""
      print(f"{status.upper():5s} {path}{suffix}", flush=True)
  print(
    f"Transform cache summary: hit={counts['hit']} build={counts['build']} "
    f"fail={counts['fail']} root={transform_cache_root()}"
  )
  return 1 if counts["fail"] else 0


if __name__ == "__main__":
  raise SystemExit(main())
