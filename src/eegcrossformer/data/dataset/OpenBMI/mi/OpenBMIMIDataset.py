from scipy.signal import resample_poly
from scipy.signal import resample
from functools import lru_cache
import hashlib
import json
import mne
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import random_split
from eegcrossformer.data.dataset.OpenBMI.io import read_openbmi_subject_mat
from eegcrossformer.data.dataset.OpenBMI.mi.cache_utils import (
  _crop_openbmi_event_window,
)
from eegcrossformer.data.dataset.kfold_split import kfold_split, stratified_kfold_split
from eegcrossformer.data.normalization import ChannelStats
from eegcrossformer.project_config import openbmi_path
from eegcrossformer.data.dataset.OpenBMI.mi.paper_moabb_transform_cache import (
  DEFAULT_EVENT_WINDOW_SECONDS,
  PAPER_MOABB_SOURCE_PROTOCOL_VERSION,
  load_or_build_transform_cache,
  normalize_paper_moabb_event_window,
)
from torch.utils.data import Dataset
from torcheeg import transforms
from typing import Literal, Optional, Sequence
import lightning as L
import numpy as np
import torch
import os
import tempfile
import time


def _exponential_moving_standardize(
  data: np.ndarray,
  factor_new: float = 1e-3,
  init_block_size: int = 1000,
  eps: float = 1e-4,
) -> np.ndarray:
  """Trial-wise approximation of braindecode's exponential standardization."""
  standardized = np.empty_like(data, dtype=np.float32)

  for trial_idx in range(data.shape[0]):
    trial = data[trial_idx].astype(np.float64, copy=False)
    mean = np.zeros_like(trial)
    var = np.zeros_like(trial)

    if init_block_size is not None and init_block_size > 0:
      init_end = min(init_block_size, trial.shape[-1])
      init_mean = np.mean(trial[:, :init_end], axis=-1, keepdims=True)
      init_var = np.var(trial[:, :init_end], axis=-1, keepdims=True)
      mean[:, :init_end] = init_mean
      var[:, :init_end] = init_var
      start = init_end
      previous_mean = init_mean[:, 0]
      previous_var = init_var[:, 0]
    else:
      start = 0
      previous_mean = trial[:, 0]
      previous_var = np.zeros(trial.shape[0], dtype=np.float64)

    for t in range(start, trial.shape[-1]):
      x_t = trial[:, t]
      current_mean = factor_new * x_t + (1.0 - factor_new) * previous_mean
      current_var = factor_new * (x_t - current_mean) ** 2 + (
        1.0 - factor_new
      ) * previous_var
      mean[:, t] = current_mean
      var[:, t] = current_var
      previous_mean = current_mean
      previous_var = current_var

    standardized[trial_idx] = ((trial - mean) / np.maximum(np.sqrt(var), eps)).astype(
      np.float32
    )

  return standardized


@lru_cache(maxsize=4)
def _load_openbmi_paper_session1_subject(subject_id: str, root_path: str):
  subject_mi_file = os.path.join(
    root_path,
    "session1",
    f"s{subject_id}",
    f"sess01_subj{subject_id.zfill(2)}_EEG_MI.mat",
  )
  data, labels = read_openbmi_subject_mat(subject_mi_file, data_key="EEG_MI_train")

  # Upstream selects channels 20:40 after preprocessing; filtering and
  # standardization are channel-wise, so selecting first is equivalent here.
  data = data[:, 20:40, :].astype(np.float64, copy=False)
  data = mne.filter.filter_data(
    data,
    sfreq=1000,
    l_freq=None,
    h_freq=40,
    verbose=False,
  )
  data = _exponential_moving_standardize(
    data,
    factor_new=1e-3,
    init_block_size=1000,
  )
  data = data[..., :3000]
  data = resample(data, 750, axis=-1)
  data = np.expand_dims(data.astype(np.float32), axis=1)
  return data, labels.astype(np.int64)


def _get_braindecode_moabb_api():
  try:
    from braindecode.datasets import MOABBDataset
    from braindecode.preprocessing import (
      Preprocessor,
      preprocess,
      exponential_moving_standardize,
      create_windows_from_events,
    )

    return (
      MOABBDataset,
      Preprocessor,
      preprocess,
      exponential_moving_standardize,
      create_windows_from_events,
    )
  except ImportError:
    try:
      from braindecode.datasets.moabb import MOABBDataset
      from braindecode.datautil.preprocess import (
        Preprocessor,
        preprocess,
        exponential_moving_standardize,
      )
      from braindecode.datautil.windowers import create_windows_from_events

      return (
        MOABBDataset,
        Preprocessor,
        preprocess,
        exponential_moving_standardize,
        create_windows_from_events,
      )
    except ImportError as exc:
      raise ImportError(
        "OpenBMI_MIPaperMOABBSession1Dataset requires braindecode/moabb. "
        "Install the project dependencies before running the MOABB raw-continuous "
        "paper replication entry."
      ) from exc


def _select_moabb_session_split(session_split, session_index: int):
  if session_index == 1:
    candidate_keys = ["session_1", "0"]
  elif session_index == 2:
    candidate_keys = ["session_2", "1"]
  else:
    raise ValueError(f"session_index must be 1 or 2, got {session_index}")

  for key in candidate_keys:
    if key in session_split:
      return session_split[key]

  raise KeyError(
    f"Could not find session {session_index} split in MOABB windows. "
    f"Available session keys: {list(session_split.keys())}"
  )


PAPER_MOABB_CACHE_VERSION = PAPER_MOABB_SOURCE_PROTOCOL_VERSION


def _paper_moabb_cache_file(
  subject_id: str,
  session_index: int,
  channels: Literal["motor20", "all"],
  event_window_seconds: tuple[float, float] = DEFAULT_EVENT_WINDOW_SECONDS,
) -> str:
  event_window_seconds = normalize_paper_moabb_event_window(event_window_seconds)
  cache_dir = os.path.join(
    str(openbmi_path), ".eegcrossformer", "OpenBMI_MI_PaperMOABB_Cache"
  )
  os.makedirs(cache_dir, exist_ok=True)
  config = {
    "version": PAPER_MOABB_CACHE_VERSION,
    "dataset": "Lee2019_MI",
    "subject_id": str(subject_id),
    "session_index": int(session_index),
    "channels": channels,
    "preprocessing": "pick_eeg_uv_0_40hz_continuous_expstd_windows_0_3s_resample_250hz",
  }
  if event_window_seconds != DEFAULT_EVENT_WINDOW_SECONDS:
    config["event_window_seconds"] = list(event_window_seconds)
  digest = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
  return os.path.join(
    cache_dir, f"s{str(subject_id).zfill(2)}_session{session_index}_{channels}_{digest}.npz"
  )


def _load_paper_moabb_cache(cache_file: str) -> tuple[np.ndarray, np.ndarray] | None:
  if not os.path.exists(cache_file):
    return None
  try:
    cached = np.load(cache_file)
    if "data" not in cached or "labels" not in cached:
      return None
    return cached["data"].astype(np.float32), cached["labels"].astype(np.int64)
  except Exception:
    try:
      os.remove(cache_file)
    except OSError:
      pass
    return None


def _save_paper_moabb_cache(
  cache_file: str,
  data: np.ndarray,
  labels: np.ndarray,
) -> None:
  cache_dir = os.path.dirname(cache_file)
  with tempfile.NamedTemporaryFile(
    dir=cache_dir,
    prefix=".tmp_paper_moabb_",
    suffix=".npz",
    delete=False,
  ) as tmp_file:
    tmp_path = tmp_file.name
  try:
    np.savez_compressed(tmp_path, data=data.astype(np.float32), labels=labels.astype(np.int64))
    os.replace(tmp_path, cache_file)
  finally:
    if os.path.exists(tmp_path):
      try:
        os.remove(tmp_path)
      except OSError:
        pass


def _load_or_build_paper_moabb_cache(
  subject_id: str,
  session_index: int,
  channels: Literal["motor20", "all"],
  build_fn,
  event_window_seconds: tuple[float, float] = DEFAULT_EVENT_WINDOW_SECONDS,
) -> tuple[np.ndarray, np.ndarray]:
  cache_file = _paper_moabb_cache_file(
    subject_id, session_index, channels, event_window_seconds
  )
  cached = _load_paper_moabb_cache(cache_file)
  if cached is not None:
    return cached

  lock_file = f"{cache_file}.lock"
  while True:
    try:
      lock_fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
      break
    except FileExistsError:
      cached = _load_paper_moabb_cache(cache_file)
      if cached is not None:
        return cached
      try:
        lock_age = time.time() - os.path.getmtime(lock_file)
      except OSError:
        lock_age = 0
      if lock_age > 6 * 60 * 60:
        try:
          os.remove(lock_file)
        except OSError:
          pass
      time.sleep(5)

  try:
    with os.fdopen(lock_fd, "w") as lock_handle:
      lock_handle.write(f"{os.getpid()}\n")
    cached = _load_paper_moabb_cache(cache_file)
    if cached is not None:
      return cached
    data, labels = build_fn()
    _save_paper_moabb_cache(cache_file, data, labels)
    return data.astype(np.float32), labels.astype(np.int64)
  finally:
    try:
      os.remove(lock_file)
    except OSError:
      pass


@lru_cache(maxsize=256)
def _load_openbmi_paper_moabb_subject_session(
  subject_id: str,
  session_index: int,
  channels: Literal["motor20", "all"] = "motor20",
  event_window_seconds: tuple[float, float] = DEFAULT_EVENT_WINDOW_SECONDS,
):
  """Load OpenBMI from MOABB and mirror upstream ADFCNN preprocessing."""
  event_window_seconds = normalize_paper_moabb_event_window(event_window_seconds)
  return _load_or_build_paper_moabb_cache(
    subject_id,
    session_index,
    channels,
    lambda: _build_openbmi_paper_moabb_subject_session(
      subject_id, session_index, channels, event_window_seconds
    ),
    event_window_seconds,
  )


def _build_openbmi_paper_moabb_subject_session(
  subject_id: str,
  session_index: int,
  channels: Literal["motor20", "all"],
  event_window_seconds: tuple[float, float] = DEFAULT_EVENT_WINDOW_SECONDS,
):
  """Build one OpenBMI subject-session with ADFCNN paper MOABB preprocessing."""
  (
    MOABBDataset,
    Preprocessor,
    preprocess,
    exponential_moving_standardize,
    create_windows_from_events,
  ) = _get_braindecode_moabb_api()
  from mne.filter import resample as mne_resample

  subject_num = int(subject_id)
  try:
    dataset = MOABBDataset(dataset_name="Lee2019_MI", subject_ids=[subject_num])
  except TypeError:
    dataset = MOABBDataset(dataset_name="Lee2019_MI", subject_ids=subject_num)

  factor_new = 1e-3
  init_block_size = 1000
  preprocessors = [
    Preprocessor(
      fn="pick_types",
      eeg=True,
      meg=False,
      stim=False,
      apply_on_array=True,
    ),
    Preprocessor(fn=lambda x: x * 1e6, apply_on_array=True),
    Preprocessor(fn="filter", l_freq=0, h_freq=40, apply_on_array=True),
    Preprocessor(
      fn=exponential_moving_standardize,
      factor_new=factor_new,
      init_block_size=init_block_size,
      apply_on_array=True,
    ),
  ]
  preprocess(dataset, preprocessors)

  sfreq = dataset.datasets[0].raw.info["sfreq"]
  if not all(ds.raw.info["sfreq"] == sfreq for ds in dataset.datasets):
    raise ValueError("MOABB Lee2019_MI raw files have inconsistent sampling rates.")

  windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=0,
    trial_stop_offset_samples=0,
    preload=True,
  )

  session_data = _select_moabb_session_split(
    windows_dataset.split("session"), session_index=session_index
  )
  x_list = []
  y_list = []
  for trial in session_data:
    x_list.append(trial[0])
    y_list.append(trial[1])

  data = np.array(x_list, dtype=np.float64)
  labels = np.array(y_list, dtype=np.int64)

  window_start, window_stop = normalize_paper_moabb_event_window(
    event_window_seconds
  )
  start_sample = round(window_start * sfreq)
  stop_sample = round(window_stop * sfreq)
  if stop_sample > data.shape[-1]:
    raise ValueError(
      f"Cue-aligned window {event_window_seconds} exceeds the MOABB epoch "
      f"with {data.shape[-1]} samples at {sfreq} Hz"
    )
  data = data[..., start_sample:stop_sample]
  data = mne_resample(data, 250 / sfreq)
  if channels == "motor20":
    data = data[:, 20:40, :]
  elif channels != "all":
    raise ValueError(f"Unsupported channel selection: {channels}")
  data = np.expand_dims(data.astype(np.float32), axis=1)
  return data, labels


def _load_openbmi_paper_moabb_session1_subject(subject_id: str):
  return _load_openbmi_paper_moabb_subject_session(subject_id, session_index=1)


def _load_openbmi_paper_moabb_session2_subject(subject_id: str):
  return _load_openbmi_paper_moabb_subject_session(subject_id, session_index=2)


def _load_openbmi_paper_moabb_session1_subject_channels(
  subject_id: str,
  channels: Literal["motor20", "all"],
  event_window_seconds: tuple[float, float] = DEFAULT_EVENT_WINDOW_SECONDS,
):
  return _load_openbmi_paper_moabb_subject_session(
    subject_id, session_index=1, channels=channels,
    event_window_seconds=event_window_seconds,
  )


def _load_openbmi_paper_moabb_session2_subject_channels(
  subject_id: str,
  channels: Literal["motor20", "all"],
  event_window_seconds: tuple[float, float] = DEFAULT_EVENT_WINDOW_SECONDS,
):
  return _load_openbmi_paper_moabb_subject_session(
    subject_id, session_index=2, channels=channels,
    event_window_seconds=event_window_seconds,
  )


class OpenBMI_MIPaperSession1Dataset(Dataset):
  """OpenBMI MI session-1-only preprocessing used for ADFCNN paper replication.

  Mirrors the upstream ADFCNN-MI OpenBMI path as closely as possible while using
  this project's local `.mat` cache instead of MOABB downloads:
    raw session1 -> 20 motor channels -> 0-40 Hz -> exponential standardization
    -> first 0-3s -> resample 1000 Hz to 250 Hz -> [1, 20, 750].
  """

  def __init__(
    self,
    subject_id: str,
    root_path=openbmi_path,
    get_item_style: Literal["dict", "tuple"] = "tuple",
  ):
    super().__init__()
    data, labels = _load_openbmi_paper_session1_subject(subject_id, str(root_path))

    self.subject_id = subject_id
    self.data = data
    self.labels = labels
    self.get_item_style = get_item_style

  def __len__(self):
    return self.data.shape[0]

  def __getitem__(self, index):
    eeg_data = self.data[index].astype(np.float32)
    label = self.labels[index]
    if self.get_item_style == "dict":
      return {"data": eeg_data, "target": label}
    return eeg_data, label, {}


class OpenBMI_MIPaperMOABBSession1Dataset(Dataset):
  """Strict ADFCNN paper replication path using MOABB raw continuous data.

  This follows the upstream ADFCNN-MI OpenBMI code order:
    MOABB Lee2019_MI raw -> pick EEG -> microvolt conversion -> 0-40 Hz filter
    -> continuous exponential moving standardization -> event windows
    -> session_1 -> 0-3s -> resample 250 Hz -> channels 20:40.
  """

  def __init__(
    self,
    subject_id: str,
    channels: Literal["motor20", "all"] = "motor20",
    get_item_style: Literal["dict", "tuple"] = "tuple",
    offline_transform_profile: str | None = None,
    event_window_seconds: tuple[float, float] = DEFAULT_EVENT_WINDOW_SECONDS,
  ):
    super().__init__()
    event_window_seconds = normalize_paper_moabb_event_window(event_window_seconds)
    if offline_transform_profile is None:
      data, labels = _load_openbmi_paper_moabb_session1_subject_channels(
        subject_id, channels=channels, event_window_seconds=event_window_seconds
      )
      self.transform_cache_directory = None
    else:
      cache_entry = load_or_build_transform_cache(
        offline_transform_profile,
        subject_id,
        session_index=1,
        channels=channels,
        event_window_seconds=event_window_seconds,
      )
      data, labels = cache_entry.data, cache_entry.labels
      self.transform_cache_directory = str(cache_entry.directory)
    self.subject_id = subject_id
    self.channels = channels
    self.offline_transform_profile = offline_transform_profile
    self.event_window_seconds = event_window_seconds
    self.data = data
    self.labels = labels
    self.get_item_style = get_item_style

  def __len__(self):
    return self.data.shape[0]

  def __getitem__(self, index):
    eeg_view, label, _ = self.get_item_view(index)
    # Keep direct indexing compatible with callers that expect a writable sample.
    eeg_data = np.array(eeg_view, dtype=np.float32, copy=True)
    if self.get_item_style == "dict":
      return {"data": eeg_data, "target": label}
    return eeg_data, label, {}

  def get_item_view(self, index):
    return self.data[index], self.labels[index], None


class OpenBMI_MIPaperMOABBSession2Dataset(OpenBMI_MIPaperMOABBSession1Dataset):
  """Strict ADFCNN paper replication path using MOABB raw session 2 only."""

  def __init__(
    self,
    subject_id: str,
    channels: Literal["motor20", "all"] = "motor20",
    get_item_style: Literal["dict", "tuple"] = "tuple",
    offline_transform_profile: str | None = None,
    event_window_seconds: tuple[float, float] = DEFAULT_EVENT_WINDOW_SECONDS,
  ):
    Dataset.__init__(self)
    event_window_seconds = normalize_paper_moabb_event_window(event_window_seconds)
    if offline_transform_profile is None:
      data, labels = _load_openbmi_paper_moabb_session2_subject_channels(
        subject_id, channels=channels, event_window_seconds=event_window_seconds
      )
      self.transform_cache_directory = None
    else:
      cache_entry = load_or_build_transform_cache(
        offline_transform_profile,
        subject_id,
        session_index=2,
        channels=channels,
        event_window_seconds=event_window_seconds,
      )
      data, labels = cache_entry.data, cache_entry.labels
      self.transform_cache_directory = str(cache_entry.directory)
    self.subject_id = subject_id
    self.channels = channels
    self.offline_transform_profile = offline_transform_profile
    self.event_window_seconds = event_window_seconds
    self.data = data
    self.labels = labels
    self.get_item_style = get_item_style


class OpenBMI_MIPaperMOABBSubjectDataset(Dataset):
  """Both-session OpenBMI subject dataset using ADFCNN paper MOABB preprocessing."""

  def __init__(
    self,
    subject_id: str,
    sessions: tuple[str, ...] = ("1", "2"),
    channels: Literal["motor20", "all"] = "motor20",
    get_item_style: Literal["dict", "tuple"] = "tuple",
    extra_info: Optional[dict] = None,
    offline_transform_profile: str | None = None,
    include_session_id: bool = False,
    event_window_seconds: tuple[float, float] = DEFAULT_EVENT_WINDOW_SECONDS,
  ):
    super().__init__()
    if len(sessions) == 0:
      raise ValueError("sessions must contain at least one session id")

    session_datasets = []
    for session in sessions:
      if session == "1":
        session_dataset = OpenBMI_MIPaperMOABBSession1Dataset(
          subject_id=subject_id,
          channels=channels,
          get_item_style=get_item_style,
          offline_transform_profile=offline_transform_profile,
          event_window_seconds=event_window_seconds,
        )
      elif session == "2":
        session_dataset = OpenBMI_MIPaperMOABBSession2Dataset(
          subject_id=subject_id,
          channels=channels,
          get_item_style=get_item_style,
          offline_transform_profile=offline_transform_profile,
          event_window_seconds=event_window_seconds,
        )
      else:
        raise ValueError(f"Unsupported OpenBMI session: {session}")
      session_datasets.append(session_dataset)

    self.subject_id = subject_id
    self.sessions = tuple(sessions)
    self.channels = channels
    self.offline_transform_profile = offline_transform_profile
    self.event_window_seconds = normalize_paper_moabb_event_window(
      event_window_seconds
    )
    self.session_datasets = session_datasets
    self._cumulative_lengths = np.cumsum(
      [len(dataset) for dataset in session_datasets], dtype=np.int64
    )
    self.labels = np.concatenate(
      [dataset.labels for dataset in session_datasets], axis=0
    ).astype(np.int64, copy=False)
    self.session_ids = np.concatenate(
      [
        np.full(len(dataset), int(session), dtype=np.int64)
        for session, dataset in zip(self.sessions, session_datasets)
      ],
      axis=0,
    )
    if offline_transform_profile is None:
      # Preserve the legacy public attributes for callers on the raw path.
      self.data = np.concatenate(
        [dataset.data for dataset in session_datasets], axis=0
      ).astype(np.float32, copy=False)
    self.get_item_style = get_item_style
    self.extra_info = extra_info
    self.include_session_id = include_session_id

  def __len__(self):
    if self.offline_transform_profile is None:
      return self.data.shape[0]
    return int(self._cumulative_lengths[-1])

  def __getitem__(self, index):
    eeg_view, label, extra_info = self.get_item_view(index)
    eeg_data = np.array(eeg_view, dtype=np.float32, copy=True)
    if self.get_item_style == "dict":
      return_dict = {"data": eeg_data, "target": label}
      if extra_info is not None:
        return_dict["extra_info"] = extra_info
      return return_dict
    return eeg_data, label, {} if extra_info is None else extra_info

  def get_item_view(self, index):
    if index < 0:
      index += len(self)
    if index < 0 or index >= len(self):
      raise IndexError(index)
    sample_extra_info = self.extra_info
    if self.include_session_id:
      sample_extra_info = dict(sample_extra_info or {})
      sample_extra_info["session_id"] = int(self.session_ids[index])
    if self.offline_transform_profile is None:
      return self.data[index], self.labels[index], sample_extra_info

    session_index = int(np.searchsorted(self._cumulative_lengths, index, side="right"))
    previous_length = 0 if session_index == 0 else int(
      self._cumulative_lengths[session_index - 1]
    )
    dataset = self.session_datasets[session_index]
    eeg_data = dataset.data[index - previous_length]
    label = dataset.labels[index - previous_length]
    return eeg_data, label, sample_extra_info


def _identity_paper_moabb_batch(batch):
  return batch


class OpenBMI_MIPaperMOABBBatchDataset(Dataset):
  """Batch-aware view over paper-MOABB datasets.

  PyTorch calls ``__getitems__`` with a complete batch of indices. The method
  allocates the final contiguous tensor once and copies each mmap-backed sample
  directly into its destination, avoiding per-sample copies followed by
  ``torch.stack``.
  """

  def __init__(
    self,
    datasets: Dataset | Sequence[Dataset],
    direct_pin_memory: bool = False,
  ):
    super().__init__()
    if isinstance(datasets, Dataset):
      datasets = [datasets]
    self.datasets = list(datasets)
    if not self.datasets:
      raise ValueError("datasets must contain at least one paper-MOABB dataset")
    if any(not hasattr(dataset, "get_item_view") for dataset in self.datasets):
      raise TypeError("paper-MOABB batch datasets must provide get_item_view()")

    self._cumulative_lengths = np.cumsum(
      [len(dataset) for dataset in self.datasets], dtype=np.int64
    )
    self.get_item_style = self.datasets[0].get_item_style
    if any(dataset.get_item_style != self.get_item_style for dataset in self.datasets):
      raise ValueError("all paper-MOABB datasets must use the same get_item_style")
    self.direct_pin_memory = direct_pin_memory

  def __len__(self):
    return int(self._cumulative_lengths[-1])

  def _resolve_index(self, index: int):
    if index < 0:
      index += len(self)
    if index < 0 or index >= len(self):
      raise IndexError(index)
    dataset_index = int(
      np.searchsorted(self._cumulative_lengths, index, side="right")
    )
    previous_length = 0 if dataset_index == 0 else int(
      self._cumulative_lengths[dataset_index - 1]
    )
    return self.datasets[dataset_index], index - previous_length

  def get_item_view(self, index: int):
    dataset, local_index = self._resolve_index(int(index))
    return dataset.get_item_view(local_index)

  def __getitem__(self, index):
    dataset, local_index = self._resolve_index(int(index))
    return dataset[local_index]

  def __getitems__(self, indices):
    indices = [int(index) for index in indices]
    if not indices:
      raise ValueError("cannot collate an empty paper-MOABB batch")

    first_eeg, _, _ = self.get_item_view(indices[0])
    sample_shape = tuple(first_eeg.shape)
    batch_eeg = torch.empty(
      (len(indices), *sample_shape),
      dtype=torch.float32,
      pin_memory=self.direct_pin_memory,
    )
    batch_labels = torch.empty(
      len(indices),
      dtype=torch.long,
      pin_memory=self.direct_pin_memory,
    )
    batch_eeg_array = batch_eeg.numpy()
    batch_label_array = batch_labels.numpy()
    extra_info = []

    for batch_index, source_index in enumerate(indices):
      eeg_view, label, sample_extra_info = self.get_item_view(source_index)
      if eeg_view.dtype != np.float32 or eeg_view.shape != sample_shape:
        raise ValueError(
          "paper-MOABB batch samples must have identical float32 shapes; "
          f"expected {sample_shape}, got {eeg_view.shape} {eeg_view.dtype}"
        )
      np.copyto(batch_eeg_array[batch_index], eeg_view, casting="no")
      batch_label_array[batch_index] = label
      extra_info.append(sample_extra_info)

    if self.get_item_style == "dict":
      batch = {"data": batch_eeg, "target": batch_labels}
      if any(info is not None for info in extra_info):
        if any(info is None for info in extra_info):
          raise ValueError("paper-MOABB batch metadata must be present for every sample")
        batch["extra_info"] = torch.utils.data.default_collate(extra_info)
      return batch

    tuple_extra_info = [info if info is not None else {} for info in extra_info]
    return (
      batch_eeg,
      batch_labels,
      torch.utils.data.default_collate(tuple_extra_info),
    )


def paper_moabb_batch_dataloader(
  datasets: Dataset | Sequence[Dataset],
  *,
  batch_size: int,
  num_workers: int,
  shuffle: bool = False,
  sampler=None,
  pin_memory: bool = True,
):
  """Create a DataLoader that bypasses per-sample copy and default stacking."""
  if shuffle and sampler is not None:
    raise ValueError("shuffle and sampler cannot both be set")
  accelerator_available = torch.cuda.is_available()
  direct_pin_memory = pin_memory and accelerator_available and num_workers == 0
  batch_dataset = OpenBMI_MIPaperMOABBBatchDataset(
    datasets,
    direct_pin_memory=direct_pin_memory,
  )
  return torch.utils.data.DataLoader(
    batch_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=shuffle,
    sampler=sampler,
    collate_fn=_identity_paper_moabb_batch,
    persistent_workers=True if num_workers > 0 else False,
    pin_memory=pin_memory and accelerator_available and not direct_pin_memory,
    prefetch_factor=2 if num_workers > 0 else None,
  )


class OpenBMI_MIPaperSession1KfoldDataModule(L.LightningDataModule):
  """ADFCNN paper-style within-session KFold on OpenBMI session 1 only."""

  def __init__(
    self,
    subject_id: str,
    fold: int,
    kfold=5,
    random_seed=42,
    batch_size=16,
    num_workers=0,
    get_item_style: Literal["tuple", "dict"] = "tuple",
  ):
    super().__init__()
    self.subject_id = subject_id
    self.fold = fold
    self.kfold = kfold
    self.random_seed = random_seed
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.get_item_style = get_item_style

  def setup(self, stage: str):
    self.dataset = OpenBMI_MIPaperSession1Dataset(
      subject_id=self.subject_id,
      get_item_style=self.get_item_style,
    )
    splitter = KFold(
      n_splits=self.kfold,
      shuffle=True,
      random_state=self.random_seed,
    )
    splits = list(splitter.split(np.arange(len(self.dataset))))
    train_idx, heldout_idx = splits[self.fold - 1]
    self.train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    self.heldout_sampler = torch.utils.data.SubsetRandomSampler(heldout_idx)

  def train_dataloader(self):
    return torch.utils.data.DataLoader(
      self.dataset,
      batch_size=self.batch_size,
      num_workers=self.num_workers,
      sampler=self.train_sampler,
    )

  def val_dataloader(self):
    return torch.utils.data.DataLoader(
      self.dataset,
      batch_size=self.batch_size,
      num_workers=self.num_workers,
      sampler=self.heldout_sampler,
    )

  def test_dataloader(self):
    return torch.utils.data.DataLoader(
      self.dataset,
      batch_size=self.batch_size,
      num_workers=self.num_workers,
      sampler=self.heldout_sampler,
    )


class OpenBMI_MIPaperMOABBSession1KfoldDataModule(OpenBMI_MIPaperSession1KfoldDataModule):
  """ADFCNN paper-style KFold using MOABB raw-continuous preprocessing."""

  def __init__(
    self,
    subject_id: str,
    fold: int,
    kfold=5,
    random_seed=42,
    batch_size=16,
    num_workers=0,
    get_item_style: Literal["tuple", "dict"] = "tuple",
    channels: Literal["motor20", "all"] = "motor20",
    offline_transform_profile: str | None = None,
    separate_validation: bool = False,
    validation_fraction: float = 0.2,
    event_window_seconds: tuple[float, float] = DEFAULT_EVENT_WINDOW_SECONDS,
  ):
    super().__init__(
      subject_id=subject_id,
      fold=fold,
      kfold=kfold,
      random_seed=random_seed,
      batch_size=batch_size,
      num_workers=num_workers,
      get_item_style=get_item_style,
    )
    self.channels = channels
    self.offline_transform_profile = offline_transform_profile
    self.separate_validation = separate_validation
    self.validation_fraction = validation_fraction
    self.event_window_seconds = normalize_paper_moabb_event_window(
      event_window_seconds
    )

  def _set_split_samplers(self):
    splitter = KFold(
      n_splits=self.kfold,
      shuffle=True,
      random_state=self.random_seed,
    )
    splits = list(splitter.split(np.arange(len(self.dataset))))
    outer_train_idx, test_idx = splits[self.fold - 1]

    if self.separate_validation:
      inner_splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=self.validation_fraction,
        random_state=self.random_seed,
      )
      inner_train_rel, val_rel = next(
        inner_splitter.split(outer_train_idx, self.dataset.labels[outer_train_idx])
      )
      train_idx = outer_train_idx[inner_train_rel]
      val_idx = outer_train_idx[val_rel]
    else:
      train_idx = outer_train_idx
      val_idx = test_idx

    self.train_indices = np.asarray(train_idx, dtype=np.int64)
    self.val_indices = np.asarray(val_idx, dtype=np.int64)
    self.test_indices = np.asarray(test_idx, dtype=np.int64)
    self.train_sampler = torch.utils.data.SubsetRandomSampler(self.train_indices)
    self.val_sampler = torch.utils.data.SubsetRandomSampler(self.val_indices)
    self.test_sampler = torch.utils.data.SubsetRandomSampler(self.test_indices)

  def setup(self, stage: str):
    self.dataset = OpenBMI_MIPaperMOABBSession1Dataset(
      subject_id=self.subject_id,
      channels=self.channels,
      get_item_style=self.get_item_style,
      offline_transform_profile=self.offline_transform_profile,
      event_window_seconds=self.event_window_seconds,
    )
    self._set_split_samplers()

  def train_dataloader(self):
    return paper_moabb_batch_dataloader(
      self.dataset,
      batch_size=self.batch_size,
      num_workers=self.num_workers,
      sampler=self.train_sampler,
    )

  def val_dataloader(self):
    return paper_moabb_batch_dataloader(
      self.dataset,
      batch_size=self.batch_size,
      num_workers=self.num_workers,
      sampler=self.val_sampler,
    )

  def test_dataloader(self):
    return paper_moabb_batch_dataloader(
      self.dataset,
      batch_size=self.batch_size,
      num_workers=self.num_workers,
      sampler=self.test_sampler,
    )


class OpenBMI_MIPaperMOABBSession2KfoldDataModule(OpenBMI_MIPaperMOABBSession1KfoldDataModule):
  """ADFCNN paper-style KFold using MOABB raw-continuous session 2."""

  def setup(self, stage: str):
    self.dataset = OpenBMI_MIPaperMOABBSession2Dataset(
      subject_id=self.subject_id,
      channels=self.channels,
      get_item_style=self.get_item_style,
      offline_transform_profile=self.offline_transform_profile,
      event_window_seconds=self.event_window_seconds,
    )
    self._set_split_samplers()


class OpenBMI_MIPaperMOABBPooledSessionsKfoldDataModule(
  OpenBMI_MIPaperMOABBSession1KfoldDataModule
):
  """Subject-dependent KFold over both PaperMOABB offline sessions.

  The two 100-trial offline sessions are preprocessed independently and then
  pooled. Both outer test folds and the optional inner validation split are
  stratified by the joint ``(session, class)`` stratum, so every split contains
  a balanced contribution from both recording days and both MI classes.
  """

  pooled_sessions = ("1", "2")

  def __init__(self, *args, include_session_id: bool = False, **kwargs):
    super().__init__(*args, **kwargs)
    self.include_session_id = include_session_id

  def _joint_session_class_strata(self) -> np.ndarray:
    labels = np.asarray(self.dataset.labels)
    session_ids = np.asarray(self.dataset.session_ids)
    if labels.ndim != 1 or session_ids.ndim != 1:
      raise ValueError("labels and session_ids must be one-dimensional")
    if len(labels) != len(self.dataset) or len(session_ids) != len(self.dataset):
      raise ValueError("labels and session_ids must align with the pooled dataset")

    _, label_codes = np.unique(labels, return_inverse=True)
    _, session_codes = np.unique(session_ids, return_inverse=True)
    num_classes = len(np.unique(label_codes))
    return session_codes * num_classes + label_codes

  def _set_split_samplers(self):
    if self.fold < 1 or self.fold > self.kfold:
      raise ValueError(f"fold must be in [1, {self.kfold}], got {self.fold}")
    if self.separate_validation and not 0 < self.validation_fraction < 1:
      raise ValueError("validation_fraction must be between 0 and 1")

    indices = np.arange(len(self.dataset), dtype=np.int64)
    joint_strata = self._joint_session_class_strata()
    splitter = StratifiedKFold(
      n_splits=self.kfold,
      shuffle=True,
      random_state=self.random_seed,
    )
    splits = list(splitter.split(indices, joint_strata))
    outer_train_idx, test_idx = splits[self.fold - 1]

    if self.separate_validation:
      inner_splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=self.validation_fraction,
        random_state=self.random_seed,
      )
      inner_train_rel, val_rel = next(
        inner_splitter.split(outer_train_idx, joint_strata[outer_train_idx])
      )
      train_idx = outer_train_idx[inner_train_rel]
      val_idx = outer_train_idx[val_rel]
    else:
      train_idx = outer_train_idx
      val_idx = test_idx

    self.joint_strata = joint_strata
    self.train_indices = np.asarray(train_idx, dtype=np.int64)
    self.val_indices = np.asarray(val_idx, dtype=np.int64)
    self.test_indices = np.asarray(test_idx, dtype=np.int64)
    self.train_sampler = torch.utils.data.SubsetRandomSampler(self.train_indices)
    self.val_sampler = torch.utils.data.SubsetRandomSampler(self.val_indices)
    self.test_sampler = torch.utils.data.SubsetRandomSampler(self.test_indices)

  def setup(self, stage: str):
    self.dataset = OpenBMI_MIPaperMOABBSubjectDataset(
      subject_id=self.subject_id,
      sessions=self.pooled_sessions,
      channels=self.channels,
      get_item_style=self.get_item_style,
      offline_transform_profile=self.offline_transform_profile,
      include_session_id=self.include_session_id,
      event_window_seconds=self.event_window_seconds,
    )
    self._set_split_samplers()


class OpenBMI_MIDataset(Dataset):
  def __init__(
    self,
    session: Literal["1", "2"],
    subject_id: str,
    root_path=openbmi_path,
    data_key="data",
    transform: Optional[transforms.EEGTransform] = None,
    extra_info: Optional[dict] = None,
    get_item_style: Literal["dict", "tuple"] = "dict",
    stats: Optional[ChannelStats] = None,
    event_window_seconds: tuple[float, float] | None = None,
  ):
    super().__init__()
    self.subject_id = subject_id
    self.session = session
    self.root_path = root_path
    self.data_key = data_key

    subject_mi_file = os.path.join(
      root_path,
      f"session{session}",
      f"s{subject_id}",
      f"sess{session.zfill(2)}_subj{subject_id.zfill(2)}_EEG_MI.mat",
    )
    self.subject_mi_file = subject_mi_file

    data, labels = read_openbmi_subject_mat(
      self.subject_mi_file, data_key="EEG_MI_train"
    )

    self.transform = transform
    self.extra_info = extra_info
    self.get_item_style = get_item_style
    self.stats = stats

    data = _crop_openbmi_event_window(data, event_window_seconds)
    data = resample_poly(data, up=1, down=4, axis=-1)  # 1000Hz -> 250Hz

    self.data = data.astype(np.float32)
    self.labels = labels

  def __len__(self):
    return self.data.shape[0]

  def __getitem__(self, index):
    eeg_data = self.data[index].astype(np.float32)

    if self.stats is not None:
      eeg_data = self.stats.apply(eeg_data).astype(np.float32)

    if self.transform is not None:
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


class OpenBMI_MI_SubjectDependent_Holdout_DataModule(L.LightningDataModule):
  def __init__(self, subject_id: str, random_seed=42, batch_size=500, num_workers=4):
    super().__init__()
    self.subject_id = subject_id
    self.random_seed = random_seed
    self.batch_size = batch_size
    self.num_workers = num_workers

  def setup(self, stage: str):
    training_data = OpenBMI_MIDataset(subject_id=self.subject_id, session="0")
    testset = OpenBMI_MIDataset(subject_id=self.subject_id, session="1")

    training_set, validation_set = random_split(
      training_data,
      [0.8, 0.2],
      generator=torch.Generator().manual_seed(self.random_seed),
    )
    if stage == "fit":
      self.train_loader = torch.utils.data.DataLoader(
        training_set, batch_size=self.batch_size, num_workers=self.num_workers
      )
      self.validation_loader = torch.utils.data.DataLoader(
        validation_set, batch_size=self.batch_size, num_workers=self.num_workers
      )
    if stage == "test":
      self.test_loader = torch.utils.data.DataLoader(
        testset, batch_size=self.batch_size, num_workers=self.num_workers
      )

  def train_dataloader(self):
    return self.train_loader

  def val_dataloader(self):
    return self.validation_loader

  def test_dataloader(self):
    return self.test_loader


class OpenBMI_MI_SessionDependent_Kfold_DataModule(L.LightningDataModule):
  def __init__(
    self,
    subject_id: str,
    fold: int,
    session: Literal["1", "2"],
    kfold=5,
    random_seed=42,
    batch_size=500,
    num_workers=4,
    transform: Optional[transforms.EEGTransform] = None,
    get_item_style: Literal["tuple", "dict"] = "dict",
    stats: Optional[ChannelStats] = None,
  ):
    super().__init__()
    self.subject_id = subject_id
    self.random_seed = random_seed
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.session = session
    self.kfold = kfold
    self.fold = fold
    self.transform = transform
    self.get_item_style = get_item_style
    self.stats = stats

  def setup(self, stage: str):
    data = OpenBMI_MIDataset(
      subject_id=self.subject_id,
      session=self.session,
      transform=self.transform,
      get_item_style=self.get_item_style,
      stats=self.stats,
    )

    training_set, validation_set, testset = kfold_split(
      data, fold=self.fold, kfold=self.kfold, random_seed=self.random_seed
    )

    if stage == "fit":
      self.train_loader = torch.utils.data.DataLoader(
        training_set, batch_size=self.batch_size, num_workers=self.num_workers
      )
      self.validation_loader = torch.utils.data.DataLoader(
        validation_set, batch_size=self.batch_size, num_workers=self.num_workers
      )
    if stage == "test":
      self.test_loader = torch.utils.data.DataLoader(
        testset, batch_size=self.batch_size, num_workers=self.num_workers
      )

  def train_dataloader(self):
    return self.train_loader

  def val_dataloader(self):
    return self.validation_loader

  def test_dataloader(self):
    return self.test_loader


class OpenBMI_MI_SubjectDependent_Kfold_DataModule(L.LightningDataModule):
  """Merges both sessions for a subject, then does k-fold split."""

  def __init__(
    self,
    subject_id: str,
    fold: int,
    kfold=5,
    random_seed=42,
    batch_size=500,
    num_workers=4,
    transform: Optional[transforms.EEGTransform] = None,
    get_item_style: Literal["tuple", "dict"] = "dict",
    stats: Optional[ChannelStats] = None,
  ):
    super().__init__()
    self.subject_id = subject_id
    self.random_seed = random_seed
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.kfold = kfold
    self.fold = fold
    self.transform = transform
    self.get_item_style = get_item_style
    self.stats = stats

  def setup(self, stage: str):
    ds1 = OpenBMI_MIDataset(
      subject_id=self.subject_id,
      session="1",
      transform=self.transform,
      get_item_style=self.get_item_style,
      stats=self.stats,
    )
    ds2 = OpenBMI_MIDataset(
      subject_id=self.subject_id,
      session="2",
      transform=self.transform,
      get_item_style=self.get_item_style,
      stats=self.stats,
    )
    merged = torch.utils.data.ConcatDataset([ds1, ds2])

    training_set, validation_set, testset = stratified_kfold_split(
      merged, fold=self.fold, kfold=self.kfold, random_seed=self.random_seed
    )

    if stage == "fit":
      self.train_loader = torch.utils.data.DataLoader(
        training_set, batch_size=self.batch_size, num_workers=self.num_workers
      )
      self.validation_loader = torch.utils.data.DataLoader(
        validation_set, batch_size=self.batch_size, num_workers=self.num_workers
      )
    if stage == "test":
      self.test_loader = torch.utils.data.DataLoader(
        testset, batch_size=self.batch_size, num_workers=self.num_workers
      )

  def train_dataloader(self):
    return self.train_loader

  def val_dataloader(self):
    return self.validation_loader

  def test_dataloader(self):
    return self.test_loader


if __name__ == "__main__":
  data = OpenBMI_MIDataset(subject_id="1", session="2")

  training_set, validation_set, testset = kfold_split(
    data, fold=1, kfold=5, random_seed=42
  )
  print(f"OpenBMI_MIDataset loaded training_set length: {len(training_set)}")
  print(f"OpenBMI_MIDataset loaded validation_set length: {len(validation_set)}")
  print(f"OpenBMI_MIDataset loaded testset length: {len(testset)}")
