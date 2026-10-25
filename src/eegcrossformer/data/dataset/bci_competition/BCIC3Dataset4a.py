from torcheeg.transforms import BandDifferentialEntropy
from eegcrossformer.project_config import bcic3_dataset4a_path
from torch.utils.data import Dataset
from typing import Literal, Optional
from torcheeg import transforms
import scipy.io as sciio
import torch as torch
import numpy as np
import os


class BcicompetitionIIIDatasetIVaRawTrailDataset(Dataset):
  root_path = ""
  cue_n_times = 350
  data_path = ""

  def __init__(
    self,
    root_path=bcic3_dataset4a_path,
    subject_code: str = "aa",  # subject_codes = ['aa', 'al', 'av', 'aw', 'ay']
    do_normalization=False,
    selected_ch_indexes=None,
    get_item_style: Literal["dict", "tuple"] = "dict",
    transform: Optional[transforms.EEGTransform] = None,
    extra_info: Optional[dict] = None,
  ) -> None:
    super().__init__()
    self.root_path = root_path
    self.do_normalization = do_normalization
    all_eeg, target, ch_names, x_pos, y_pos = self.read_mat(subject_code)
    self.data = all_eeg
    self.target = target
    self.ch_names = ch_names
    self.x_pos = x_pos
    self.y_pos = y_pos
    self.selected_ch_indexes = selected_ch_indexes
    self.get_item_style = get_item_style
    self.transform = transform
    self.extra_info = extra_info

  def read_mat(self, subject_code):
    root_path = self.root_path
    cue_n_times = self.cue_n_times

    mat_full_path = os.path.join(root_path, f"data_set_IVa_{subject_code}.mat")
    self.data_path = mat_full_path
    mat_dict = sciio.loadmat(mat_full_path)
    true_label_path = os.path.join(root_path, f"true_labels_{subject_code}.mat")
    true_label_mat_dict = sciio.loadmat(true_label_path)
    mrk = mat_dict["mrk"]
    nfo = mat_dict["nfo"]

    cue_onsets = np.squeeze(mrk[0][0][0])
    target = np.squeeze(true_label_mat_dict["true_y"]) - 1
    class_names = np.squeeze(mrk[0][0][2])
    class_names = np.array([class_names[0][0], class_names[1][0]])
    x_pos = np.squeeze(nfo[0][0][3])
    y_pos = np.squeeze(nfo[0][0][4])
    ch_names = np.array(list(map(lambda x: x[0], np.squeeze(nfo[0][0][2]))))
    eeg = mat_dict["cnt"] * 0.1  # [T, C]
    eeg = (eeg.T - np.mean(eeg, axis=1)).T  # common average reference
    # sos = signal.butter(8, [4, 40], btype="band", fs=100, output='sos')
    # eeg = signal.sosfilt(sos, eeg, axis=1)
    all_eeg = None
    for cue_onset in cue_onsets:
      mi_eeg = eeg[cue_onset : cue_onset + cue_n_times]
      if all_eeg is None:
        all_eeg = mi_eeg
      else:
        all_eeg = np.vstack([all_eeg, mi_eeg])
    return all_eeg, target, ch_names, x_pos, y_pos

  def __len__(self):
    return len(self.target)

  def __getitem__(self, idx):
    eeg = self.data[idx * self.cue_n_times : (idx + 1) * self.cue_n_times].T
    if self.selected_ch_indexes != None:
      eeg = np.take(eeg, self.selected_ch_indexes, axis=0)

    eeg_data = (
      self.transform(eeg=eeg.astype(np.float32))["eeg"]
      if self.transform is not None
      else eeg.astype(np.float32)
    )
    if self.get_item_style == "dict":
      return_dict = {
        "data": eeg_data,
        "target": self.target[idx],
      }
      if self.extra_info is not None:
        return_dict["extra_info"] = self.extra_info
      return return_dict
    else:
      return_tuple = (
        eeg_data,
        self.target[idx],
        {} if self.extra_info is None else self.extra_info,
      )
      return return_tuple


class BCIC34aDEDataset(BcicompetitionIIIDatasetIVaRawTrailDataset):
  def __init__(
    self,
    root_path=bcic3_dataset4a_path,
    subject_code: str = "aa",
    do_normalization=False,
    selected_ch_indexes=None,
    band_dict=None,
    feature_save_path=None,
  ) -> None:
    super().__init__(
      root_path=root_path,
      subject_code=subject_code,
      do_normalization=do_normalization,
      selected_ch_indexes=selected_ch_indexes,
    )
    self.band_dict = band_dict or {
      "delta": [1, 4],
      "theta": [4, 8],
      "alpha": [8, 14],
      "beta": [14, 31],
      "gamma": [31, 49],
    }
    self.feature_save_path = feature_save_path
    self.transform = BandDifferentialEntropy(band_dict=self.band_dict)

    self.features = self._compute_features()
    if self.feature_save_path:
      self._save_features()

  def _compute_features(self):
    features = []
    for idx in range(len(self)):
      sample = super().__getitem__(idx)
      eeg_data = sample["data"]
      de_feature = self.transform(eeg=eeg_data)
      de_feature = de_feature["eeg"]
      features.append(de_feature)
    return np.stack(features)

  def _save_features(self):
    if not os.path.exists(os.path.dirname(self.feature_save_path)):
      os.makedirs(os.path.dirname(self.feature_save_path))
    np.save(self.feature_save_path, self.features)
    print(f"Features saved to {self.feature_save_path}")

  def __getitem__(self, idx):
    return {
      "data": self.features[idx].astype(np.float32),
      "target": self.target[idx],
    }

  def __len__(self):
    return len(self.target)


class BCI3IVaSSVEPFormerDataset(Dataset):
  def __init__(
    self,
    root_path=bcic3_dataset4a_path,
    subject_code: str = "aa",
    do_normalization=False,
    selected_ch_indexes=None,
    band=[8, 64],
    resolution=0.5,
  ) -> None:
    super().__init__()
    self.root_path = root_path
    self.do_normalization = do_normalization
    self.selected_ch_indexes = selected_ch_indexes
    self.band = band
    self.resolution = resolution

    # Load raw data
    all_eeg, target, ch_names, x_pos, y_pos = self.read_mat(subject_code)
    self.data = all_eeg
    self.target = target
    self.ch_names = ch_names
    self.x_pos = x_pos
    self.y_pos = y_pos

  def read_mat(self, subject_code):
    root_path = self.root_path
    cue_n_times = 350  # Assuming this is the length of each trial

    mat_full_path = os.path.join(root_path, f"data_set_IVa_{subject_code}.mat")
    mat_dict = sciio.loadmat(mat_full_path)
    true_label_path = os.path.join(root_path, f"true_labels_{subject_code}.mat")
    true_label_mat_dict = sciio.loadmat(true_label_path)
    mrk = mat_dict["mrk"]
    nfo = mat_dict["nfo"]

    cue_onsets = np.squeeze(mrk[0][0][0])
    target = np.squeeze(true_label_mat_dict["true_y"]) - 1
    class_names = np.squeeze(mrk[0][0][2])
    class_names = np.array([class_names[0][0], class_names[1][0]])
    x_pos = np.squeeze(nfo[0][0][3])
    y_pos = np.squeeze(nfo[0][0][4])
    ch_names = np.array(list(map(lambda x: x[0], np.squeeze(nfo[0][0][2]))))
    eeg = mat_dict["cnt"] * 0.1  # [T, C]
    eeg = (eeg.T - np.mean(eeg, axis=1)).T  # common average reference

    all_eeg = None
    for cue_onset in cue_onsets:
      mi_eeg = eeg[cue_onset : cue_onset + cue_n_times]

      if all_eeg is None:
        all_eeg = mi_eeg
      else:
        all_eeg = np.vstack([all_eeg, mi_eeg])

    return all_eeg, target, ch_names, x_pos, y_pos

  def __len__(self):
    return len(self.target)

  def __getitem__(self, idx):
    eeg = self.data[idx * 350 : (idx + 1) * 350].T  # Assuming 350 time points per trial
    if self.selected_ch_indexes is not None:
      eeg = np.take(eeg, self.selected_ch_indexes, axis=0)

    # Apply FFT transformation
    eeg_tensor = torch.tensor(eeg, dtype=torch.float32)
    nfft = round(100 / self.resolution)  # Assuming sampling rate is 100Hz
    fft_start = int(round(self.band[0] / self.resolution))
    fft_end = int(round(self.band[1] / self.resolution)) + 1

    with torch.no_grad():
      x = torch.fft.fft(eeg_tensor, n=nfft) / 350
      real = x.real[:, fft_start:fft_end]
      imag = x.imag[:, fft_start:fft_end]
      transformed_eeg = torch.cat((real, imag), axis=-1)

    return {"data": transformed_eeg, "target": self.target[idx]}


if __name__ == "__main__":
  dataset = BcicompetitionIIIDatasetIVaRawTrailDataset()
  print(len(dataset))
