from typing import Literal
from pymatreader import read_mat
import numpy as np


def read_openbmi_subject_mat(
  subject_mi_path: str, data_key: Literal["EEG_SSVEP_train", "EEG_MI_train"]
) -> tuple[np.ndarray, np.ndarray]:
  mat_data = read_mat(subject_mi_path)
  data = mat_data[data_key]
  mi_trails = data["smt"].transpose(1, 2, 0)
  mi_labels = data["y_dec"]
  return mi_trails, mi_labels - 1
