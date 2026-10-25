from typing import List, Optional, Sequence, Tuple, Union, Any

from torcheeg.transforms import EEGTransform, BandSignal
import scipy.signal as signal
import numpy as np
from typing import Dict
import torch

FREQ_BANDS = [
  [4, 8],
  [8, 12],
  [12, 16],
  [16, 20],
  [20, 24],
  [24, 28],
  [28, 32],
  [32, 36],
  [36, 40],
]


def transform_to_fbmultiview(model_input: np.ndarray, sfreq=100, bands=FREQ_BANDS):
  # model_input.shape: [B, C, L]
  B, C, L = model_input.shape
  batch_list = []
  for i in range(0, B):
    batch_list.append(
      np.expand_dims(
        create_bandmultivieweeg(
          np.squeeze(model_input[i, :, :]), bands=bands, sfreq=sfreq
        ),
        0,
      )
    )
  return np.concatenate(batch_list, axis=0)


class AddViewDimTransform(EEGTransform):
  def __init__(self, apply_to_baseline: bool = False):
    super().__init__(apply_to_baseline)

  def apply(self, eeg: any, baseline: Union[any, None] = None, **kwargs) -> any:
    """
    eeg.shape [Batch, Electrode Number, N_Times]
    output [Batch, View, Channel, Length]
    """
    if not isinstance(eeg, np.ndarray):
      raise ValueError("eeg should be a ndarray")
    return np.expand_dims(eeg, 1)


def create_bandmultivieweeg(
  eeg_trail: np.ndarray, bands: List[Tuple[float, float]], sfreq=100
):

  # eeg_trail.shape: [C, L]
  views = []
  for low, high in bands:
    if low <= 0:
      sos = signal.butter(2, high, btype="lowpass", fs=sfreq, output="sos")
    else:
      sos = signal.butter(
        2,
        (low, high),
        btype="bandpass",
        fs=sfreq,
        output="sos",
      )
    # Zero-phase filtering (the usual offline choice)
    y = signal.sosfiltfilt(sos, eeg_trail, axis=-1)
    views.append(y)
  return np.stack(views, axis=0)  # [V, C, L]


class TrailFilteringBankTransform(EEGTransform):
  def __init__(
    self, freq_bands=FREQ_BANDS, sfreq: float = 100.0, apply_to_baseline: bool = False
  ):
    super().__init__(apply_to_baseline)
    self.freq_bands = freq_bands
    self.sfreq = float(sfreq)

  def apply(self, eeg: Any, baseline: Union[Any, None] = None, **kwargs) -> Any:
    """
    eeg.shape: [Electrode Number, N_Times] or [1, Electrode Number, N_Times]
    output: [View, Channel, Length]

    Supports both numpy.ndarray and torch.Tensor inputs.
    Returns the same type and dtype as the input.
    """
    is_torch_tensor = isinstance(eeg, torch.Tensor)

    if is_torch_tensor:
      original_device = eeg.device
      original_dtype = eeg.dtype
      eeg_np = eeg.detach().cpu().numpy()
    elif isinstance(eeg, np.ndarray):
      original_dtype = eeg.dtype
      eeg_np = eeg
    else:
      raise ValueError("eeg should be a ndarray or torch.Tensor")

    ndim = eeg_np.ndim
    if ndim == 3:
      if eeg_np.shape[0] != 1:
        raise ValueError(
          f"For 3D input, first dimension must be 1, got shape {eeg_np.shape}"
        )
      eeg_np = np.squeeze(eeg_np, axis=0)
    elif ndim != 2:
      raise ValueError(
        f"eeg should be 2D [Electrode, Time] or 3D [1, Electrode, Time], got {ndim}D with shape {eeg_np.shape}"
      )

    # Pass sfreq explicitly; create_bandmultivieweeg otherwise assumes 100 Hz
    result = create_bandmultivieweeg(
      eeg_trail=eeg_np, bands=self.freq_bands, sfreq=self.sfreq
    )

    if is_torch_tensor:
      result = torch.from_numpy(result).to(dtype=original_dtype, device=original_device)
    else:
      if result.dtype != original_dtype:
        result = result.astype(original_dtype)

    return result


class CARTransform(EEGTransform):
  """
  Common Average Reference (CAR):
    For each time point t, subtract the mean across channels.

  Supported shapes:
    - [C, T]
    - [V, C, T]  (including [1, C, T])

  Returns the same type (numpy / torch), dtype and (for torch) device as input.
  """

  def __init__(
    self,
    exclude: Optional[Sequence[int]] = None,
    apply_to_baseline: bool = False,
  ):
    super().__init__(apply_to_baseline)
    self.exclude = None if exclude is None else list(exclude)

  def apply(self, eeg: Any, baseline: Union[Any, None] = None, **kwargs) -> Any:
    """
    eeg: np.ndarray or torch.Tensor
    """
    is_torch_tensor = isinstance(eeg, torch.Tensor)

    # -------- Torch path (keep on-device) --------
    if is_torch_tensor:
      original_device = eeg.device
      original_dtype = eeg.dtype

      x = eeg
      if x.ndim == 2:
        # [C, T] -> mean over C
        ch_dim = 0
      elif x.ndim == 3:
        # [V, C, T] -> mean over C
        ch_dim = 1
      else:
        raise ValueError(
          f"eeg should be 2D [C, T] or 3D [V, C, T], got {x.ndim}D with shape {tuple(x.shape)}"
        )

      # Build channel index (optionally excluding some channels)
      if self.exclude:
        C = x.shape[ch_dim]
        keep_idx = [i for i in range(C) if i not in set(self.exclude)]
        if len(keep_idx) == 0:
          raise ValueError(
            "exclude removes all channels; CAR needs at least 1 channel to compute the reference."
          )
        idx = torch.as_tensor(keep_idx, device=original_device, dtype=torch.long)
        x_ref = torch.index_select(x, dim=ch_dim, index=idx)
      else:
        x_ref = x

      # torch.mean requires floating type; if not, temporarily cast
      if not torch.is_floating_point(x_ref):
        x_ref_float = x_ref.float()
        ref = x_ref_float.mean(dim=ch_dim, keepdim=True)
        out = x.float() - ref
        out = out.to(dtype=original_dtype, device=original_device)
      else:
        ref = x_ref.mean(dim=ch_dim, keepdim=True)
        out = x - ref
        out = out.to(dtype=original_dtype, device=original_device)

      return out

    # -------- NumPy path --------
    if isinstance(eeg, np.ndarray):
      original_dtype = eeg.dtype
      x = eeg

      if x.ndim == 2:
        # [C, T]
        ch_axis = 0
      elif x.ndim == 3:
        # [V, C, T]
        ch_axis = 1
      else:
        raise ValueError(
          f"eeg should be 2D [C, T] or 3D [V, C, T], got {x.ndim}D with shape {x.shape}"
        )

      if self.exclude:
        C = x.shape[ch_axis]
        keep_idx = [i for i in range(C) if i not in set(self.exclude)]
        if len(keep_idx) == 0:
          raise ValueError(
            "exclude removes all channels; CAR needs at least 1 channel to compute the reference."
          )
        x_ref = np.take(x, indices=keep_idx, axis=ch_axis)
      else:
        x_ref = x

      ref = x_ref.mean(axis=ch_axis, keepdims=True)
      out = x - ref

      # Keep numpy output dtype consistent with input (optional but matches your style)
      if out.dtype != original_dtype:
        out = out.astype(original_dtype, copy=False)

      return out

    raise ValueError("eeg should be a numpy.ndarray or torch.Tensor")


class FilteringBankTransform(EEGTransform):
  def __init__(
    self, freq_bands=FREQ_BANDS, apply_to_baseline: bool = False, sfreq: float = 100.0
  ):
    super().__init__(apply_to_baseline)
    self.freq_bands = freq_bands
    self.sfreq = float(sfreq)

  def apply(self, eeg: any, baseline: Union[any, None] = None, **kwargs) -> any:
    """
    eeg.shape: [Batch, Electrode Number, N_Times] or [Batch, 1, Electrode Number, N_Times]
    output: [Batch, View, Channel, Length]

    Supports both numpy.ndarray and torch.Tensor inputs.
    Returns the same type and dtype as the input.
    """
    is_torch_tensor = isinstance(eeg, torch.Tensor)

    if is_torch_tensor:
      original_device = eeg.device
      original_dtype = eeg.dtype
      eeg_np = eeg.cpu().numpy()
    elif isinstance(eeg, np.ndarray):
      original_dtype = eeg.dtype
      eeg_np = eeg
    else:
      raise ValueError("eeg should be a ndarray or torch.Tensor")

    ndim = eeg_np.ndim
    if ndim == 4:
      if eeg_np.shape[1] != 1:
        raise ValueError(
          f"For 4D input, second dimension must be 1, got shape {eeg_np.shape}"
        )
      eeg_np = np.squeeze(eeg_np, axis=1)
    elif ndim != 3:
      raise ValueError(
        f"eeg should be 3D [Batch, Electrode, Time] or 4D [Batch, 1, Electrode, Time], got {ndim}D with shape {eeg_np.shape}"
      )

    result = transform_to_fbmultiview(eeg_np, bands=self.freq_bands, sfreq=self.sfreq)

    if is_torch_tensor:
      result = torch.from_numpy(result).to(dtype=original_dtype, device=original_device)
    else:
      # Ensure numpy output has same dtype as input
      if result.dtype != original_dtype:
        result = result.astype(original_dtype)

    return result


class PaddingTransform(EEGTransform):
  def __init__(self, padding=0):
    super().__init__(False)
    self.padding = padding

  def apply(self, eeg: any, baseline: Union[any, None] = None, **kwargs) -> any:
    """
    eeg.shape [Batch, Electrode Number, N_Times]
    output [Batch, View, Channel, Length]
    """
    if isinstance(eeg, torch.Tensor):
      eeg = eeg.numpy()
    elif isinstance(eeg, np.ndarray):
      pass
    else:
      raise ValueError("eeg should be a ndarray or torch tensor")
    return np.concatenate([eeg, np.zeros([*eeg.shape[:-1], self.padding])], axis=-1)


ArrayLike = Union[np.ndarray, "torch.Tensor"]


class MultiViewStackingTransform(EEGTransform):
  """
  Stack EEG views along the channel dimension.

  Supports both:
  - [Batch, View, Channel, Length] -> [Batch, View * Channel, Length]
  - [View, Channel, Length] -> [View * Channel, Length]

  Works with numpy.ndarray and torch.Tensor, preserving the input type.
  """

  def __init__(self, freq_bands=FREQ_BANDS, apply_to_baseline: bool = False):
    # apply_to_baseline kept for API compatibility but unused
    super().__init__(apply_to_baseline=False)
    self.freq_bands = freq_bands

  def _is_torch(self, x: Any) -> bool:
    return isinstance(x, torch.Tensor)

  def _validate(self, x: ArrayLike) -> Tuple[int, ...]:
    if x is None:
      raise ValueError("`eeg` is None.")
    ndim = getattr(x, "ndim", None)
    if ndim == 4:
      b, v, c, length = x.shape
      if v <= 0 or c <= 0:
        raise ValueError(
          f"`View` and `Channel` must be > 0, got View={v}, Channel={c}."
        )
      return b, v, c, length
    elif ndim == 3:
      v, c, length = x.shape
      if v <= 0 or c <= 0:
        raise ValueError(
          f"`View` and `Channel` must be > 0, got View={v}, Channel={c}."
        )
      return v, c, length
    else:
      raise ValueError(
        f"Expected shape [Batch, View, Channel, Length] or [View, Channel, Length], got shape={getattr(x, 'shape', None)}"
      )

  def apply(self, eeg: np.ndarray | torch.Tensor, **kwargs) -> ArrayLike:
    """
    Reshape inputs:
    - [B, V, C, L] -> [B, V*C, L]
    - [V, C, L] -> [V*C, L]

    Reshapes without copies when possible.
    Preserves the input dtype in the output.
    """
    if not (isinstance(eeg, np.ndarray) or self._is_torch(eeg)):
      raise ValueError("`eeg` must be a numpy.ndarray or a torch.Tensor.")

    # Store original dtype to ensure output has same dtype as input
    original_dtype = eeg.dtype

    dims = self._validate(eeg)

    if len(dims) == 4:
      # 4D input: [Batch, View, Channel, Length]
      b, v, c, length = dims
      if self._is_torch(eeg):
        if not eeg.is_contiguous():
          eeg = eeg.contiguous()
        result = eeg.reshape(b, v * c, length)
        # Ensure output dtype matches input dtype
        if result.dtype != original_dtype:
          result = result.to(dtype=original_dtype)
        return result
      result = np.reshape(eeg, (b, v * c, length))
      # Ensure output dtype matches input dtype
      if result.dtype != original_dtype:
        result = result.astype(original_dtype)
      return result
    else:
      # 3D input: [View, Channel, Length]
      v, c, length = dims
      if self._is_torch(eeg):
        if not eeg.is_contiguous():
          eeg = eeg.contiguous()
        result = eeg.reshape(v * c, length)
        # Ensure output dtype matches input dtype
        if result.dtype != original_dtype:
          result = result.to(dtype=original_dtype)
        return result
      result = np.reshape(eeg, (v * c, length))
      # Ensure output dtype matches input dtype
      if result.dtype != original_dtype:
        result = result.astype(original_dtype)
      return result


class BatchBandSignal:
  def __init__(
    self, sampling_rate=128, order=5, band_dict=None, apply_to_baseline=False
  ):
    self.band_signal = BandSignal(
      sampling_rate=sampling_rate,
      order=order,
      band_dict=band_dict,
      apply_to_baseline=apply_to_baseline,
    )

  def __call__(self, eeg: np.ndarray) -> Dict[str, np.ndarray]:
    B, C, L = eeg.shape
    band_eeg_batch = []

    for i in range(B):
      band_eeg = self.band_signal(eeg=eeg[i])["eeg"]  # shape [num_bands, C, L]
      band_eeg_batch.append(band_eeg)

    band_eeg_batch = np.stack(band_eeg_batch, axis=0)  # shape [B, num_bands, C, L]

    return {"eeg": band_eeg_batch}
