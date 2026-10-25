"""Shared cache helpers (transform serialization, npz validation)."""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from typing import Optional

import numpy as np
from torcheeg import transforms
from torcheeg.transforms import Compose


def _serialize_value(value: Any) -> Any:
  """Recursively convert transform state to deterministic JSON-compatible data."""
  if isinstance(value, np.generic):
    return _serialize_value(value.item())
  if isinstance(value, np.ndarray):
    return _serialize_value(value.tolist())
  if isinstance(value, (str, int, float, bool, type(None))):
    return value
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, Mapping):
    return {
      str(key): _serialize_value(item)
      for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
    }
  if isinstance(value, (list, tuple)):
    return [_serialize_value(item) for item in value]
  if isinstance(value, Compose) or isinstance(value, transforms.EEGTransform):
    return serialize_transform(value)
  if callable(value):
    return {
      "callable": f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', value.__class__.__qualname__)}"
    }
  return {
    "type": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
    "repr": repr(value),
  }


def serialize_transform(transform: Optional[transforms.EEGTransform]) -> Optional[dict]:
  """Serialize a torcheeg transform (or Compose) without dropping parameters."""
  if transform is None:
    return None

  if isinstance(transform, Compose):
    return {
      "type": "Compose",
      "module": transform.__class__.__module__,
      "transforms": [serialize_transform(t) for t in transform.transforms],
    }

  params: dict = {}
  if hasattr(transform, "__dict__"):
    for key, value in transform.__dict__.items():
      if key.startswith("_") or key == "transforms":
        continue
      params[key] = _serialize_value(value)

  return {
    "type": transform.__class__.__name__,
    "module": transform.__class__.__module__,
    "params": params,
  }


def validate_npz_cache(cache_file: str, required_keys: tuple = ("data",)) -> bool:
  try:
    if not os.path.exists(cache_file):
      return False
    arr = np.load(cache_file)
    for k in required_keys:
      if k not in arr:
        return False
    if arr["data"].ndim < 2:
      return False
    return True
  except Exception as e:
    print(f"Warning: Cache file {cache_file} is corrupted: {e}")
    return False
