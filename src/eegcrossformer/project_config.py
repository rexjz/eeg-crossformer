"""Portable paths and cohort definitions for the five public protocols.

Configure paths through environment variables or a ``.env`` in the current
working directory. Importing this module does not create folders or require
the datasets to be present.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)


def _path_from_env(name: str, default: Path, legacy_name: str | None = None) -> Path:
  value = os.getenv(name)
  if not value and legacy_name:
    value = os.getenv(legacy_name)
  return Path(value or default).expanduser().resolve()


DATA_DIR = _path_from_env("DATA_DIR", Path.cwd() / "data")
OUTPUT_DIR = _path_from_env("OUTPUT_DIR", Path.cwd() / "outputs")
BCIC34A_ROOT = _path_from_env(
  "BCIC34A_ROOT", DATA_DIR / "BCIC34a", "bcic3_dataset4a_path"
)
SEEDV_ROOT = _path_from_env("SEEDV_ROOT", DATA_DIR / "SEED-V", "seedV_eeg_path")
OPENBMI_CACHE_ROOT = _path_from_env(
  "OPENBMI_CACHE_ROOT", DATA_DIR / "OpenBMI", "openbmi_path"
)

# Preserve internal framework names while exposing descriptive public settings.
PROJECT_PATH = Path(__file__).resolve().parent
LOG_PATH = OUTPUT_DIR
bcic3_dataset4a_path = str(BCIC34A_ROOT)
seedV_eeg_path = str(SEEDV_ROOT)
openbmi_path = str(OPENBMI_CACHE_ROOT)

bcic3_dataset4a_subjects_code = ["aa", "al", "av", "aw", "ay"]
seedv_subjects_code = [str(subject) for subject in range(1, 17)]
openbmi_subjects = [str(subject) for subject in range(1, 55)]
