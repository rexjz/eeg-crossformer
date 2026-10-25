"""Base experiment for pooled-two-session OpenBMI PaperMOABB KFold."""

from eegcrossformer.data.dataset.OpenBMI.mi.OpenBMIMIDataset import (
  OpenBMI_MIPaperMOABBPooledSessionsKfoldDataModule,
)
from eegcrossformer.experiments.by_dataset.openbmi_mi.kfold._paper_moabb_all_channels import (
  PaperMOABBAllChannelsOpenBMIMIKfoldExperiment,
)


POOLED_SESSION_KEY = "pooled"
POOLED_SESSIONS = ("1", "2")


class OpenBMIPaperMOABBPooledSessionKfoldExperiment(
  PaperMOABBAllChannelsOpenBMIMIKfoldExperiment
):
  """Model-agnostic base for subject-dependent KFold over both sessions."""

  include_session_id = False

  def __init__(self, *args, sessions=None, **kwargs):
    if sessions is not None and tuple(sessions) != POOLED_SESSIONS:
      raise ValueError("The pooled-session protocol requires sessions ('1', '2')")
    super().__init__(*args, sessions=[POOLED_SESSION_KEY], **kwargs)

  def get_protocol_settings(self):
    settings = super().get_protocol_settings()
    window_start, window_stop = self.event_window_seconds
    settings.update(
      {
        "protocol": (
          f"{self.model_display_name} paper OpenBMI pooled-two-session "
          "subject-dependent KFold, all 62 EEG channels"
        ),
        "preprocessing": (
          "each session independently: raw -> pick EEG -> uV -> 0-40Hz -> "
          "continuous exponential standardization -> cue-aligned event windows -> "
          f"fixed [{window_start:g},{window_stop:g})s -> "
          "resample 250Hz -> all 62 EEG channels; then pool sessions 1 and 2"
        ),
        "sessions": list(POOLED_SESSIONS),
        "session_handling": "preprocess independently, then pool before splitting",
        "split_stratification": "joint (session, class)",
        "split_semantics": self.get_split_semantics(),
        "training_seed_formula": (
          "base + (subject_id - 1) * kfold + fold - 1"
        ),
      }
    )
    return settings

  def get_split_semantics(self) -> str:
    prefix = (
      f"outer subject-dependent {self.kfold}-fold over pooled sessions 1 and 2, "
      "jointly stratified by session and class"
    )
    if self.has_fixed_budget():
      if self.diagnostic_validation_as_test:
        return (
          f"{prefix}; train on all outer-training trials for a fixed training "
          "budget; reuse the outer test fold as validation diagnostics "
          "without checkpoint selection; test final weights once"
        )
      return (
        f"{prefix}; train on all outer-training trials for a fixed training "
        "budget with validation disabled; test the untouched outer fold once"
      )
    if self.two_stage_retrain:
      return (
        f"{prefix}; stage 1 uses a joint-stratified validation split from outer-"
        "training trials to select optimizer steps; stage 2 retrains from scratch "
        "on all outer-training trials and tests the untouched outer fold once"
      )
    return (
      f"{prefix}; use a joint-stratified validation split from outer-training "
      "trials for checkpoint selection"
    )

  def get_datamodule(
    self,
    subject_id,
    session,
    fold,
    num_workers=0,
    separate_validation: bool | None = None,
  ):
    if session != POOLED_SESSION_KEY:
      raise ValueError(f"Expected pooled session key, got {session!r}")
    if separate_validation is None:
      separate_validation = self.separate_validation
    return OpenBMI_MIPaperMOABBPooledSessionsKfoldDataModule(
      batch_size=self.batch_size,
      subject_id=subject_id,
      fold=fold,
      kfold=self.kfold,
      get_item_style="tuple",
      num_workers=num_workers,
      channels="all",
      offline_transform_profile=self.offline_transform_profile,
      random_seed=self.split_random_seed,
      separate_validation=separate_validation,
      validation_fraction=self.validation_fraction,
      include_session_id=self.include_session_id,
      event_window_seconds=self.event_window_seconds,
    )

  def get_fold_training_seed(self, subject_id: str, session: str, fold: int) -> int:
    if session != POOLED_SESSION_KEY:
      raise ValueError(f"Expected pooled session key, got {session!r}")
    return self.training_seed + (int(subject_id) - 1) * self.kfold + fold - 1

  def get_fold_version(self, subject_id: str, session: str, fold: int) -> str:
    if session != POOLED_SESSION_KEY:
      raise ValueError(f"Expected pooled session key, got {session!r}")
    return f"subject{subject_id.zfill(2)}_sessions01-02_fold{fold}"
