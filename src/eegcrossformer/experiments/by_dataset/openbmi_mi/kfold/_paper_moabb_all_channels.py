"""Shared OpenBMI paper-MOABB all-channel k-fold protocol glue."""

from argparse import ArgumentParser, BooleanOptionalAction
from datetime import datetime
import json
import os
from typing import Callable

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
import torch

from eegcrossformer.data.dataset.OpenBMI.mi.OpenBMIMIDataset import (
  OpenBMI_MIPaperMOABBSession1KfoldDataModule,
  OpenBMI_MIPaperMOABBSession2KfoldDataModule,
)
from eegcrossformer.data.dataset.OpenBMI.mi.paper_moabb_transform_cache import (
  DEFAULT_EVENT_WINDOW_SECONDS,
  transform_cache_settings,
)
from eegcrossformer.experiments.by_dataset.openbmi_mi.kfold.OpenBMIMIKfoldExperiment import (
  OpenBMIMIKfoldExperiment,
)
from eegcrossformer.experiments.task_settings.mi_task import get_mi_classification_task
from eegcrossformer.experiments.utils.parse_input_devices import parse_input_devices_arg
from eegcrossformer.experiments.utils.summary_logs import generate_summary
from eegcrossformer.models.BaseLightingModelWrapper import LightningModelWrapper
from eegcrossformer.project_config import LOG_PATH, openbmi_subjects


N_TIME = 750
N_CHAN = 62
NUM_CLASSES = 2


def parse_sessions_arg(value: str) -> list[str]:
  value = value.strip().lower()
  if value in {"both", "all", "1,2", "2,1"}:
    return ["1", "2"]
  if value in {"1", "session1", "session_1"}:
    return ["1"]
  if value in {"2", "session2", "session_2"}:
    return ["2"]
  raise ValueError("--sessions must be one of: 1, 2, both")


def parse_subjects_arg(value: str) -> list[str]:
  value = value.strip().lower()
  if value == "all":
    return openbmi_subjects
  if value == "first10":
    return [str(i) for i in range(1, 11)]
  subjects = [part.strip() for part in value.split(",") if part.strip()]
  if not subjects:
    raise ValueError("--subjects must be 'all', 'first10', or comma-separated ids")
  return subjects


def parse_folds_arg(value: str) -> list[int] | None:
  value = value.strip().lower()
  if value == "all":
    return None
  folds = [int(part.strip()) for part in value.split(",") if part.strip()]
  if not folds:
    raise ValueError("--folds must be 'all' or comma-separated fold ids")
  invalid = [fold for fold in folds if fold < 1]
  if invalid:
    raise ValueError("--folds values must be positive")
  return folds


class PaperMOABBAllChannelsOpenBMIMIKfoldExperiment(OpenBMIMIKfoldExperiment):
  model_name: str = ""
  model_display_name: str = ""
  model_factory: Callable[..., object] | None = None
  model_kwargs: dict = {}
  input_shape: list[int] = [1, N_CHAN, N_TIME]
  adapter_note: str = "none"
  lr: float = 1e-3
  weight_decay: float = 0.001
  offline_transform_profile: str | None = None
  event_window_seconds: tuple[float, float] = DEFAULT_EVENT_WINDOW_SECONDS
  split_random_seed: int = 42
  separate_validation: bool = True
  validation_fraction: float = 0.2
  training_seed: int = 42
  selection_monitor: str = "val_accuracy"

  def __init__(
    self,
    *args,
    sessions: list[str] | None = None,
    folds: list[int] | None = None,
    training_seed: int | None = None,
    two_stage_retrain: bool = False,
    fixed_training_steps: int | None = None,
    fixed_training_epochs: int | None = None,
    diagnostic_validation_as_test: bool = False,
    **kwargs,
  ):
    super().__init__(*args, **kwargs)
    for name, value in (("fixed_training_steps", fixed_training_steps),
                        ("fixed_training_epochs", fixed_training_epochs)):
      if value is not None and value < 1:
        raise ValueError(f"{name} must be positive when provided")
    if fixed_training_steps is not None and fixed_training_epochs is not None:
      raise ValueError(
        "fixed_training_steps and fixed_training_epochs are mutually exclusive"
      )
    fixed_budget = fixed_training_steps is not None or fixed_training_epochs is not None
    if two_stage_retrain and fixed_budget:
      raise ValueError(
        "two_stage_retrain and a fixed training budget are mutually exclusive"
      )
    if diagnostic_validation_as_test and not fixed_budget:
      raise ValueError(
        "diagnostic_validation_as_test requires a fixed training budget"
      )
    self.sessions = sessions or ["1", "2"]
    self.folds = folds
    if training_seed is not None:
      if training_seed < 0:
        raise ValueError("training_seed must be non-negative")
      self.training_seed = int(training_seed)
    self.two_stage_retrain = two_stage_retrain
    self.fixed_training_steps = fixed_training_steps
    self.fixed_training_epochs = fixed_training_epochs
    self.diagnostic_validation_as_test = diagnostic_validation_as_test

  def get_model_name(self) -> str:
    return self.model_name

  def get_protocol_settings(self):
    window_start, window_stop = self.event_window_seconds
    return {
      "protocol": (
        f"{self.model_display_name} paper OpenBMI within-session: "
        "MOABB raw-continuous session-specific 5-fold, all 62 EEG channels"
      ),
      "data_source": "MOABBDataset Lee2019_MI raw continuous",
      "preprocessing": (
        "raw -> pick EEG -> uV -> 0-40Hz -> continuous exponential "
        f"standardization -> cue-aligned event windows -> selected session -> "
        f"fixed [{window_start:g},{window_stop:g})s -> "
        "resample 250Hz -> all 62 EEG channels"
      ),
      "event_anchor": "MOABB Lee2019_MI cue onset",
      "event_window_seconds": [window_start, window_stop],
      "temporal_crop": f"fixed cue-aligned [{window_start:g},{window_stop:g}) s",
      "augmentation": "none",
      "channels": "all",
      "split_semantics": self.get_split_semantics(),
      "outer_test_folds": self.kfold,
      "validation_fraction_of_outer_train": self.validation_fraction,
      "split_random_seed": self.split_random_seed,
      "two_stage_retrain": self.two_stage_retrain,
      "fixed_training_steps": self.fixed_training_steps,
      "fixed_training_epochs": self.fixed_training_epochs,
      "diagnostic_validation_as_test": self.diagnostic_validation_as_test,
      "protocol_adapter": self.adapter_note,
      "batch_loading": "batch-aware direct source-to-contiguous-tensor collation",
    }

  def has_fixed_budget(self) -> bool:
    return self.fixed_training_steps is not None or self.fixed_training_epochs is not None

  def fixed_budget_description(self) -> str:
    if self.fixed_training_steps is not None:
      return f"fixed {self.fixed_training_steps} optimizer steps"
    return f"fixed {self.fixed_training_epochs} epochs"

  def get_split_semantics(self) -> str:
    if self.has_fixed_budget():
      if self.diagnostic_validation_as_test:
        return (
          "outer session-specific 5-fold; train on all outer-training trials "
          "for a fixed training budget; reuse the outer test fold as "
          "validation diagnostics without checkpoint selection; test final "
          "weights once"
        )
      return (
        "outer session-specific 5-fold test; train on all outer-training "
        "trials for a fixed training budget with validation disabled; "
        "test the untouched outer fold once"
      )
    if self.two_stage_retrain:
      return (
        "outer session-specific 5-fold test; stage 1 uses a stratified "
        "validation split from outer-training trials to select optimizer steps; "
        "stage 2 retrains from scratch on all outer-training trials and tests "
        "the untouched outer fold once"
      )
    return (
      "outer session-specific 5-fold test; stratified validation split from "
      "outer-training trials"
    )

  def get_datamodule(
    self,
    subject_id,
    session,
    fold,
    num_workers=0,
    separate_validation: bool | None = None,
  ):
    datamodule_cls = {
      "1": OpenBMI_MIPaperMOABBSession1KfoldDataModule,
      "2": OpenBMI_MIPaperMOABBSession2KfoldDataModule,
    }[session]
    if separate_validation is None:
      separate_validation = self.separate_validation
    return datamodule_cls(
      batch_size=self.batch_size,
      subject_id=subject_id,
      fold=fold,
      kfold=self.kfold,
      get_item_style="tuple",
      num_workers=num_workers,
      channels="all",
      offline_transform_profile=self.offline_transform_profile,
      event_window_seconds=self.event_window_seconds,
      random_seed=self.split_random_seed,
      separate_validation=separate_validation,
      validation_fraction=self.validation_fraction,
    )

  def get_fold_training_seed(self, subject_id: str, session: str, fold: int) -> int:
    return (
      self.training_seed
      + (int(subject_id) - 1) * 10
      + (int(session) - 1) * self.kfold
      + fold
      - 1
    )

  def get_fold_version(self, subject_id: str, session: str, fold: int) -> str:
    return f"subject{subject_id.zfill(2)}_session{session.zfill(2)}_fold{fold}"

  def create_model(self, model_kwargs: dict | None = None):
    model_factory = self.__class__.model_factory
    if model_factory is None:
      raise ValueError("model_factory must be set by subclasses")
    input_model_kwargs = dict(self.model_kwargs)
    if model_kwargs is not None:
      input_model_kwargs.update(model_kwargs)

    model = model_factory(**input_model_kwargs)
    return LightningModelWrapper(
      model=model,
      lr=self.lr,
      weight_decay=self.weight_decay,
      task=get_mi_classification_task(
        num_classes=NUM_CLASSES,
        reg_batch=lambda batch: (batch[0], batch[1], {}),
      ),
    )

  def get_selection_callbacks(self) -> tuple[list[object], ModelCheckpoint]:
    callbacks: list[object] = []
    if self.early_stop:
      callbacks.append(
        EarlyStopping(
          monitor=self.selection_monitor,
          min_delta=0.0001,
          patience=self.patience,
          verbose=False,
          mode="max",
        )
      )
    checkpoint_callback = ModelCheckpoint(
      save_top_k=1,
      monitor=self.selection_monitor,
      mode="max",
      filename=self.checkpoint_name,
    )
    callbacks.append(checkpoint_callback)
    return callbacks, checkpoint_callback

  def get_retrain_callbacks(self) -> list[object]:
    return []

  def get_fixed_training_metadata(self) -> dict:
    return {}

  @staticmethod
  def read_selected_training_budget(checkpoint_path: str) -> tuple[int, int]:
    checkpoint = torch.load(
      checkpoint_path,
      map_location="cpu",
      weights_only=False,
    )
    selected_epoch = int(checkpoint["epoch"]) + 1
    selected_steps = int(checkpoint["global_step"])
    if selected_epoch < 1 or selected_steps < 1:
      raise ValueError(
        "Stage-1 checkpoint must contain a positive epoch and optimizer-step budget"
      )
    return selected_epoch, selected_steps

  def run_selection_stage(
    self,
    *,
    subject_code: str,
    session: str,
    fold: int,
    version: str,
    logger_name: str,
    model_kwargs: dict,
    num_workers: int,
    devices,
    precision: str,
  ) -> tuple[int, int, float, object]:
    model = self.create_model(model_kwargs=model_kwargs)
    datamodule = self.get_datamodule(
      subject_id=subject_code,
      session=session,
      fold=fold,
      num_workers=num_workers,
      separate_validation=True,
    )
    logger = CSVLogger(
      save_dir=self.log_base_dir,
      name=logger_name,
      version=os.path.join("_selection", version),
    )
    callbacks, checkpoint_callback = self.get_selection_callbacks()
    trainer = L.Trainer(
      max_epochs=self.max_epoches,
      max_steps=getattr(self, "training_max_steps", -1),
      accelerator=getattr(self, "accelerator", "gpu"),
      devices=devices,
      logger=logger,
      callbacks=callbacks,
      precision=precision,
      deterministic="warn",
    )
    trainer.fit(model=model, datamodule=datamodule)
    selected_epoch, selected_steps = self.read_selected_training_budget(
      checkpoint_callback.best_model_path
    )
    best_score = float(checkpoint_callback.best_model_score.detach().cpu())
    return selected_epoch, selected_steps, best_score, datamodule

  def run_retrain_stage(
    self,
    *,
    subject_code: str,
    session: str,
    fold: int,
    version: str,
    logger_name: str,
    model_kwargs: dict,
    num_workers: int,
    devices,
    precision: str,
    selected_steps: int,
    diagnostic_validation_as_test: bool = False,
  ) -> tuple[object, str]:
    model = self.create_model(model_kwargs=model_kwargs)
    datamodule = self.get_datamodule(
      subject_id=subject_code,
      session=session,
      fold=fold,
      num_workers=num_workers,
      separate_validation=False,
    )
    logger = CSVLogger(
      save_dir=self.log_base_dir,
      name=logger_name,
      version=version,
    )
    retrain_callbacks = self.get_retrain_callbacks()
    trainer = L.Trainer(
      max_epochs=self.max_epoches,
      max_steps=selected_steps,
      accelerator=getattr(self, "accelerator", "gpu"),
      devices=devices,
      logger=logger,
      callbacks=retrain_callbacks,
      precision=precision,
      deterministic="warn",
      enable_checkpointing=False,
      limit_val_batches=1.0 if diagnostic_validation_as_test else 0,
      num_sanity_val_steps=0,
    )
    trainer.fit(model=model, datamodule=datamodule)
    triggered_guards = [
      callback
      for callback in retrain_callbacks
      if getattr(callback, "stopped_for_nonfinite", False)
    ]
    failed_recoveries = [
      callback
      for callback in triggered_guards
      if not getattr(callback, "recovery_applied", False)
    ]
    if failed_recoveries:
      raise RuntimeError("Non-finite training was detected but recovery failed")
    if selected_steps > 0 and trainer.global_step != selected_steps and not triggered_guards:
      raise RuntimeError(
        f"Stage 2 stopped at global step {trainer.global_step}; "
        f"expected {selected_steps}"
      )
    if selected_steps < 0 and trainer.current_epoch != self.max_epoches and not triggered_guards:
      raise RuntimeError(
        f"Stage 2 stopped after {trainer.current_epoch} epochs; "
        f"expected {self.max_epoches}"
      )
    if triggered_guards:
      print(
        f"Stage 2 stopped at global step {trainer.global_step} after a "
        "non-finite training signal and restored its last finite checkpoint"
      )

    checkpoint_dir = os.path.join(logger.log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    final_checkpoint_path = os.path.join(checkpoint_dir, "final.ckpt")
    trainer.save_checkpoint(final_checkpoint_path)
    trainer.test(
      model=model,
      datamodule=datamodule,
      ckpt_path=final_checkpoint_path,
    )
    return datamodule, final_checkpoint_path

  def write_two_stage_metadata(
    self,
    *,
    logger_name: str,
    version: str,
    selected_epoch: int,
    selected_steps: int,
    best_score: float,
    selection_datamodule,
    retrain_datamodule,
    final_checkpoint_path: str,
  ) -> None:
    metadata = {
      "selection_monitor": self.selection_monitor,
      "selected_epoch": selected_epoch,
      "selected_optimizer_steps": selected_steps,
      "selection_best_score": best_score,
      "selection_train_trials": len(selection_datamodule.train_indices),
      "selection_validation_trials": len(selection_datamodule.val_indices),
      "outer_test_trials": len(selection_datamodule.test_indices),
      "retrain_trials": len(retrain_datamodule.train_indices),
      "stage2_validation_batches": 0,
      "final_checkpoint": final_checkpoint_path,
    }
    metadata_path = os.path.join(
      self.log_base_dir,
      logger_name,
      version,
      "two_stage_selection.json",
    )
    with open(metadata_path, "w") as metadata_file:
      json.dump(metadata, metadata_file, indent=2)

  def write_fixed_training_metadata(
    self,
    *,
    logger_name: str,
    version: str,
    datamodule,
    final_checkpoint_path: str,
  ) -> None:
    metadata = {
      "fixed_optimizer_steps": self.fixed_training_steps,
      "fixed_epochs": self.fixed_training_epochs,
      "training_trials": len(datamodule.train_indices),
      "outer_test_trials": len(datamodule.test_indices),
      "diagnostic_validation_as_test": self.diagnostic_validation_as_test,
      "diagnostic_validation_trials": (
        len(datamodule.val_indices)
        if self.diagnostic_validation_as_test
        else 0
      ),
      "checkpoint_selection": "none_final_weights",
      "outer_test_evaluations": 1,
      "final_checkpoint": final_checkpoint_path,
    }
    metadata.update(self.get_fixed_training_metadata())
    metadata_path = os.path.join(
      self.log_base_dir,
      logger_name,
      version,
      "fixed_training.json",
    )
    with open(metadata_path, "w") as metadata_file:
      json.dump(metadata, metadata_file, indent=2)

  def run(
    self,
    precision="32-true",
    run_tag=None,
    devices=1,
    num_workers=0,
    model_kwargs: dict | None = None,
  ):
    model_kwargs = model_kwargs or {}
    fold_values = self.folds if self.folds is not None else list(range(1, self.kfold + 1))
    prefix = f"{run_tag}_" if run_tag is not None else ""
    logger_name = f"{prefix}{self.experiment_tag}"
    fixed_training = self.has_fixed_budget()
    stage2_training = self.two_stage_retrain or fixed_training
    diagnostic_validation = (
      fixed_training and self.diagnostic_validation_as_test
    )
    experiment_settings = {
      "accelerator": getattr(self, "accelerator", "gpu"),
      "max_steps": getattr(self, "fixed_training_steps", None) or getattr(self, "training_max_steps", -1),
      "training_seed": getattr(self, "training_seed", None),
      "precision": precision,
      "devices": devices,
      "num_workers": num_workers,
      "model_kwargs_override": model_kwargs,
      "sessions": self.sessions,
      "folds": fold_values,
      "heldout_fold_role": (
        "diagnostic_validation_and_final_test"
        if diagnostic_validation
        else "test_only"
      ),
      "validation_source": (
        "outer test fold (diagnostics only; never used for selection)"
        if diagnostic_validation
        else None
        if fixed_training
        else "stratified split from outer-training trials"
      ),
      "training_mode": (
        "fixed_steps"
        if fixed_training
        else "two_stage_retrain"
        if self.two_stage_retrain
        else "separate_validation"
      ),
      "two_stage_retrain": self.two_stage_retrain,
      "fixed_training_steps": self.fixed_training_steps,
      "fixed_training_epochs": self.fixed_training_epochs,
      "diagnostic_validation_as_test": diagnostic_validation,
      "stage1_selection_monitor": self.selection_monitor,
      "stage2_training_data": (
        "all outer-training trials" if stage2_training else None
      ),
      "stage2_training_budget": (
        "exact optimizer global_step from stage-1 best checkpoint"
        if self.two_stage_retrain
        else self.fixed_budget_description()
        if fixed_training
        else None
      ),
      "stage2_validation_batches": (
        "enabled_diagnostics_only"
        if diagnostic_validation
        else 0
        if stage2_training
        else None
      ),
      "outer_test_evaluations_per_fold": 1,
      "batch_size": self.batch_size,
      "lr": self.lr,
      "weight_decay": self.weight_decay,
      "max_epoches": self.max_epoches,
      "early_stop": self.early_stop if not fixed_training else False,
      "patience": self.patience,
      "kfold": self.kfold,
      "model": self.model_display_name,
      "input_shape": self.input_shape,
      "training_seed_base": self.training_seed,
      "training_seed_formula": (
        "base + (subject_id - 1) * 10 + (session - 1) * kfold + fold - 1"
      ),
      "deterministic": "warn",
    }
    experiment_settings.update(self.get_protocol_settings())
    experiment_settings.update(
      transform_cache_settings(
        self.offline_transform_profile,
        channels="all",
        event_window_seconds=self.event_window_seconds,
      )
    )
    file_path = os.path.join(self.log_base_dir, logger_name, "experiment_settings.json")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as json_file:
      json.dump(experiment_settings, json_file, indent=4)

    experiment_start_time = datetime.now()
    total_folds = len(self.subjects) * len(self.sessions) * len(fold_values)
    completed_folds = 0
    fold_times = []

    for subject_code in self.subjects:
      for session in self.sessions:
        for fold in fold_values:
          fold_start_time = datetime.now()
          fold_training_seed = self.get_fold_training_seed(
            subject_code, session, fold
          )
          L.seed_everything(fold_training_seed, workers=True)
          print(
            f"[{fold_start_time.strftime('%Y-%m-%d %H:%M:%S')}] "
            f"Starting subject {subject_code}, session {session}, fold {fold} "
            f"({completed_folds + 1}/{total_folds}), seed {fold_training_seed}"
          )

          version = self.get_fold_version(subject_code, session, fold)
          if self.two_stage_retrain:
            (
              selected_epoch,
              selected_steps,
              best_score,
              selection_datamodule,
            ) = self.run_selection_stage(
              subject_code=subject_code,
              session=session,
              fold=fold,
              version=version,
              logger_name=logger_name,
              model_kwargs=model_kwargs,
              num_workers=num_workers,
              devices=devices,
              precision=precision,
            )
            print(
              f"Stage 1 selected epoch {selected_epoch}, "
              f"optimizer steps {selected_steps}, "
              f"{self.selection_monitor} {best_score:.4f}"
            )
            L.seed_everything(fold_training_seed, workers=True)
            retrain_datamodule, final_checkpoint_path = self.run_retrain_stage(
              subject_code=subject_code,
              session=session,
              fold=fold,
              version=version,
              logger_name=logger_name,
              model_kwargs=model_kwargs,
              num_workers=num_workers,
              devices=devices,
              precision=precision,
              selected_steps=selected_steps,
            )
            self.write_two_stage_metadata(
              logger_name=logger_name,
              version=version,
              selected_epoch=selected_epoch,
              selected_steps=selected_steps,
              best_score=best_score,
              selection_datamodule=selection_datamodule,
              retrain_datamodule=retrain_datamodule,
              final_checkpoint_path=final_checkpoint_path,
            )
          elif fixed_training:
            retrain_datamodule, final_checkpoint_path = self.run_retrain_stage(
              subject_code=subject_code,
              session=session,
              fold=fold,
              version=version,
              logger_name=logger_name,
              model_kwargs=model_kwargs,
              num_workers=num_workers,
              devices=devices,
              precision=precision,
              selected_steps=self.fixed_training_steps or -1,
              diagnostic_validation_as_test=diagnostic_validation,
            )
            self.write_fixed_training_metadata(
              logger_name=logger_name,
              version=version,
              datamodule=retrain_datamodule,
              final_checkpoint_path=final_checkpoint_path,
            )
          else:
            model = self.create_model(model_kwargs=model_kwargs)
            datamodule = self.get_datamodule(
              subject_id=subject_code,
              session=session,
              fold=fold,
              num_workers=num_workers,
            )
            logger = CSVLogger(
              save_dir=self.log_base_dir, name=logger_name, version=version
            )
            callbacks, _ = self.get_selection_callbacks()
            trainer = L.Trainer(
              max_epochs=self.max_epoches,
              max_steps=getattr(self, "training_max_steps", -1),
              accelerator=getattr(self, "accelerator", "gpu"),
              devices=devices,
              logger=logger,
              callbacks=callbacks,
              precision=precision,
              deterministic="warn",
            )
            trainer.fit(model=model, datamodule=datamodule)
            trainer.test(model=model, datamodule=datamodule, ckpt_path="best")

          fold_end_time = datetime.now()
          fold_duration = (fold_end_time - fold_start_time).total_seconds()
          fold_times.append(fold_duration)
          completed_folds += 1

          avg_fold_time = sum(fold_times) / len(fold_times)
          remaining_folds = total_folds - completed_folds
          estimated_remaining_seconds = avg_fold_time * remaining_folds
          hours, remainder = divmod(int(estimated_remaining_seconds), 3600)
          minutes, seconds = divmod(remainder, 60)
          elapsed_time = (fold_end_time - experiment_start_time).total_seconds()
          elapsed_hours, elapsed_remainder = divmod(int(elapsed_time), 3600)
          elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)

          print(
            f"[{fold_end_time.strftime('%Y-%m-%d %H:%M:%S')}] "
            f"Completed fold in {fold_duration:.1f}s | "
            f"Elapsed: {elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_seconds:02d} | "
            f"Avg fold time: {avg_fold_time:.1f}s | "
            f"Est. remaining: {hours:02d}:{minutes:02d}:{seconds:02d}"
          )
          print("-" * 80)

    log_root_path = os.path.join(self.log_base_dir, logger_name)
    summary_pd = generate_summary(log_root_path)
    summary_pd.to_csv(os.path.join(log_root_path, "summary.csv"))
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(summary_pd.to_string())
    print("=" * 80)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Experiment completed")


def add_common_paper_moabb_args(parser: ArgumentParser, default_batch_size: int):
  parser.add_argument("--batch_size", default=default_batch_size, type=int)
  parser.add_argument("--max_epoches", default=200, type=int)
  parser.add_argument("--patience", default=20, type=int)
  parser.add_argument("--devices", default="1")
  parser.add_argument("--precision", default="32-true")
  parser.add_argument("--num_workers", default=0, type=int)
  parser.add_argument("--sessions", default="both")
  parser.add_argument("--subjects", default="all")
  parser.add_argument("--folds", default="all")
  parser.add_argument(
    "--training_seed",
    default=42,
    type=int,
    help=(
      "base training seed; deterministic per-unit offsets are added by the "
      "protocol"
    ),
  )
  parser.add_argument(
    "--run_tag",
    default=None,
    help="optional logger prefix used to isolate repeated-seed runs",
  )
  parser.add_argument("--smoke_test", action="store_true")
  parser.add_argument("--no_early_stop", action="store_true")
  parser.add_argument("--two_stage_retrain", action="store_true")
  parser.add_argument(
    "--fixed_training_steps",
    default=0,
    type=int,
    help="positive value trains on the full outer-training split for a fixed step budget",
  )
  parser.add_argument(
    "--diagnostic_validation_as_test",
    action=BooleanOptionalAction,
    default=False,
    help="log the outer test fold as validation diagnostics in fixed-step mode",
  )


def run_common_paper_moabb_cli(args, experiment_cls, experiment_kwargs=None):
  experiment_kwargs = experiment_kwargs or {}
  fixed_training_steps = args.fixed_training_steps or None
  if args.two_stage_retrain:
    fixed_training_steps = None
  elif args.smoke_test and fixed_training_steps is not None:
    fixed_training_steps = 5
  diagnostic_validation_as_test = (
    args.diagnostic_validation_as_test and fixed_training_steps is not None
  )
  experiment = experiment_cls(
    batch_size=16 if args.smoke_test else args.batch_size,
    max_epoches=1 if args.smoke_test else args.max_epoches,
    early_stop=False if args.smoke_test else not args.no_early_stop,
    patience=3 if args.smoke_test else args.patience,
    sessions=parse_sessions_arg(args.sessions),
    subjects=["1"] if args.smoke_test else parse_subjects_arg(args.subjects),
    folds=[1] if args.smoke_test else parse_folds_arg(args.folds),
    training_seed=args.training_seed,
    two_stage_retrain=args.two_stage_retrain,
    fixed_training_steps=fixed_training_steps,
    diagnostic_validation_as_test=diagnostic_validation_as_test,
    **experiment_kwargs,
  )
  experiment.log_base_dir = LOG_PATH
  experiment.run(
    devices=parse_input_devices_arg(args.devices),
    precision=args.precision,
    num_workers=args.num_workers,
    run_tag="smoke" if args.smoke_test else args.run_tag,
  )
