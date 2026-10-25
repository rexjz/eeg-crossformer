"""Shared OpenBMI paper-MOABB all-channel LOSO protocol glue."""

from argparse import ArgumentParser
from datetime import datetime
import json
import os
from typing import Callable

import lightning as L
import pandas as pd
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from eegcrossformer.data.dataset.OpenBMI.mi.OpenBMIMILOSODataModule import (
  OpenBMI_MIPaperMOABBLOSODataModule,
)
from eegcrossformer.data.dataset.OpenBMI.mi.paper_moabb_transform_cache import (
  DEFAULT_EVENT_WINDOW_SECONDS,
  transform_cache_settings,
)
from eegcrossformer.experiments.by_dataset.openbmi_mi.loso.OpenBMIMILOSOExperiment import (
  OpenBMIMILOSOExperiment,
  openbmi_subjects,
)
from eegcrossformer.experiments.task_settings.mi_task import get_mi_classification_task
from eegcrossformer.experiments.utils.parse_input_devices import parse_input_devices_arg
from eegcrossformer.experiments.utils.summary_logs import (
  generate_summary_accross_versions,
)
from eegcrossformer.models.BaseLightingModelWrapper import LightningModelWrapper
from eegcrossformer.project_config import LOG_PATH
from eegcrossformer.utils.metric_plot import plot_metrics_from_csv


N_TIME = 750
N_CHAN = 62
NUM_CLASSES = 2


def parse_sessions_arg(value: str) -> tuple[str, ...]:
  value = value.strip().lower()
  if value in {"both", "all", "1,2", "2,1"}:
    return ("1", "2")
  if value in {"1", "session1", "session_1"}:
    return ("1",)
  if value in {"2", "session2", "session_2"}:
    return ("2",)
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


class PaperMOABBAllChannelsOpenBMIMILOSOExperiment(OpenBMIMILOSOExperiment):
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

  def __init__(
    self,
    *args,
    num_workers: int = 0,
    sessions: tuple[str, ...] = ("1", "2"),
    source_subjects: list[str] | None = None,
    training_seed: int | None = None,
    deterministic=False,
    **kwargs,
  ):
    super().__init__(*args, **kwargs)
    self.num_workers = num_workers
    self.sessions = sessions
    self.source_subjects = source_subjects
    self.training_seed = training_seed
    self.deterministic = deterministic

  def get_subject_training_seed(self, subject: str) -> int | None:
    """Return an order-independent seed for one left-out-subject fold."""

    if self.training_seed is None:
      return None
    return self.training_seed + int(subject) - 1

  def get_model_name(self) -> str:
    return self.model_name

  def get_protocol_settings(self):
    window_start, window_stop = self.event_window_seconds
    return {
      "protocol": (
        f"{self.model_display_name} paper OpenBMI LOSO: "
        "MOABB raw-continuous selected sessions, all 62 EEG channels"
      ),
      "data_source": "MOABBDataset Lee2019_MI raw continuous",
      "preprocessing": (
        "raw -> pick EEG -> uV -> 0-40Hz -> continuous exponential "
        "standardization -> cue-aligned event windows -> selected sessions -> "
        f"fixed [{window_start:g},{window_stop:g})s -> "
        "resample 250Hz -> all 62 EEG channels"
      ),
      "event_anchor": "MOABB Lee2019_MI cue onset",
      "event_window_seconds": [window_start, window_stop],
      "temporal_crop": f"fixed cue-aligned [{window_start:g},{window_stop:g}) s",
      "sessions": list(self.sessions),
      "channels": "all",
      "loso_split": "leftout subject test; deterministic source-subject validation split",
      "augmentation": "none",
      "protocol_adapter": self.adapter_note,
      "batch_loading": "batch-aware direct source-to-contiguous-tensor collation",
    }

  def get_datamodule(self, subject_id):
    return OpenBMI_MIPaperMOABBLOSODataModule(
      leftout_subject_id=subject_id,
      num_val_subjects=self.num_val_subjects,
      batch_size=self.batch_size,
      random_seed=self.random_seed,
      num_workers=self.num_workers,
      sessions=self.sessions,
      channels="all",
      get_item_style="tuple",
      source_subjects=self.source_subjects,
      offline_transform_profile=self.offline_transform_profile,
      event_window_seconds=self.event_window_seconds,
    )

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

  def run(self, precision="32-true", run_tag=None, devices=1, model_kwargs: dict = {}):
    prefix = f"{run_tag}_" if run_tag is not None else ""
    logger_name = f"{prefix}{self.experiment_tag}"
    experiment_settings = {
      "accelerator": getattr(self, "accelerator", "gpu"),
      "max_steps": getattr(self, "fixed_training_steps", None) or getattr(self, "training_max_steps", -1),
      "precision": precision,
      "devices": devices,
      "model_kwargs_override": model_kwargs,
      "sessions": list(self.sessions),
      "subjects": self.subjects,
      "batch_size": self.batch_size,
      "lr": self.lr,
      "weight_decay": self.weight_decay,
      "max_epoches": self.max_epoches,
      "early_stop": self.early_stop,
      "patience": self.patience,
      "num_val_subjects": self.num_val_subjects,
      "random_seed": self.random_seed,
      "training_seed": self.training_seed,
      "subject_training_seed_formula": (
        "training_seed + leftout_subject_id - 1"
        if self.training_seed is not None
        else None
      ),
      "deterministic": self.deterministic,
      "num_workers": self.num_workers,
      "source_subjects": self.source_subjects,
      "model": self.model_display_name,
      "input_shape": self.input_shape,
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
    versions = []
    completed_subjects = 0
    subject_times = []

    for subject in self.subjects:
      subject_start_time = datetime.now()
      subject_training_seed = self.get_subject_training_seed(subject)
      if subject_training_seed is not None:
        L.seed_everything(subject_training_seed, workers=True)
      print(
        f"[{subject_start_time.strftime('%Y-%m-%d %H:%M:%S')}] "
        f"Starting leftout subject {subject} "
        f"({completed_subjects + 1}/{len(self.subjects)})"
      )

      model = self.create_model(model_kwargs=model_kwargs)
      datamodule = self.get_datamodule(subject)
      version = f"subject{subject}"
      versions.append(version)
      logger = CSVLogger(save_dir=self.log_base_dir, name=logger_name, version=version)
      tf_logger = TensorBoardLogger(
        save_dir=self.log_base_dir, name=logger_name, version=version
      )
      callbacks = []

      if self.early_stop:
        callbacks.append(
          EarlyStopping(
            monitor="val_accuracy",
            min_delta=0.0001,
            patience=self.patience,
            verbose=False,
            mode="max",
          )
        )

      checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_accuracy",
        mode="max",
        filename=self.checkpoint_name,
      )
      callbacks.append(checkpoint_callback)
      trainer = L.Trainer(
        max_epochs=self.max_epoches,
        max_steps=getattr(self, "training_max_steps", -1),
        accelerator=getattr(self, "accelerator", "gpu"),
        devices=devices,
        logger=[logger, tf_logger],
        callbacks=callbacks,
        precision=precision,
        deterministic=self.deterministic,
      )
      trainer.fit(model=model, datamodule=datamodule)
      trainer.test(
        model=model,
        datamodule=datamodule,
        ckpt_path="best",
        weights_only=False,
      )

      metrics_csv = os.path.join(self.log_base_dir, logger_name, version, "metrics.csv")
      if os.path.exists(metrics_csv):
        try:
          output_dir = os.path.join(
            self.log_base_dir, logger_name, version, "metric_plot"
          )
          plot_metrics_from_csv(metrics_csv, output_dir=output_dir)
        except Exception as exc:
          print(f"Warning: Failed to generate plots for {version}: {exc}")

      subject_end_time = datetime.now()
      duration = (subject_end_time - subject_start_time).total_seconds()
      subject_times.append(duration)
      completed_subjects += 1
      avg_subject_time = sum(subject_times) / len(subject_times)
      remaining_subjects = len(self.subjects) - completed_subjects
      remaining_seconds = int(avg_subject_time * remaining_subjects)
      hours, remainder = divmod(remaining_seconds, 3600)
      minutes, seconds = divmod(remainder, 60)
      elapsed_seconds = int((subject_end_time - experiment_start_time).total_seconds())
      elapsed_hours, elapsed_remainder = divmod(elapsed_seconds, 3600)
      elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
      print(
        f"[{subject_end_time.strftime('%Y-%m-%d %H:%M:%S')}] "
        f"Completed subject in {duration:.1f}s | "
        f"Elapsed: {elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_seconds:02d} | "
        f"Avg subject time: {avg_subject_time:.1f}s | "
        f"Est. remaining: {hours:02d}:{minutes:02d}:{seconds:02d}"
      )
      print("-" * 80)

    log_root_path = os.path.join(self.log_base_dir, logger_name)
    summary_pd: pd.DataFrame = generate_summary_accross_versions(
      log_root_path, versions=versions
    )
    summary_pd.to_csv(os.path.join(log_root_path, "summary.csv"))
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(summary_pd.to_string())
    print("=" * 80)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Experiment completed")


def add_common_paper_moabb_loso_args(
  parser: ArgumentParser,
  default_batch_size: int,
  default_max_epoches: int = 200,
  default_patience: int = 20,
):
  parser.add_argument("--batch_size", default=default_batch_size, type=int)
  parser.add_argument("--max_epoches", default=default_max_epoches, type=int)
  parser.add_argument("--patience", default=default_patience, type=int)
  parser.add_argument("--num_val_subjects", default=5, type=int)
  parser.add_argument("--devices", default="1")
  parser.add_argument("--precision", default="32-true")
  parser.add_argument("--num_workers", default=0, type=int)
  parser.add_argument("--sessions", default="both")
  parser.add_argument("--subjects", default="all")
  parser.add_argument("--smoke_test", action="store_true")
  parser.add_argument("--no_early_stop", action="store_true")


def run_common_paper_moabb_loso_cli(args, experiment_cls):
  experiment = experiment_cls(
    batch_size=16 if args.smoke_test else args.batch_size,
    max_epoches=1 if args.smoke_test else args.max_epoches,
    early_stop=False if args.smoke_test else not args.no_early_stop,
    patience=3 if args.smoke_test else args.patience,
    random_seed=42,
    num_val_subjects=1 if args.smoke_test else args.num_val_subjects,
    subjects=["1"] if args.smoke_test else parse_subjects_arg(args.subjects),
    num_workers=args.num_workers,
    sessions=parse_sessions_arg(args.sessions),
    source_subjects=["2", "3"] if args.smoke_test else None,
  )
  experiment.log_base_dir = LOG_PATH
  experiment.run(
    devices=parse_input_devices_arg(args.devices),
    precision=args.precision,
    run_tag="smoke" if args.smoke_test else None,
  )
