import json
import os
from datetime import datetime
from argparse import ArgumentParser
from eegcrossformer.project_config import LOG_PATH, seedv_subjects_code
from eegcrossformer.experiments.utils.summary_logs import generate_kfold_summary
from eegcrossformer.experiments.base_experiment import Experiment
from eegcrossformer.experiments.utils.parse_input_devices import parse_input_devices_arg
from eegcrossformer.data.dataset.seedV.SEEDVSessionDependentDataModule import (
  SEEDVSessionDependentDataModule,
)
from eegcrossformer.utils.metric_plot import plot_metrics_from_csv
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelSummary

class SEEDVSessionDependentKFoldExperiment(Experiment):
  def __init__(
    self,
    subjects=seedv_subjects_code,
    patience: int = 50,
    max_epochs: int = 500,
    sessions: list[str] | None = None,
    folds: list[int] | None = None,
    training_seed: int | None = None,
    deterministic=False,
    *args,
    **kwargs,
  ):
    super().__init__(*args, **kwargs)
    self.experiment_seed = 42  # Fixed seed for reproducible k-fold splitting
    self.log_base_dir = LOG_PATH
    self.num_folds = 5
    self.subjects = subjects
    self.patience = patience
    self.max_epochs = max_epochs
    self.sessions = sessions or ["1", "2", "3"]
    self.folds = folds or list(range(1, self.num_folds + 1))
    self.training_seed = training_seed
    self.deterministic = deterministic

  def get_unit_training_seed(
    self,
    subject_code: str,
    session_id: str,
    fold: int,
  ) -> int | None:
    """Return a shard-independent seed for one subject/session/fold unit."""

    if self.training_seed is None:
      return None
    subject_index = seedv_subjects_code.index(subject_code)
    session_index = int(session_id) - 1
    return (
      self.training_seed
      + subject_index * 15
      + session_index * self.num_folds
      + (fold - 1)
    )

  def get_datamodule(
    self, subject_code, session_id, fold, num_workers
  ) -> L.LightningDataModule:
    """create an new datamodule instance"""
    return SEEDVSessionDependentDataModule(
      batch_size=self.batch_size,
      num_workers=num_workers,
      subject_code=subject_code,
      session_id=session_id,
      kfold=self.num_folds,
      test_fold=fold,
      kfold_seed=self.experiment_seed,
    )

  def run(
    self,
    model_kwargs: dict = {},
    run_tag=None,
    precision="32-true",
    devices=1,
    num_workers=4,
  ):
    prefix = f"{run_tag}_" if run_tag is not None else ""
    logger_name = f"{prefix}{self.experiment_tag}"
    experiment_settings = {
      "accelerator": getattr(self, "accelerator", "gpu"),
      "max_steps": getattr(self, "fixed_training_steps", None) or getattr(self, "training_max_steps", -1),
      "precision": precision,
      "devices": devices,
      "model_kwargs_override": model_kwargs,
      "subjects": self.subjects,
      "sessions": self.sessions,
      "folds": self.folds,
      "channels": getattr(self, "channels", "62 EEG channels"),
      "batch_size": self.batch_size,
      "max_epochs": self.max_epochs,
      "patience": self.patience,
      "seed": self.experiment_seed,
      "training_seed": self.training_seed,
      "unit_training_seed_formula": (
        "training_seed + subject_index * 15 + "
        "(session - 1) * 5 + (fold - 1)"
        if self.training_seed is not None
        else None
      ),
      "deterministic": self.deterministic,
      "input_shape": getattr(self, "input_shape", "protocol-native"),
      "protocol": "SEED-V subject/session within-session 5-fold",
    }
    protocol_settings_fn = getattr(self, "get_protocol_settings", None)
    if callable(protocol_settings_fn):
      experiment_settings.update(protocol_settings_fn())
    file_path = os.path.join(self.log_base_dir, logger_name, "experiment_settings.json")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as json_file:
      json.dump(experiment_settings, json_file, indent=4)

    # Track experiment timing
    experiment_start_time = datetime.now()
    total_folds = len(self.subjects) * len(self.sessions) * len(self.folds)
    completed_folds = 0
    fold_times = []

    for subject_code in self.subjects:
      for session_id in self.sessions:
        for fold in self.folds:
          fold_start_time = datetime.now()
          unit_training_seed = self.get_unit_training_seed(
            subject_code,
            session_id,
            fold,
          )
          if unit_training_seed is not None:
            L.seed_everything(unit_training_seed, workers=True)
          print(f"[{fold_start_time.strftime('%Y-%m-%d %H:%M:%S')}] Starting subject {subject_code}, session {session_id}, fold {fold} ({completed_folds + 1}/{total_folds})")

          model = self.create_model(model_kwargs=model_kwargs)
          datamodule = self.get_datamodule(
            subject_code, session_id, fold, num_workers=num_workers
          )
          version = (
            f"subject{subject_code.zfill(2)}_session{session_id.zfill(2)}_fold{fold}"
          )
          logger = CSVLogger(
            save_dir=self.log_base_dir, name=logger_name, version=version
          )
          monitor = getattr(self, "monitor_metric", "val_accuracy")
          early_stop_callback = EarlyStopping(
            monitor=monitor,
            min_delta=0.0001,
            patience=self.patience,
            verbose=False,
            mode="max",
          )
          checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor=monitor,
            mode="max",
            filename=self.checkpoint_name,
          )
          trainer = L.Trainer(
            max_epochs=self.max_epochs,
            max_steps=getattr(self, "training_max_steps", -1),
            accelerator=getattr(self, "accelerator", "gpu"),
            devices=devices,
            logger=logger,
            callbacks=[early_stop_callback, checkpoint_callback, ModelSummary(max_depth=-1)],
            precision=precision,
            deterministic=self.deterministic,
          )
          trainer.fit(model=model, datamodule=datamodule)
          trainer.test(model=model, datamodule=datamodule, ckpt_path="best")

          # Generate metric plots
          metrics_csv = os.path.join(self.log_base_dir, logger_name, version, "metrics.csv")
          if os.path.exists(metrics_csv):
            try:
              print(f"\nGenerating metric plots for {version}...")
              output_dir = os.path.join(self.log_base_dir, logger_name, version, "metric_plot")
              plot_metrics_from_csv(metrics_csv, output_dir=output_dir)
              print(f"Plots saved to: {output_dir}")
            except Exception as e:
              print(f"Warning: Failed to generate plots for {version}: {e}")

          # Calculate timing statistics
          fold_end_time = datetime.now()
          fold_duration = (fold_end_time - fold_start_time).total_seconds()
          fold_times.append(fold_duration)
          completed_folds += 1

          # Calculate and print estimated remaining time
          avg_fold_time = sum(fold_times) / len(fold_times)
          remaining_folds = total_folds - completed_folds
          estimated_remaining_seconds = avg_fold_time * remaining_folds
          estimated_remaining_time = int(estimated_remaining_seconds)
          hours, remainder = divmod(estimated_remaining_time, 3600)
          minutes, seconds = divmod(remainder, 60)

          elapsed_time = (fold_end_time - experiment_start_time).total_seconds()
          elapsed_hours, elapsed_remainder = divmod(int(elapsed_time), 3600)
          elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)

          print(f"[{fold_end_time.strftime('%Y-%m-%d %H:%M:%S')}] Completed fold in {fold_duration:.1f}s | "
                f"Elapsed: {elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_seconds:02d} | "
                f"Avg fold time: {avg_fold_time:.1f}s | "
                f"Est. remaining: {hours:02d}:{minutes:02d}:{seconds:02d}")
          print("-" * 80)

    log_root_path = os.path.join(self.log_base_dir, logger_name)
    kfold_summary, df = generate_kfold_summary(log_root_path)
    df.to_csv(os.path.join(log_root_path, "summary.csv"))
    kfold_summary.to_csv(os.path.join(log_root_path, "kfold_summary.csv"))
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(df)
    print("="*80)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Experiment completed")


def parse_seedv_subjects_arg(value: str) -> list[str]:
  value = value.strip().lower()
  if value == "all":
    return seedv_subjects_code
  subjects = [part.strip() for part in value.split(",") if part.strip()]
  if not subjects:
    raise ValueError("--subjects must be 'all' or comma-separated subject ids")
  return subjects


def parse_seedv_sessions_arg(value: str) -> list[str]:
  value = value.strip().lower()
  if value in {"all", "both"}:
    return ["1", "2", "3"]
  sessions = [part.strip() for part in value.split(",") if part.strip()]
  invalid = [session for session in sessions if session not in {"1", "2", "3"}]
  if not sessions or invalid:
    raise ValueError("--sessions must be 'all' or comma-separated values from 1,2,3")
  return sessions


def parse_seedv_folds_arg(value: str) -> list[int]:
  value = value.strip().lower()
  if value == "all":
    return list(range(1, 6))
  folds = [int(part.strip()) for part in value.split(",") if part.strip()]
  invalid = [fold for fold in folds if fold < 1 or fold > 5]
  if not folds or invalid:
    raise ValueError("--folds must be 'all' or comma-separated values from 1..5")
  return folds


def add_common_seedv_kfold_args(
  parser: ArgumentParser,
  default_batch_size: int,
  default_precision: str = "32-true",
  default_num_workers: int = 4,
):
  parser.add_argument("--batch_size", default=default_batch_size, type=int)
  parser.add_argument("--max_epochs", default=500, type=int)
  parser.add_argument("--patience", default=50, type=int)
  parser.add_argument("--devices", default="1")
  parser.add_argument("--precision", default=default_precision)
  parser.add_argument("--num_workers", default=default_num_workers, type=int)
  parser.add_argument("--subjects", default="all")
  parser.add_argument("--sessions", default="all")
  parser.add_argument("--folds", default="all")
  parser.add_argument("--smoke_test", action="store_true")


def run_seedv_kfold_cli(args, experiment_cls):
  experiment = experiment_cls(
    batch_size=16 if args.smoke_test else args.batch_size,
    patience=3 if args.smoke_test else args.patience,
    max_epochs=1 if args.smoke_test else args.max_epochs,
    subjects=[seedv_subjects_code[0]]
    if args.smoke_test
    else parse_seedv_subjects_arg(args.subjects),
    sessions=["1"] if args.smoke_test else parse_seedv_sessions_arg(args.sessions),
    folds=[1] if args.smoke_test else parse_seedv_folds_arg(args.folds),
  )
  experiment.run(
    devices=parse_input_devices_arg(args.devices),
    precision=args.precision,
    num_workers=args.num_workers,
    run_tag="smoke" if args.smoke_test else None,
  )
