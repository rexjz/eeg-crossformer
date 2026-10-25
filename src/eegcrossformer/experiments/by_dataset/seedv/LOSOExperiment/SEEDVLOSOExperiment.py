"""Base class for SEED-V LOSO experiments.

Mirrors `OpenBMIMILOSOExperiment` — iterates over each subject as the
held-out test target, training on the rest.
"""

import json
import os
from datetime import datetime

import lightning as L
import pandas as pd
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from eegcrossformer.data.dataset.seedV.SEEDVLOSODataModule import SEEDVLOSODataModule
from eegcrossformer.experiments.base_experiment import Experiment
from eegcrossformer.experiments.utils.summary_logs import (
  generate_summary_accross_versions,
)
from eegcrossformer.project_config import LOG_PATH, seedv_subjects_code
from eegcrossformer.utils.metric_plot import plot_metrics_from_csv


class SEEDVLOSOExperiment(Experiment):
  def __init__(
    self,
    num_val_subjects: int = 3,
    patience: int = 30,
    early_stop: bool = True,
    max_epoches: int = 2000,
    random_seed: int = 42,
    subjects: list = None,
    *args,
    **kwargs,
  ):
    super().__init__(*args, **kwargs)
    self.experiment_seed = 42
    self.log_base_dir = LOG_PATH
    self.patience = patience
    self.early_stop = early_stop
    self.max_epoches = max_epoches
    self.num_val_subjects = num_val_subjects
    self.random_seed = random_seed
    self.subjects = subjects if subjects is not None else list(seedv_subjects_code)

  def get_datamodule(self, subject_code: str) -> L.LightningDataModule:
    """Default DataModule — subclasses normally override to inject
    offline_transform / online_transform / normalization choices."""
    return SEEDVLOSODataModule(
      leftout_subject_code=subject_code,
      num_val_subjects=self.num_val_subjects,
      batch_size=self.batch_size,
      random_seed=self.random_seed,
      num_workers=4,
    )

  def run(
    self,
    precision: str = "32-true",
    run_tag: str = None,
    devices: int = 1,
    model_kwargs: dict = {},
  ):
    prefix = f"{run_tag}_" if run_tag is not None else ""
    logger_name = f"{prefix}{self.experiment_tag}"
    experiment_settings = {
      "accelerator": getattr(self, "accelerator", "gpu"),
      "max_steps": getattr(self, "fixed_training_steps", None) or getattr(self, "training_max_steps", -1),
      "training_seed": getattr(self, "training_seed", None),
      "training_seed_formula": "training_seed + leftout_subject - 1",
      "precision": precision,
      "devices": devices,
      "model_kwargs_override": model_kwargs,
      "batch_size": self.batch_size,
      "max_epoches": self.max_epoches,
      "patience": self.patience,
      "early_stop": self.early_stop,
      "num_val_subjects": self.num_val_subjects,
      "random_seed": self.random_seed,
      "subjects": self.subjects,
    }
    if hasattr(self, "get_protocol_settings"):
      experiment_settings["protocol_settings"] = self.get_protocol_settings()
    file_path = os.path.join(self.log_base_dir, logger_name, "experiment_settings.json")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as fp:
      json.dump(experiment_settings, fp, indent=4)

    versions = []
    for subject in self.subjects:
      dt_str = datetime.now().isoformat(timespec="seconds")
      print(f"========== SEEDV LOSO leftout subject {subject} {dt_str} ==========")

      training_seed = getattr(self, "training_seed", None)
      if training_seed is not None:
        # One seed per held-out subject, independent of which subjects ran before.
        L.seed_everything(training_seed + int(subject) - 1, workers=True)
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
      callbacks.append(
        ModelCheckpoint(
          save_top_k=1,
          monitor="val_accuracy",
          mode="max",
          filename=self.checkpoint_name,
        )
      )

      trainer = L.Trainer(
        max_epochs=self.max_epoches,
        max_steps=getattr(self, "training_max_steps", -1),
        accelerator=getattr(self, "accelerator", "gpu"),
        devices=devices,
        logger=[logger, tf_logger],
        callbacks=callbacks,
        precision=precision,
      )
      trainer.fit(model=model, datamodule=datamodule)
      trainer.test(model=model, datamodule=datamodule, ckpt_path="best")

      metrics_csv = os.path.join(self.log_base_dir, logger_name, version, "metrics.csv")
      if os.path.exists(metrics_csv):
        try:
          out_dir = os.path.join(self.log_base_dir, logger_name, version, "metric_plot")
          plot_metrics_from_csv(metrics_csv, output_dir=out_dir)
        except Exception as e:
          print(f"Warning: failed to plot metrics for {version}: {e}")

    log_root_path = os.path.join(self.log_base_dir, logger_name)
    summary_pd: pd.DataFrame = generate_summary_accross_versions(
      log_root_path, versions=versions
    )
    summary_pd.to_csv(os.path.join(log_root_path, "summary.csv"))
    print("=========== summary ==========\n")
    print(summary_pd)
