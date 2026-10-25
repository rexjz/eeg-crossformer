import json
import os

import pandas as pd
from eegcrossformer.data.dataset.OpenBMI.mi.OpenBMIMILOSODataModule import (
  OpenBMIMILOSODataModule,
)
from eegcrossformer.experiments.utils.summary_logs import generate_summary_accross_versions
from eegcrossformer.project_config import LOG_PATH
from eegcrossformer.experiments.base_experiment import Experiment
import lightning as L
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from datetime import datetime
from eegcrossformer.utils.metric_plot import plot_metrics_from_csv


openbmi_subjects = [str(i) for i in range(1, 55)]

class OpenBMIMILOSOExperiment(Experiment):
  def __init__(
    self,
    num_val_subjects=5,
    patience=50,
    early_stop=True,
    max_epoches=2000,
    random_seed=42,
    subjects: list[str] = openbmi_subjects,
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
    self.subjects = subjects

  def get_datamodule(self, subject_id) -> L.LightningDataModule:
    """Create a new datamodule instance for the given leftout subject"""
    return OpenBMIMILOSODataModule(
      leftout_subject_id=subject_id,
      num_val_subjects=self.num_val_subjects,
      batch_size=self.batch_size,
      random_seed=self.random_seed,
      num_workers=4,
      num_processes=None,  # Use all available CPUs
    )

  def get_protocol_settings(self) -> dict:
    return {}

  def run(self, precision="32-true", run_tag=None, devices=1, model_kwargs: dict = {}):
    prefix = f"{run_tag}_" if run_tag is not None else ""
    logger_name = f"{prefix}{self.experiment_tag}"
    experiment_settings = {
      "accelerator": getattr(self, "accelerator", "gpu"),
      "max_steps": getattr(self, "fixed_training_steps", None) or getattr(self, "training_max_steps", -1),
      "training_seed": getattr(self, "training_seed", None),
      "precision": precision,
      "devices": devices,
      "model_kwargs_override": model_kwargs,
      "model": self.model_name,
      "subjects": self.subjects,
      "batch_size": self.batch_size,
      "max_epoches": self.max_epoches,
      "early_stop": self.early_stop,
      "patience": self.patience,
      "num_val_subjects": self.num_val_subjects,
      "random_seed": self.random_seed,
    }
    experiment_settings.update(self.get_protocol_settings())
    file_path = os.path.join(self.log_base_dir, logger_name, "experiment_settings.json")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as json_file:
      json.dump(experiment_settings, json_file, indent=4)

    versions = []
    for subject in self.subjects:
      dt_str = datetime.now().isoformat(timespec="seconds")
      print(f"========== leftout subject {subject} {dt_str} ==========")

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
        early_stop_callback = EarlyStopping(
          monitor="val_accuracy",
          min_delta=0.0001,
          patience=self.patience,
          verbose=False,
          mode="max",
        )
        callbacks.append(early_stop_callback)

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
      )
      trainer.fit(model=model, datamodule=datamodule)
      trainer.test(
        model=model,
        datamodule=datamodule,
        ckpt_path="best",
        weights_only=False,
      )

      # Generate metric plots for this fold
      metrics_csv = os.path.join(self.log_base_dir, logger_name, version, "metrics.csv")
      if os.path.exists(metrics_csv):
        try:
          print(f"\nGenerating metric plots for {version}...")
          output_dir = os.path.join(
            self.log_base_dir, logger_name, version, "metric_plot"
          )
          plot_metrics_from_csv(metrics_csv, output_dir=output_dir)
          print(f"Plots saved to: {output_dir}")
        except Exception as e:
          print(f"Warning: Failed to generate plots for {version}: {e}")

    log_root_path = os.path.join(self.log_base_dir, logger_name)
    summary_pd: pd.DataFrame = generate_summary_accross_versions(
      log_root_path, versions=versions
    )
    summary_pd.to_csv(os.path.join(log_root_path, "summary.csv"))

    print("=========== summary ==========\n\n")
    print(summary_pd)
