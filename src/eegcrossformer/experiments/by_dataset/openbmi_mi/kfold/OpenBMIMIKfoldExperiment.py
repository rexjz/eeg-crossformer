import json
from eegcrossformer.data.dataset.OpenBMI.mi.OpenBMIMIDataset import (
  OpenBMI_MI_SessionDependent_Kfold_DataModule,
)
from datetime import datetime
import os
from eegcrossformer.project_config import LOG_PATH, openbmi_subjects
from eegcrossformer.experiments.utils.summary_logs import generate_summary
from eegcrossformer.experiments.base_experiment import Experiment
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


class OpenBMIMIKfoldExperiment(Experiment):
  def __init__(
    self,
    subjects=openbmi_subjects,
    kfold=5,
    patience=50,
    early_stop=True,
    max_epoches=2000,
    *args,
    **kwargs,
  ):
    super().__init__(*args, **kwargs)
    self.experiment_seed = 42
    self.log_base_dir = LOG_PATH
    self.subjects = subjects
    self.patience = patience
    self.early_stop = early_stop
    self.max_epoches = max_epoches
    self.kfold = kfold

  def get_datamodule(self, subject_id, session, fold) -> L.LightningDataModule:
    """create an new datamodule instance"""
    return OpenBMI_MI_SessionDependent_Kfold_DataModule(
      batch_size=self.batch_size,
      subject_id=subject_id,
      session=session,
      fold=fold,
      kfold=self.kfold,
      get_item_style="tuple",
      num_workers=4
    )

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
    }
    file_path = os.path.join(self.log_base_dir, logger_name, "experiment_settings.json")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as json_file:
      json.dump(experiment_settings, json_file, indent=4)

    # Track experiment timing
    experiment_start_time = datetime.now()
    total_folds = len(self.subjects) * 2 * self.kfold  # subjects * sessions * folds
    completed_folds = 0
    fold_times = []

    for subject_code in self.subjects:
      for session in ["1", "2"]:
        for fold in range(1, self.kfold + 1):
          fold_start_time = datetime.now()
          print(f"[{fold_start_time.strftime('%Y-%m-%d %H:%M:%S')}] Starting subject {subject_code}, session {session}, fold {fold} ({completed_folds + 1}/{total_folds})")

          model = self.create_model(model_kwargs=model_kwargs)
          datamodule = self.get_datamodule(
            subject_id=subject_code, session=session, fold=fold
          )
          version = (
            f"subject{subject_code.zfill(2)}_session{session.zfill(2)}_fold{fold}"
          )
          logger = CSVLogger(
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
            logger=logger,
            callbacks=callbacks,
            precision=precision,
          )
          trainer.fit(model=model, datamodule=datamodule)
          trainer.test(model=model, datamodule=datamodule, ckpt_path="best")

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
    summary_pd = generate_summary(log_root_path)
    summary_pd.to_csv(os.path.join(log_root_path, "summary.csv"))
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(summary_pd.to_string())
    print("="*80)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Experiment completed")
