import json
import os
from argparse import ArgumentParser
from typing import Literal, TypeAlias
from eegcrossformer.data.dataset.bci_competition.BCIC34aSessionDependentKFoldDataModule import (
  BCIC34aSessionDependentKFoldDataModule,
)
from eegcrossformer.data.dataset.bci_competition.cache_utils import (
  bcic34a_cache_settings,
)
from eegcrossformer.project_config import LOG_PATH
from eegcrossformer.experiments.utils.summary_logs import generate_kfold_summary
from eegcrossformer.experiments.base_experiment import Experiment
from eegcrossformer.experiments.utils.parse_input_devices import parse_input_devices_arg
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torcheeg.transforms.base_transform import BaseTransform


TrainerPrecision: TypeAlias = Literal[
  64,
  32,
  16,
  "transformer-engine",
  "transformer-engine-float16",
  "16-true",
  "16-mixed",
  "bf16-true",
  "bf16-mixed",
  "32-true",
  "64-true",
  "64",
  "32",
  "16",
  "bf16",
]

_BCI34A_SUBJECT_ORDER = ("aa", "al", "av", "aw", "ay")


class BCIC34aSubjectDependentKfoldExperiment(Experiment):
  def __init__(
    self,
    num_folds=5,
    subjects=["aa", "al", "av", "aw", "ay"],
    patience: int = 50,
    max_epochs: int = 500,
    folds: list[int] | None = None,
    training_seed: int | None = None,
    deterministic: bool | Literal["warn"] = False,
    small_training_mode: bool = False,
    small_training_max_epochs: int = 400,
    *args,
    **kwargs,
  ):
    super().__init__(*args, **kwargs)
    self.experiment_seed = 42  # Fixed seed for reproducible k-fold splitting
    self.log_base_dir = LOG_PATH
    self.num_folds = num_folds
    self.subjects = subjects
    self.patience = patience
    self.max_epochs = max_epochs
    self.folds = folds or list(range(1, self.num_folds + 1))
    self.training_seed = training_seed
    self.deterministic = deterministic
    # Small-training switch: 20% train / 80% test, no validation split, fixed epochs, test with the final weights
    self.small_training_mode = small_training_mode
    self.small_training_max_epochs = small_training_max_epochs

  def get_fold_training_seed(self, subject_code: str, fold: int) -> int | None:
    if self.training_seed is None:
      return None
    subject_index = _BCI34A_SUBJECT_ORDER.index(subject_code)
    return self.training_seed + subject_index * self.num_folds + (fold - 1)

  def get_online_transform(self) -> None | BaseTransform:
    return None

  def get_datamodule(self, fold, subject_code, num_workers=4) -> L.LightningDataModule:
    """create an new datamodule instance"""
    return BCIC34aSessionDependentKFoldDataModule(
      fold=fold,
      batch_size=self.batch_size,
      subject_code=subject_code,
      kfold=self.num_folds,
      random_seed=self.experiment_seed,
      num_workers=num_workers,
      transform=self.get_online_transform(),
      small_training_mode=self.small_training_mode,
    )

  def run(
    self,
    model_kwargs: dict = {},
    run_tag=None,
    precision: TrainerPrecision | None = "32-true",
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
      "sessions": getattr(self, "sessions", "subject-dependent protocol"),
      "folds": self.folds,
      "channels": getattr(self, "channels", "118 EEG channels"),
      "batch_size": self.batch_size,
      "max_epochs": self.max_epochs,
      "patience": self.patience,
      "seed": self.experiment_seed,
      "split_seed": self.experiment_seed,
      "training_seed": self.training_seed,
      "fold_training_seed_formula": (
        "training_seed + subject_index * num_folds + (fold - 1)"
        if self.training_seed is not None
        else None
      ),
      "deterministic": self.deterministic,
      "augmentation_protocol": getattr(
        self, "augmentation_protocol", "random_crop_cutmix_random_erasing_rta_m5"
      ),
      "input_shape": getattr(self, "input_shape", [1, 118, 350]),
      "protocol": "BCI Competition III Dataset 4a subject-dependent 5-fold",
    }
    protocol_settings_fn = getattr(self, "get_protocol_settings", None)
    if callable(protocol_settings_fn):
      experiment_settings.update(protocol_settings_fn())
    experiment_settings.update(bcic34a_cache_settings(self.get_online_transform()))
    file_path = os.path.join(self.log_base_dir, logger_name, "experiment_settings.json")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as json_file:
      json.dump(experiment_settings, json_file, indent=4)

    for subject_code in self.subjects:
      for fold in self.folds:
        fold_training_seed = self.get_fold_training_seed(subject_code, fold)
        if fold_training_seed is not None:
          L.seed_everything(fold_training_seed, workers=True)

        model = self.create_model(model_kwargs=model_kwargs)
        datamodule = self.get_datamodule(fold, subject_code, num_workers=num_workers)
        version = f"subject{subject_code}_fold{fold}"
        logger = CSVLogger(
          save_dir=self.log_base_dir, name=logger_name, version=version
        )

        if self.small_training_mode:
          # No validation split: no early stopping or best-checkpoint selection; fixed epochs, test with the final weights
          trainer = L.Trainer(
            max_epochs=self.small_training_max_epochs,
            max_steps=getattr(self, "training_max_steps", -1),
            accelerator=getattr(self, "accelerator", "gpu"),
            devices=devices,
            logger=logger,
            callbacks=[],
            precision=precision,
            deterministic=self.deterministic,
            limit_val_batches=0,
            num_sanity_val_steps=0,
          )
          trainer.fit(model=model, datamodule=datamodule)
          trainer.test(model=model, datamodule=datamodule)
        else:
          monitor = getattr(self, "monitor_metric", "validation_accuracy")
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
            callbacks=[early_stop_callback, checkpoint_callback],
            precision=precision,
            deterministic=self.deterministic,
          )
          trainer.fit(model=model, datamodule=datamodule)
          trainer.test(model=model, datamodule=datamodule, ckpt_path="best")

    log_root_path = os.path.join(self.log_base_dir, logger_name)
    kfold_summary, df = generate_kfold_summary(log_root_path)
    df.to_csv(os.path.join(log_root_path, "summary.csv"))
    kfold_summary.to_csv(os.path.join(log_root_path, "kfold_summary.csv"))


def parse_bci34a_subjects_arg(value: str) -> list[str]:
  value = value.strip().lower()
  all_subjects = ["aa", "al", "av", "aw", "ay"]
  if value == "all":
    return all_subjects
  subjects = [part.strip() for part in value.split(",") if part.strip()]
  invalid = [subject for subject in subjects if subject not in all_subjects]
  if not subjects or invalid:
    raise ValueError("--subjects must be 'all' or comma-separated BCI34a ids")
  return subjects


def parse_bci34a_folds_arg(value: str) -> list[int]:
  value = value.strip().lower()
  if value == "all":
    return list(range(1, 6))
  folds = [int(part.strip()) for part in value.split(",") if part.strip()]
  invalid = [fold for fold in folds if fold < 1 or fold > 5]
  if not folds or invalid:
    raise ValueError("--folds must be 'all' or comma-separated values from 1..5")
  return folds


def add_common_bci34a_kfold_args(
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
  parser.add_argument("--folds", default="all")
  parser.add_argument("--smoke_test", action="store_true")
  parser.add_argument("--small_training_mode", action="store_true")
  parser.add_argument("--small_training_max_epochs", default=300, type=int)


def run_bci34a_kfold_cli(args, experiment_cls):
  experiment = experiment_cls(
    batch_size=16 if args.smoke_test else args.batch_size,
    patience=3 if args.smoke_test else args.patience,
    max_epochs=1 if args.smoke_test else args.max_epochs,
    subjects=["aa"] if args.smoke_test else parse_bci34a_subjects_arg(args.subjects),
    folds=[1] if args.smoke_test else parse_bci34a_folds_arg(args.folds),
    small_training_mode=args.small_training_mode,
    small_training_max_epochs=args.small_training_max_epochs,
  )
  experiment.run(
    devices=parse_input_devices_arg(args.devices),
    precision=args.precision,
    num_workers=args.num_workers,
    run_tag="smoke" if args.smoke_test else None,
  )
