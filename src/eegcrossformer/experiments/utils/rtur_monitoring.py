"""Per-unit CSV monitoring and isolated, fixed training probes for Gain RTUR."""

from contextlib import contextmanager
from copy import deepcopy
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Subset, default_collate

from .rtur_monitoring_metrics import (
  GAIN_METRICS, METRIC_COLUMNS, RTUREpochAccumulator, collect_rtur_metrics,
)


@dataclass(frozen=True)
class RTURMonitoringConfig:
  enabled: bool = True
  step_interval: int = 1
  probe_size: int = 16
  probe_seed: int = 42
  probe_every_epochs: int = 1
  probe_vector_size: int = 4
  probe_vector_every_epochs: int = 10

  def __post_init__(self):
    if (
      self.step_interval < 1 or self.probe_size < 0 or self.probe_every_epochs < 1
      or self.probe_vector_size < 0 or self.probe_vector_every_epochs < 1
    ):
      raise ValueError("step/probe intervals must be positive; probe_size may be zero")


@contextmanager
def preserved_rng():
  python_state, numpy_state = random.getstate(), np.random.get_state()
  devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
  with torch.random.fork_rng(devices=devices):
    try:
      yield
    finally:
      random.setstate(python_state)
      np.random.set_state(numpy_state)


def sample_identity(dataset, index):
  """Resolve Subsets and dataset wrappers without reading or sampling data."""
  if isinstance(dataset, Subset):
    return sample_identity(dataset.dataset, int(dataset.indices[index]))
  datasets = getattr(dataset, "datasets", None)
  if datasets is not None:
    for part, child in enumerate(datasets):
      if index < len(child):
        return {"part": part, **sample_identity(child, index)}
      index -= len(child)
  child = getattr(dataset, "dataset", None)
  if child is not None:
    return sample_identity(child, index)
  return {"dataset": type(dataset).__name__, "index": int(index)}


class FixedTrainingProbe:
  def __init__(self, model, datamodule, config, batch_transform=None):
    self.batch_transform = batch_transform
    with preserved_rng():
      loader = datamodule.train_dataloader()
      indices = getattr(loader.sampler, "indices", range(len(loader.dataset)))
      candidates = np.asarray(list(indices), dtype=np.int64)
      chosen = np.random.default_rng(config.probe_seed).choice(
        candidates, size=min(config.probe_size, len(candidates)), replace=False
      )
      self.sample_ids = [sample_identity(loader.dataset, int(i)) for i in chosen]
      self.loader_indices = chosen.tolist()
      self.batch = default_collate([loader.dataset[int(i)] for i in chosen])
      # Construct before the first training forward, while gate caches are empty.
      self.model = deepcopy(model).eval()

  @torch.no_grad()
  def evaluate(
    self, student, loss_config, *, step, horizon, precision_context, token_limit=0,
  ):
    with preserved_rng(), precision_context():
      self.model.load_state_dict(student.state_dict())
      self.model.eval()
      device = next(student.parameters()).device
      self.model.to(device)
      batch = self.batch
      if self.batch_transform is not None:
        batch = self.batch_transform(batch)
      inputs, labels = batch[0].to(device), batch[1].to(device)
      outputs = self.model(inputs)
      losses = self.model.compute_loss(
        outputs, labels.reshape(-1), config=loss_config, inputs=inputs,
        progress=min(step / max(horizon, 1), 1.), update_teacher=False,
        global_step=step,
      )
      return collect_rtur_metrics(losses, outputs, labels, token_limit=token_limit)


class RTURTrainingMonitor:
  def __init__(self, config, loss_config, schedule_override=None, probe_transform=None):
    self.config, self.loss_config = config, loss_config
    self.schedule_override, self.probe_transform = schedule_override, probe_transform
    self.started = False
    self.files = {}
    self.writers = {}
    self.accumulators = {}
    self.probe = None

  def start(self, module):
    if not self.config.enabled:
      return
    trainer = module.trainer
    planned = trainer.max_steps if trainer.max_steps > 0 else int(trainer.estimated_stepping_batches)
    self.horizon = self.schedule_override or planned
    if self.loss_config.schedule_horizon_steps is not None:
      self.horizon = min(self.horizon, self.loss_config.schedule_horizon_steps)
    self.path = Path(trainer.log_dir or trainer.default_root_dir) / "rtur_monitoring"
    self.path.mkdir(parents=True, exist_ok=True)
    dm = trainer.datamodule
    unit_fields = ("subject_id", "subject_code", "subject", "participant", "session", "session_id", "sessions", "pooled_sessions", "fold", "fold_index", "test_fold", "seed", "random_seed", "test_subject", "leftout_subject_id", "held_out_subject", "training_subjects", "validation_subjects")
    unit = {key: getattr(dm, key) for key in unit_fields if hasattr(dm, key)}
    settings = {}
    for parent in (self.path.parent, self.path.parent.parent):
      settings_path = parent / "experiment_settings.json"
      if settings_path.exists():
        settings = json.loads(settings_path.read_text())
        break
    metadata = {
      "protocol": type(dm).__name__, "unit": unit, "training_seed": torch.initial_seed(),
      "rtur_config": asdict(self.loss_config), "monitoring_config": asdict(self.config),
      "effective_schedule_horizon": self.horizon, "planned_training_steps": planned,
      "precision": str(trainer.precision), "experiment_settings": settings,
      "step_semantics": "train: zero-based optimizer update index; epoch/probe: completed updates",
      "aggregation": {"raw_shape": "active trials", "rank_and_pair_order": "valid trials, normalized inside each trial", "gate_and_ce": "all trials", "weighted_rtur": "all optimizer steps, including zero supervision", "spread_median": "all trial spreads", "temporal_std": "population std inside each trial"},
    }
    is_gain = hasattr(self.loss_config, "gain_tau")
    if is_gain:
      metadata["aggregation"].update({
        "raw_shape": "unweighted shape among positive-gain trials",
        "gain_metrics": "all trials; gain/shape_weighted retains confidence and coverage",
        "candidate_constraints": "gate-space zero-sum box and trust radius",
      })
    if self.config.probe_size:
      self.probe = FixedTrainingProbe(module.model, dm, self.config, self.probe_transform)
      metadata["probe"] = {"mode": "probe_train_eval", "seed": self.config.probe_seed, "sample_ids": self.probe.sample_ids, "loader_indices": self.probe.loader_indices, "preprocessing": "dataset deterministic transform + validation batch transform; no training augmentation"}
    (self.path / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    columns = ["optimizer_step", "epoch", "mode", "ema_update_count", "gradient_sample_step", *METRIC_COLUMNS]
    if is_gain:
      columns.extend(GAIN_METRICS)
    for name in ("steps", "epochs", "probe"):
      self.files[name] = (self.path / f"{name}.csv").open("w", newline="")
      self.writers[name] = csv.DictWriter(self.files[name], fieldnames=columns)
      self.writers[name].writeheader()
      self.files[name].flush()
    if is_gain and self.probe is not None and self.config.probe_vector_size:
      token_columns = [
        "optimizer_step", "epoch", "mode", "probe_index", "loader_index",
        "token_index", "utility", "target_centered", "router_logit_centered",
        "gate", "candidate_gate", "anchor_gate", "ce_advantage", "confidence",
      ]
      self.files["probe_tokens"] = (self.path / "probe_tokens.csv").open("w", newline="")
      self.writers["probe_tokens"] = csv.DictWriter(
        self.files["probe_tokens"], fieldnames=token_columns,
      )
      self.writers["probe_tokens"].writeheader()
      self.files["probe_tokens"].flush()
    self.started = True

  def write(self, stream, values, module, *, step=None, mode="train"):
    row = {"optimizer_step": int(module.global_step) if step is None else step,
           "epoch": int(module.current_epoch), "mode": mode,
           "ema_update_count": int(module.model.ema_update_count), **values}
    if "router_grad/main_ce_norm" in values and stream == "steps":
      row["gradient_sample_step"] = row["optimizer_step"]
    self.writers[stream].writerow(row)
    self.files[stream].flush()

  def record(self, batch, module, stage, step):
    if not self.started or stage not in ("train", "val"):
      return
    self.accumulators.setdefault(stage, RTUREpochAccumulator()).update(batch)
    if stage == "train" and (step % self.config.step_interval == 0 or "router_grad/main_ce_norm" in batch.values):
      self.write("steps", batch.values, module, step=step)

  def finish_epoch(self, module, stage):
    accumulator = self.accumulators.pop(stage, None)
    if accumulator is not None:
      self.write("epochs", accumulator.compute(), module, mode=stage)
    if stage == "train" and self.probe is not None and (int(module.current_epoch) + 1) % self.config.probe_every_epochs == 0:
      epoch = int(module.current_epoch) + 1
      token_limit = self.config.probe_vector_size if (
        "probe_tokens" in self.writers
        and (epoch == 1 or epoch % self.config.probe_vector_every_epochs == 0)
      ) else 0
      batch = self.probe.evaluate(module.model, self.loss_config, step=int(module.global_step),
                                  horizon=self.horizon, precision_context=module.trainer.precision_plugin.forward_context,
                                  token_limit=token_limit)
      self.write("probe", batch.values, module, mode="probe_train_eval")
      if batch.token_values is not None:
        self.write_probe_tokens(batch.token_values, module)

  def write_probe_tokens(self, values, module):
    count, tokens = values["gate"].shape
    for trial in range(count):
      for token in range(tokens):
        self.writers["probe_tokens"].writerow({
          "optimizer_step": int(module.global_step),
          "epoch": int(module.current_epoch), "mode": "probe_train_eval",
          "probe_index": trial, "loader_index": self.probe.loader_indices[trial],
          "token_index": token,
          **{key: float(array[trial, token]) for key, array in values.items()},
        })
    self.files["probe_tokens"].flush()

  def close(self):
    for handle in self.files.values():
      handle.close()
    self.started = False
