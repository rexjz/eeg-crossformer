"""Recover the most recent finite training state after numerical failure."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import os
from pathlib import Path
from typing import Any, Iterable

import torch
from lightning.pytorch import Callback, LightningModule, Trainer


def _is_finite_tensor(value: torch.Tensor) -> bool:
  if not (value.is_floating_point() or value.is_complex()):
    return True
  return bool(torch.isfinite(value.detach()).all().item())


def _iter_tensors(value: Any, prefix: str = "") -> Iterable[tuple[str, torch.Tensor]]:
  if isinstance(value, torch.Tensor):
    yield prefix, value
    return
  if isinstance(value, Mapping):
    for index, (key, child) in enumerate(value.items()):
      key_name = str(key) if isinstance(key, (str, int)) else f"item_{index}"
      child_prefix = f"{prefix}.{key_name}" if prefix else key_name
      yield from _iter_tensors(child, child_prefix)
    return
  if isinstance(value, (list, tuple)):
    for index, child in enumerate(value):
      child_prefix = f"{prefix}.{index}" if prefix else str(index)
      yield from _iter_tensors(child, child_prefix)


def _first_nonfinite(
  values: Iterable[tuple[str, torch.Tensor]],
) -> str | None:
  for name, value in values:
    if not _is_finite_tensor(value):
      return name
  return None


def _clone_to_cpu(value: Any) -> Any:
  if isinstance(value, torch.Tensor):
    return value.detach().to(device="cpu").clone()
  if isinstance(value, Mapping):
    return {key: _clone_to_cpu(child) for key, child in value.items()}
  if isinstance(value, list):
    return [_clone_to_cpu(child) for child in value]
  if isinstance(value, tuple):
    return tuple(_clone_to_cpu(child) for child in value)
  return deepcopy(value)


class FiniteTrainingGuard(Callback):
  """Keep one rolling finite state and restore it after numerical failure.

  The callback observes the optimized loss before backward, gradients after
  unscaling, parameters after each optimizer step, and selected epoch metrics.
  It never selects a checkpoint using validation or test information.
  """

  def __init__(
    self,
    *,
    checkpoint_filename: str = "last_finite.ckpt",
    checkpoint_interval_epochs: int = 50,
    monitored_metrics: Iterable[str] = (),
  ):
    super().__init__()
    filename = Path(checkpoint_filename)
    if filename.name != checkpoint_filename:
      raise ValueError("checkpoint_filename must be a filename, not a path")
    if checkpoint_interval_epochs < 1:
      raise ValueError("checkpoint_interval_epochs must be positive")
    self.checkpoint_filename = checkpoint_filename
    self.checkpoint_interval_epochs = checkpoint_interval_epochs
    self.monitored_metrics = tuple(monitored_metrics)
    self.checkpoint_path: str | None = None
    self.checkpoint_epoch: int | None = None
    self.checkpoint_global_step: int | None = None
    self.last_finite_epoch: int | None = None
    self.last_finite_global_step: int | None = None
    self.finite_snapshot_count = 0
    self.checkpoint_write_count = 0
    self.triggered = False
    self.trigger_reason: str | None = None
    self.detected_epoch: int | None = None
    self.detected_global_step: int | None = None
    self.training_end_global_step: int | None = None
    self.recovery_applied = False
    self._last_finite_state: dict[str, Any] | None = None

  @property
  def stopped_for_nonfinite(self) -> bool:
    return self.triggered

  def _resolve_checkpoint_path(self, trainer: Trainer) -> Path:
    log_dir = trainer.log_dir or trainer.default_root_dir
    return Path(log_dir) / "checkpoints" / self.checkpoint_filename

  def _capture_finite_state(
    self,
    trainer: Trainer,
    pl_module: LightningModule,
    *,
    epoch: int,
  ) -> None:
    self._last_finite_state = {
      "state_dict": _clone_to_cpu(pl_module.state_dict()),
      "optimizer_states": [
        _clone_to_cpu(optimizer.state_dict())
        for optimizer in trainer.optimizers
      ],
    }
    self.last_finite_epoch = epoch
    self.last_finite_global_step = int(trainer.global_step)
    self.finite_snapshot_count += 1

  def _persist_finite_checkpoint(self, trainer: Trainer) -> None:
    checkpoint_path = self._resolve_checkpoint_path(trainer)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_name(
      f".{checkpoint_path.name}.temporary"
    )
    trainer.save_checkpoint(str(temporary_path))
    os.replace(temporary_path, checkpoint_path)
    self.checkpoint_path = str(checkpoint_path)
    self.checkpoint_epoch = self.last_finite_epoch
    self.checkpoint_global_step = self.last_finite_global_step
    self.checkpoint_write_count += 1

  def _trip(self, trainer: Trainer, reason: str) -> None:
    if self.triggered:
      return
    self.triggered = True
    self.trigger_reason = reason
    self.detected_epoch = int(trainer.current_epoch)
    self.detected_global_step = int(trainer.global_step)
    trainer.should_stop = True
    print(
      "FiniteTrainingGuard detected "
      f"{reason} at epoch {self.detected_epoch}, "
      f"global step {self.detected_global_step}; training will restore "
      "the last finite state."
    )

  @staticmethod
  def _clear_gradients(pl_module: LightningModule) -> None:
    for parameter in pl_module.parameters():
      parameter.grad = None

  def on_train_start(
    self,
    trainer: Trainer,
    pl_module: LightningModule,
  ) -> None:
    bad_state = _first_nonfinite(pl_module.state_dict().items())
    if bad_state is not None:
      raise RuntimeError(
        f"Training started with non-finite model state: {bad_state}"
      )
    self._capture_finite_state(trainer, pl_module, epoch=-1)
    self._persist_finite_checkpoint(trainer)

  def on_before_backward(
    self,
    trainer: Trainer,
    pl_module: LightningModule,
    loss: torch.Tensor,
  ) -> None:
    if not _is_finite_tensor(loss):
      self._trip(trainer, "non-finite optimized loss")

  def on_before_optimizer_step(
    self,
    trainer: Trainer,
    pl_module: LightningModule,
    optimizer: torch.optim.Optimizer,
  ) -> None:
    if not self.triggered:
      bad_gradient = _first_nonfinite(
        (name, parameter.grad)
        for name, parameter in pl_module.named_parameters()
        if parameter.grad is not None
      )
      if bad_gradient is not None:
        self._trip(trainer, f"non-finite gradient: {bad_gradient}")
    if self.triggered:
      self._clear_gradients(pl_module)

  def on_train_batch_end(
    self,
    trainer: Trainer,
    pl_module: LightningModule,
    outputs,
    batch,
    batch_idx: int,
  ) -> None:
    if self.triggered:
      return
    bad_parameter = _first_nonfinite(pl_module.named_parameters())
    if bad_parameter is not None:
      self._trip(trainer, f"non-finite parameter: {bad_parameter}")

  def on_train_epoch_end(
    self,
    trainer: Trainer,
    pl_module: LightningModule,
  ) -> None:
    if self.triggered:
      return

    for metric_name in self.monitored_metrics:
      metric = trainer.callback_metrics.get(metric_name)
      if isinstance(metric, torch.Tensor) and not _is_finite_tensor(metric):
        self._trip(trainer, f"non-finite metric: {metric_name}")
        return

    bad_state = _first_nonfinite(pl_module.state_dict().items())
    if bad_state is not None:
      self._trip(trainer, f"non-finite model state: {bad_state}")
      return

    for optimizer_index, optimizer in enumerate(trainer.optimizers):
      bad_optimizer_state = _first_nonfinite(
        _iter_tensors(optimizer.state, f"optimizer_{optimizer_index}")
      )
      if bad_optimizer_state is not None:
        self._trip(
          trainer,
          f"non-finite optimizer state: {bad_optimizer_state}",
        )
        return

    epoch = int(trainer.current_epoch)
    self._capture_finite_state(trainer, pl_module, epoch=epoch)
    if (epoch + 1) % self.checkpoint_interval_epochs == 0:
      self._persist_finite_checkpoint(trainer)

  @torch.no_grad()
  def on_train_end(
    self,
    trainer: Trainer,
    pl_module: LightningModule,
  ) -> None:
    self.training_end_global_step = int(trainer.global_step)
    if not self.triggered:
      return
    if self._last_finite_state is None:
      raise RuntimeError("No finite training state is available for recovery")

    pl_module.load_state_dict(
      self._last_finite_state["state_dict"],
      strict=True,
    )
    optimizer_states = self._last_finite_state["optimizer_states"]
    if len(optimizer_states) == len(trainer.optimizers):
      for optimizer, optimizer_state in zip(
        trainer.optimizers,
        optimizer_states,
      ):
        optimizer.load_state_dict(optimizer_state)
    self._clear_gradients(pl_module)

    bad_state = _first_nonfinite(pl_module.state_dict().items())
    if bad_state is not None:
      raise RuntimeError(
        f"Recovered snapshot contains non-finite model state: {bad_state}"
      )
    self.recovery_applied = True
    self._persist_finite_checkpoint(trainer)

  def protocol_metadata(self) -> dict[str, Any]:
    return {
      "finite_training_guard": (
        "rolling_train_only_state_with_periodic_checkpoint"
      ),
      "finite_guard_monitored_metrics": list(self.monitored_metrics),
      "finite_guard_checkpoint": self.checkpoint_path,
      "finite_guard_snapshot_storage": "cpu_memory_each_finite_epoch",
      "finite_guard_snapshot_count": self.finite_snapshot_count,
      "finite_guard_checkpoint_interval_epochs": (
        self.checkpoint_interval_epochs
      ),
      "finite_guard_checkpoint_writes": self.checkpoint_write_count,
      "finite_guard_checkpoint_epoch_zero_based": self.checkpoint_epoch,
      "finite_guard_checkpoint_global_step": self.checkpoint_global_step,
      "finite_guard_triggered": self.triggered,
      "finite_guard_trigger_reason": self.trigger_reason,
      "finite_guard_detected_epoch_zero_based": self.detected_epoch,
      "finite_guard_detected_global_step": self.detected_global_step,
      "finite_guard_last_epoch_zero_based": self.last_finite_epoch,
      "finite_guard_last_global_step": self.last_finite_global_step,
      "finite_guard_training_end_global_step": self.training_end_global_step,
      "finite_guard_recovery_applied": self.recovery_applied,
      "checkpoint_selection": (
        "last_finite_recovery_after_nonfinite"
        if self.triggered
        else "none_final_weights"
      ),
    }
