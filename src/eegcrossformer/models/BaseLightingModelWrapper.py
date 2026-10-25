from typing import Callable, Dict, Any, Literal, Tuple, Optional, Union, Iterable, TypedDict
from dataclasses import dataclass
import os
import torch
from torch import nn
import lightning as L

Batch = Any
BatchFn = Callable[[Batch], Tuple[Any, Any, Dict[str, Any]]]
LossFn = Callable[[torch.Tensor, torch.Tensor, Dict[str, Any]], torch.Tensor]
MetricsFn = Callable[[torch.Tensor, torch.Tensor, Dict[str, Any]], Dict[str, torch.Tensor]]
PostprocessFn = Callable[[Any, Dict[str, Any]], Any]

class LossComponents(TypedDict, total=False):
    """Optional dictionary for recording individual loss components.

    Usage in loss_fn:
        ctx["loss_components"] = {"ce_loss": ce_val.detach(), "reg_loss": reg_val.detach()}

    Usage in metrics_fn:
        components = ctx.get("loss_components", {})
    """
    pass  # Allows arbitrary string keys with torch.Tensor values

OptFactory = Callable[[Iterable[nn.Parameter]], Union[torch.optim.Optimizer, list]]
SchFactory = Callable[[Union[torch.optim.Optimizer, list], L.Trainer], Any]


@dataclass
class TaskPlugin:
  """Task-specific plugin for model training.

  Attributes:
    batch_fn: Extracts (x, y, ctx) from batch. ctx is a mutable dict.
    loss_fn: Computes loss. Can optionally store components in ctx["loss_components"].
    metrics_fn: Computes metrics. Can read ctx["loss_components"] if available.
    postprocess_fn: Post-processes outputs for prediction.

  Loss Component Recording (Optional):
    To record individual loss components for logging:

    1. In loss_fn, store components in ctx:
       ```python
       def my_loss_fn(outputs, targets, ctx):
           ce_loss = F.cross_entropy(outputs, targets)
           reg_loss = compute_regularization(outputs)

           # Optional: Record components (must detach!)
           ctx["loss_components"] = {
               "ce_loss": ce_loss.detach(),
               "reg_loss": reg_loss.detach()
           }

           return ce_loss + 0.1 * reg_loss
       ```

    2. In metrics_fn, read components:
       ```python
       def my_metrics_fn(outputs, targets, ctx):
           metrics = {"accuracy": compute_accuracy(outputs, targets)}

           # Add loss components to logged metrics
           if "loss_components" in ctx:
               metrics.update(ctx["loss_components"])

           return metrics
       ```

  Notes:
    - Loss components MUST be detached (.detach()) to avoid memory leaks
    - This feature is completely optional and backward compatible
    - The ctx dict is the same object passed to both loss_fn and metrics_fn
    - Avoid reserved keys: "device", "outputs", "targets", "domain_id"
  """
  batch_fn: BatchFn
  loss_fn: LossFn
  metrics_fn: Optional[MetricsFn] = None
  epoch_metrics: Optional[nn.Module] = None
  metrics_output_fn: Callable[[Any], Any] = lambda outputs: outputs
  postprocess_fn: Optional[PostprocessFn] = None  # used by predict_step
  train_batch_transform: Optional[Callable[[Any, Any], Tuple[Any, Any]]] = None
  val_batch_transform: Optional[Callable[[Any, Any], Tuple[Any, Any]]] = None
  test_batch_transform: Optional[Callable[[Any, Any], Tuple[Any, Any]]] = None
  monitoring: Any = None


class LightningModelWrapper(L.LightningModule):
  def __init__(
    self,
    model: nn.Module,
    task: TaskPlugin,  # task adapter
    optimizer_fn: Optional[OptFactory] = None,
    scheduler_fn: Optional[SchFactory] = None,
    # Default hyperparameters, used only by the factories, not by the step logic
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    **kwargs,
  ):
    super().__init__(**kwargs)
    self.model = model
    self.task = task
    self.save_hyperparameters(ignore=["model", "task", "optimizer_fn", "scheduler_fn"])
    self._default_lr = lr
    self._default_wd = weight_decay
    self.optimizer_fn = optimizer_fn or (
      lambda params: torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    )
    self.scheduler_fn = scheduler_fn  # may be None
    self.stage_metrics = (
      nn.ModuleDict(
        {
          f"{stage}_metrics": task.epoch_metrics.clone()
          for stage in ("train", "val", "test")
        }
      )
      if task.epoch_metrics is not None
      else None
    )

  # ---- Lightning interface ----
  def forward(self, x):
    return self.model(*x) if isinstance(x, tuple) else self.model(x) 

  def configure_optimizers(self):
    opt = self.optimizer_fn(self.parameters())
    if self.scheduler_fn is None:
      return opt
    sch = self.scheduler_fn(opt, self.trainer)
    if sch is None:
      return opt
    return (
      {"optimizer": opt, "lr_scheduler": sch}
      if isinstance(opt, torch.optim.Optimizer)
      else [opt, sch]
    )

  # ---- shared step ----
  def _shared_step(self, batch, stage: Literal["train", "val", "test"]):
    # Apply batch transform based on stage
    if stage == "train" and self.task.train_batch_transform is not None:
      batch = self.task.train_batch_transform(batch)
    elif stage == "val" and self.task.val_batch_transform is not None:
      batch = self.task.val_batch_transform(batch)
    elif stage == "test" and self.task.test_batch_transform is not None:
      batch = self.task.test_batch_transform(batch)

    # after_batch_fn = self.task.batch_fn(batch)
    # print(f"after_batch_fn {after_batch_fn}")

    x, y, ctx = self.task.batch_fn(batch)
    ctx["device"] = self.device
    # Losses with a training schedule (for example the Gain RTUR warm-up/ramp)
    # need stage and optimizer-step metadata, but TaskPlugin deliberately keeps
    # the loss signature framework-agnostic.  Put the minimal read-only
    # metadata in the existing context instead of coupling those losses to the
    # Lightning wrapper.
    ctx["stage"] = stage
    ctx["global_step"] = int(self.global_step)
    trainer = getattr(self, "_trainer", None)
    total_training_steps = None
    if trainer is not None:
      max_steps = getattr(trainer, "max_steps", None)
      if isinstance(max_steps, int) and max_steps > 0:
        total_training_steps = max_steps
      else:
        estimated = getattr(trainer, "estimated_stepping_batches", None)
        if isinstance(estimated, (int, float)) and 0 < estimated < float("inf"):
          total_training_steps = int(estimated)
    ctx["total_training_steps"] = total_training_steps
    ctx["training_progress"] = (
      min(float(ctx["global_step"]) / total_training_steps, 1.0)
      if total_training_steps
      else 1.0
    )
    monitor = self.task.monitoring
    ctx["rtur_monitoring_enabled"] = (
      monitor is not None and monitor.started and stage in ("train", "val")
      and not getattr(trainer, "sanity_checking", False)
    )
    outputs = self(x)
    loss = self.task.loss_fn(outputs, y, ctx)
    if "rtur_batch_metrics" in ctx:
      monitor.record(ctx.pop("rtur_batch_metrics"), self, stage, ctx["global_step"])

    if self.stage_metrics is not None:
      metric_outputs = self.task.metrics_output_fn(outputs)
      self.stage_metrics[f"{stage}_metrics"].update(
        metric_outputs, y.reshape(-1).long()
      )

    # Determine batch size for logging
    batch_size = y.size(0) if isinstance(y, torch.Tensor) else len(y)

    self.log(
      f"{stage}_loss",
      loss,
      prog_bar=True,
      on_step=(stage == "train"),
      on_epoch=True,
      sync_dist=True,
      batch_size=batch_size,
    )

    if self.task.metrics_fn is not None:
      mdict = self.task.metrics_fn(outputs, y, ctx)
      if self.stage_metrics is not None:
        epoch_metric_names = set(
          self.stage_metrics[f"{stage}_metrics"].keys()
        )
        mdict = {
          name: value
          for name, value in mdict.items()
          if name not in epoch_metric_names
        }
      # Prefix metrics with the stage to avoid name collisions
      mdict = {f"{stage}_{k}": v for k, v in mdict.items()}
      if mdict:
        self.log_dict(
          mdict, prog_bar=(stage != "train"), on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size
        )

    return loss, {"outputs": outputs, "targets": y,**ctx}

  def training_step(self, batch, batch_idx):
    loss, _ = self._shared_step(batch, "train")
    return loss

  def _log_epoch_metrics(self, stage: Literal["train", "val", "test"]):
    if self.stage_metrics is None:
      return
    metrics = self.stage_metrics[f"{stage}_metrics"]
    values = {
      f"{stage}_{name}": value
      for name, value in metrics.compute().items()
    }
    self.log_dict(
      values,
      prog_bar=(stage != "train"),
      on_step=False,
      on_epoch=True,
      sync_dist=False,
    )
    metrics.reset()

  def on_train_epoch_end(self):
    self._log_epoch_metrics("train")
    if self.task.monitoring is not None and self.task.monitoring.started:
      self.task.monitoring.finish_epoch(self, "train")

  def on_validation_epoch_end(self):
    self._log_epoch_metrics("val")
    if self.task.monitoring is not None and self.task.monitoring.started:
      self.task.monitoring.finish_epoch(self, "val")

  def on_fit_start(self):
    if self.task.monitoring is not None:
      self.task.monitoring.start(self)

  def on_fit_end(self):
    if self.task.monitoring is not None:
      self.task.monitoring.close()

  def on_test_epoch_end(self):
    self._log_epoch_metrics("test")

  def on_before_optimizer_step(self, optimizer):
    # Log per-branch gradient norms if the model supports it
    model = self.model
    if hasattr(model, "get_branch_grad_norms"):
      grad_norms = model.get_branch_grad_norms()
      for k, v in grad_norms.items():
        self.log(k, v, on_step=False, on_epoch=True, prog_bar=False)

  def validation_step(self, batch, batch_idx):
    self._shared_step(batch, "val")

  def test_step(self, batch, batch_idx):
    _, ctx = self._shared_step(batch, "test")

    # save test logits to draw AUC Curve, confusion matrix... in future
    x_hat = ctx["outputs"]
    targets = ctx["targets"]
    log_path = self.trainer.log_dir
    test_logits_folder = os.path.join(log_path, "test_logits")
    if not os.path.exists(test_logits_folder):
      os.makedirs(test_logits_folder)
    torch.save(
      {"y_pred": x_hat, "y_true": targets},
      os.path.join(test_logits_folder, f"test_batch{batch_idx}.pt"),
    )

    return ctx  # callbacks handle saving and other I/O

  def predict_step(self, batch, batch_idx, dataloader_idx=0):
    x, _, ctx = self.task.batch_fn(batch)
    outputs = self(x)
    return (
      self.task.postprocess_fn(outputs, ctx) if self.task.postprocess_fn else outputs
    )
