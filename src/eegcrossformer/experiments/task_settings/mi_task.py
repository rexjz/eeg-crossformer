from torchmetrics import Accuracy, CohenKappa, F1Score, MetricCollection

import torch

from eegcrossformer.models.BaseLightingModelWrapper import TaskPlugin


def _reg_batch(batch):
  return batch["data"], batch["target"], {}


def mi_loss(outputs: torch.Tensor, targets: torch.Tensor, ctx):
  targets = targets.reshape(-1).long()
  return torch.nn.functional.cross_entropy(
    outputs.reshape(targets.numel(), -1), targets
  )


def create_epoch_metrics(num_classes: int) -> MetricCollection:
  """Metrics that must be accumulated over a complete stage epoch."""

  return MetricCollection(
    {
      "accuracy": Accuracy(
        task="multiclass", num_classes=num_classes, top_k=1, average="micro"
      ),
      "balanced_accuracy": Accuracy(
        task="multiclass", num_classes=num_classes, top_k=1, average="macro"
      ),
      "f1score": F1Score(
        task="multiclass", num_classes=num_classes, top_k=1, average="macro"
      ),
      "kappa": CohenKappa(task="multiclass", num_classes=num_classes),
    }
  )


def get_mi_classification_task(
  num_classes,
  reg_batch: callable = lambda x:x,
  train_batch_transform: callable = None,
  val_batch_transform: callable = None,
  test_batch_transform: callable = None,
  loss_fn: callable = None,
  metrics_fn_batch_adapt_fn: callable = lambda x:x
) -> TaskPlugin:
  return TaskPlugin(
    batch_fn=reg_batch,
    loss_fn=mi_loss if loss_fn is None else loss_fn,
    metrics_fn=create_batch_diagnostics(metrics_fn_batch_adapt_fn),
    epoch_metrics=create_epoch_metrics(num_classes),
    metrics_output_fn=metrics_fn_batch_adapt_fn,
    postprocess_fn=lambda out, ctx: out.squeeze(),
    train_batch_transform=train_batch_transform,
    val_batch_transform=val_batch_transform,
    test_batch_transform=test_batch_transform,
  )


def get_mi_classification_task_moe_specific(
  num_classes, reg_batch: callable = _reg_batch
) -> TaskPlugin:
  return get_mi_classification_task(
    num_classes=num_classes,
    reg_batch=reg_batch,
    loss_fn=lambda outputs, targets, _ctx: mi_loss(
      outputs[0], targets, _ctx
    ),
    metrics_fn_batch_adapt_fn=lambda outputs: outputs[0],
  )


def create_batch_diagnostics(batch_adapt_fn: callable):
  """Keep additive compatibility metrics; nonlinear metrics live at epoch level."""

  def metrics_fn(outputs, targets, ctx):
    targets = targets.reshape(-1).long()
    logits = batch_adapt_fn(outputs).reshape(targets.numel(), -1)
    metrics = {"accuracy": logits.argmax(dim=1).eq(targets).float().mean()}
    metrics.update(ctx.get("loss_components", {}))
    return metrics

  return metrics_fn
