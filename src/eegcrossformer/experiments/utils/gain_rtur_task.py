"""Training task for EEGCrossformer: Gain RTUR loss, metrics and monitoring hooks."""

from __future__ import annotations

from typing import Callable

import torch

from eegcrossformer.experiments.task_settings.mi_task import get_mi_classification_task
from eegcrossformer.models.eegcrossformer import EEGCrossformer, GainRTURLoss, GainRTURLossConfig

from .rtur_monitoring import RTURMonitoringConfig, RTURTrainingMonitor
from .rtur_monitoring_metrics import collect_rtur_metrics


def router_gradient_diagnostics(model: EEGCrossformer, losses: GainRTURLoss) -> dict[str, torch.Tensor]:
    """Router gradient norms of main CE and RTUR and their cosine, without touching ``.grad``."""

    parameters = tuple(model.fusion.gate.parameters())
    ce_grads = torch.autograd.grad(losses.ce_main, parameters, retain_graph=True, allow_unused=True)
    rtur_grads = torch.autograd.grad(losses.rtur.total, parameters, retain_graph=True, allow_unused=True)

    def flatten(grads):
        pieces = [grad.reshape(-1) for grad in grads if grad is not None]
        return torch.cat(pieces) if pieces else losses.total.new_zeros(1)

    ce_vector, rtur_vector = flatten(ce_grads), flatten(rtur_grads)
    ce_norm, rtur_norm = ce_vector.norm(), rtur_vector.norm()
    denominator = ce_norm * rtur_norm
    cosine = torch.where(
        denominator > 0.0,
        torch.dot(ce_vector, rtur_vector) / denominator.clamp_min(1e-12),
        denominator.new_zeros(()),
    )
    return {
        "router_grad_norm_main_ce": ce_norm.detach(),
        "router_grad_norm_rtur": rtur_norm.detach(),
        "router_grad_cosine_main_rtur": cosine.detach(),
    }


def create_loss_fn(model: EEGCrossformer, config: GainRTURLossConfig,
                   schedule_total_steps_override: int | None = None) -> Callable:
    """Bind the model so its EMA teacher and router diagnostics share one state."""

    if schedule_total_steps_override is not None and schedule_total_steps_override < 1:
        raise ValueError("schedule_total_steps_override must be positive or None")
    last_gradient_diagnostics = {}

    def loss_fn(outputs, labels, ctx):
        stage = ctx.get("stage", "train")
        global_step = int(ctx.get("global_step", 0))
        total_steps = (schedule_total_steps_override if schedule_total_steps_override is not None
                       else ctx.get("total_training_steps"))
        if total_steps is not None:
            schedule_steps = int(total_steps)
            if config.schedule_horizon_steps is not None:
                schedule_steps = min(schedule_steps, config.schedule_horizon_steps)
            progress = min(global_step / max(schedule_steps, 1), 1.0)
        else:
            progress = float(ctx.get("training_progress", 1.0))
        losses = model.compute_loss(
            outputs, labels.reshape(-1), config=config, inputs=ctx.get("inputs"), progress=progress,
            update_teacher=(stage == "train"), global_step=global_step,
        )
        components = losses.detached_components()
        components.update({
            "loss_total": losses.total.detach(),
            "weighted_aux_ce_low": (config.auxiliary_ce_weight * losses.ce_low).detach(),
            "weighted_aux_ce_high": (config.auxiliary_ce_weight * losses.ce_high).detach(),
            "weighted_rtur_shape": (losses.rtur.schedule_scale * config.shape_weight * losses.rtur.shape).detach(),
            "ema_update_count": model.ema_update_count.detach().float(),
        })
        interval = config.gradient_diagnostic_interval
        sampled_gradient = None
        if stage == "train" and interval > 0 and global_step % interval == 0 and torch.is_grad_enabled():
            last_gradient_diagnostics.clear()
            last_gradient_diagnostics.update(router_gradient_diagnostics(model, losses))
            sampled_gradient = last_gradient_diagnostics
        if stage == "train":
            components.update(last_gradient_diagnostics)
        ctx["loss_components"] = components
        if ctx.get("rtur_monitoring_enabled", False):
            ctx["rtur_batch_metrics"] = collect_rtur_metrics(losses, outputs, labels, sampled_gradient)
        return losses.total

    return loss_fn


def create_metrics_fn(num_classes: int):
    """Main and auxiliary-branch accuracies, branch agreement and gate statistics."""

    def metrics_fn(outputs, labels, ctx):
        targets = labels.reshape(-1)
        out_main = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        main_logits = out_main.reshape(targets.numel(), num_classes)
        pred_main = main_logits.argmax(dim=1)
        metrics = {"accuracy": pred_main.eq(targets).float().mean()}
        if isinstance(outputs, (tuple, list)) and len(outputs) >= 8:
            out_main, out_low, out_high, *_features, gate = outputs
            low_logits = out_low.reshape(targets.numel(), num_classes)
            high_logits = out_high.reshape(targets.numel(), num_classes)
            mean_logits = 0.5 * (low_logits + high_logits)
            pred_low, pred_high, pred_mean = low_logits.argmax(dim=1), high_logits.argmax(dim=1), mean_logits.argmax(dim=1)
            metrics.update({
                "acc_main": pred_main.eq(targets).float().mean(),
                "acc_low": pred_low.eq(targets).float().mean(),
                "acc_high": pred_high.eq(targets).float().mean(),
                "acc_mean_logits": pred_mean.eq(targets).float().mean(),
                "oracle_branch_acc": (pred_low.eq(targets) | pred_high.eq(targets)).float().mean(),
                "agreement_low_high": pred_low.eq(pred_high).float().mean(),
                "agreement_main_mean_logits": pred_main.eq(pred_mean).float().mean(),
            })
            if gate is not None:
                gate_per_token = gate.detach().mean(dim=-1)
                metrics.update({
                    "gate_mean": gate_per_token.mean(),
                    "gate_std": gate_per_token.std(unbiased=False),
                    "gate_token_var": gate_per_token.var(dim=1, unbiased=False).mean(),
                })
        if "loss_components" in ctx:
            metrics.update(ctx["loss_components"])
        return metrics

    return metrics_fn


def create_gain_rtur_task(model: EEGCrossformer, *, num_classes: int, config: GainRTURLossConfig,
                          train_batch_transform=None, val_batch_transform=None, test_batch_transform=None,
                          schedule_total_steps_override: int | None = None,
                          monitoring_config: RTURMonitoringConfig | None = None):
    """Classification task that also hands the transformed input to the EMA teacher."""

    task = get_mi_classification_task(
        num_classes=num_classes,
        reg_batch=lambda batch: (batch[0], batch[1], {"inputs": batch[0]}),
        loss_fn=create_loss_fn(model, config, schedule_total_steps_override=schedule_total_steps_override),
        train_batch_transform=train_batch_transform,
        val_batch_transform=val_batch_transform,
        test_batch_transform=test_batch_transform,
        metrics_fn_batch_adapt_fn=lambda outputs: outputs[0],
    )
    task.metrics_fn = create_metrics_fn(num_classes)
    task.monitoring = RTURTrainingMonitor(
        monitoring_config or RTURMonitoringConfig(), config, schedule_total_steps_override, val_batch_transform,
    )
    return task
