"""Shared final-model construction and command-line options for five protocols."""

from argparse import ArgumentParser
from dataclasses import asdict

from eegcrossformer.models.eegcrossformer import EEGCrossformer, GainRTURLossConfig
from eegcrossformer.models.BaseLightingModelWrapper import LightningModelWrapper
from eegcrossformer.experiments.utils.gain_rtur_task import create_gain_rtur_task
from eegcrossformer.experiments.utils.rtur_monitoring import RTURMonitoringConfig
from eegcrossformer.experiments.utils.parse_input_devices import parse_input_devices_arg


TOKEN_RECEPTIVE_FIELD_MS = 300
TOKEN_HOP_MS = 100


def unified_tokenizer_spec(sampling_rate_hz, input_time):
    """Shared 300-ms receptive field / 100-ms hop grid with valid convolution.

    Both branches receive identical kernel, pool and stride settings, so paired
    low/high tokens cover the same interval. The odd temporal kernel spans about
    200 ms and the power-averaging window fills the rest of the receptive field.
    See docs/tokenizer_geometry.md.
    """
    if sampling_rate_hz <= 0 or input_time <= 0:
        raise ValueError('sampling_rate_hz and input_time must be positive')
    kernel = round(0.2 * sampling_rate_hz)
    if kernel % 2 == 0:
        kernel += 1 if kernel < TOKEN_RECEPTIVE_FIELD_MS * sampling_rate_hz / 1000 else -1
    receptive_field = round(TOKEN_RECEPTIVE_FIELD_MS * sampling_rate_hz / 1000)
    pool = receptive_field - kernel + 1
    hop = round(TOKEN_HOP_MS * sampling_rate_hz / 1000)
    if pool <= 0 or hop <= 0 or input_time < receptive_field:
        raise ValueError('temporal receptive field does not fit this input')
    tokens = (input_time - receptive_field) // hop + 1
    return {
        'model_kwargs': {'kernel_size': kernel, 'pool_size': pool, 'pool_stride': hop},
        'protocol_settings': {
            'model_variant': 'EEGCrossformer',
            'token_receptive_field_ms': TOKEN_RECEPTIVE_FIELD_MS,
            'token_receptive_field_samples': receptive_field,
            'token_hop_ms': TOKEN_HOP_MS,
            'temporal_kernel_sizes': [kernel, kernel],
            'branch_temporal_kernel_sizes': [kernel, kernel],
            'branch_temporal_kernel_durations_ms': [1000 * kernel / sampling_rate_hz] * 2,
            'temporal_kernel_parity': 'odd',
            'branch_temporal_kernels_are_odd': True,
            'branch_temporal_kernel_padding': [0, 0],
            'branch_pool_sizes': [pool, pool],
            'branch_pool_strides': [hop, hop],
            'temporal_token_grid': {'pool_sizes': [pool, pool], 'pool_strides': [hop, hop],
                                    'tokens_per_trial': tokens},
            'tokens_per_trial': tokens,
            'num_aligned_tokens': tokens,
            'temporal_readout_kernel_size': tokens,
            'tokenizer_alignment': 'architecture-identical low/high tokenizers with valid convolution',
        },
    }


def add_common_args(parser, *, batch_size, max_epochs, patience, precision,
                    num_workers, seed=42, radius=.025, tau=.01,
                    folds=False, sessions=False, val_subjects=None, max_steps=-1):
    parser.add_argument('--subjects', default='all')
    if folds:
        parser.add_argument('--folds', default='all')
    if sessions:
        parser.add_argument('--sessions', default='all')
    if val_subjects is not None:
        parser.add_argument('--num-val-subjects', '--num_val_subjects', dest='num_val_subjects', type=int, default=val_subjects)
    parser.add_argument('--batch-size', '--batch_size', dest='batch_size', type=int, default=batch_size)
    parser.add_argument('--max-epochs', '--max_epochs', '--max_epoches', dest='max_epochs', type=int, default=max_epochs)
    parser.add_argument('--max-steps', '--fixed_training_steps', dest='max_steps', type=int, default=max_steps)
    parser.add_argument('--patience', type=int, default=patience)
    parser.add_argument('--seed', '--training_seed', dest='seed', type=int, default=seed)
    parser.add_argument('--split-seed', dest='split_seed', type=int, default=42)
    parser.add_argument('--num-workers', '--num_workers', dest='num_workers', type=int, default=num_workers)
    parser.add_argument('--devices', default='1')
    parser.add_argument('--accelerator', choices=('gpu', 'cpu', 'auto'), default='gpu')
    parser.add_argument('--precision', default=precision)
    parser.add_argument('--model-backend', choices=('gla', 'gla_cpu'), default='gla')
    parser.add_argument('--learning-rate', '--learning_rate', dest='learning_rate', type=float, default=3e-4)
    parser.add_argument('--weight-decay', dest='weight_decay', type=float, default=.05)
    parser.add_argument('--run-tag', '--run_tag', dest='run_tag')
    parser.add_argument('--smoke-test', '--smoke_test', dest='smoke_test', action='store_true')
    parser.add_argument('--rtur', choices=('on', 'off'), default='on')
    parser.add_argument('--auxiliary-ce-weight', dest='auxiliary_ce_weight', type=float, default=.3)
    parser.add_argument('--rtur-shape-weight', dest='rtur_shape_weight', type=float, default=.1)
    parser.add_argument('--rtur-horizon', dest='rtur_horizon', type=int, default=2500,
                        help='RTUR schedule horizon; 0 uses the complete planned budget.')
    parser.add_argument('--gain-radius', dest='gain_radius', type=float, default=radius)
    parser.add_argument('--gain-tau', dest='gain_tau', type=float, default=tau)
    parser.add_argument('--no-monitoring', action='store_true')
    parser.add_argument('--probe-size', type=int, default=16)
    parser.add_argument('--gradient-interval', type=int, default=50)
    return parser


def config_from_args(args):
    return GainRTURLossConfig(
        auxiliary_ce_weight=args.auxiliary_ce_weight,
        shape_weight=args.rtur_shape_weight if args.rtur == 'on' else 0.,
        candidate_steps=3, candidate_step_size=.1,
        candidate_radius=args.gain_radius, gate_epsilon=1e-4, gain_tau=args.gain_tau,
        schedule_horizon_steps=args.rtur_horizon or None,
        gradient_diagnostic_interval=args.gradient_interval,
    )


def recipe_kwargs_from_args(args):
    return dict(
        rtur_config=config_from_args(args),
        monitoring_config=RTURMonitoringConfig(enabled=not args.no_monitoring,
                                               probe_size=args.probe_size),
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        model_backend=args.model_backend,
    )


def configure_run(experiment, args):
    experiment.accelerator = args.accelerator
    experiment.training_max_steps = 5 if args.smoke_test else args.max_steps
    experiment.experiment_seed = args.split_seed
    experiment.split_random_seed = args.split_seed
    experiment.random_seed = args.split_seed
    return experiment


def run_kwargs(args):
    return dict(devices=parse_input_devices_arg(args.devices), precision=args.precision,
                run_tag='smoke' if args.smoke_test else args.run_tag)


def center_crop_batch(batch, input_time):
    start = (batch[0].shape[-1] - input_time) // 2
    return (batch[0][..., start:start + input_time], batch[1], *batch[2:])


class FinalRecipeMixin:
    """Build EEGCrossformer and its Gain RTUR task on top of a protocol base."""

    model_name = 'EEGCrossformer'
    model_display_name = 'EEGCrossformer'
    default_gain_radius = .025
    default_gain_tau = .01
    lr = 3e-4
    weight_decay = .05
    schedule_total_steps_override = None

    def __init__(self, *args, rtur_config=None, monitoring_config=None,
                 learning_rate=3e-4, weight_decay=.05, model_backend='gla', **kwargs):
        self.rtur_config = rtur_config or GainRTURLossConfig(
            candidate_steps=3, candidate_step_size=.1,
            candidate_radius=self.default_gain_radius, gate_epsilon=1e-4,
            gain_tau=self.default_gain_tau,
        )
        self.monitoring_config = monitoring_config
        self.lr = self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model_backend = model_backend
        super().__init__(*args, **kwargs)

    def get_model_name(self):
        return 'EEGCrossformer'

    def final_protocol_settings(self):
        return {
            'model_variant': 'EEGCrossformer',
            'fusion_norm': 'none', 'post_fusion_layernorm': False,
            'router_type': 'linear', 'branch_encoder_type': self.model_backend,
            'branch_depths': [2, 2], 'embedding_dims': [20, 20],
            'rtur_enabled': self.rtur_config.shape_weight > 0,
            'rtur_loss_config': asdict(self.rtur_config),
            'loss': 'ce_main + auxiliary_ce_weight*(ce_low+ce_high) + schedule*shape_weight*gain_shape',
            'learning_rate': self.lr, 'weight_decay': self.weight_decay,
            'optimizer': 'Adam',
            'rtur_schedule_total_steps_override': self.schedule_total_steps_override,
        }

    def build_model(self, model_kwargs=None, *, train_transform=None, eval_transform=None):
        model = EEGCrossformer(
            n_chan=self.input_shape[1], input_time=self.input_shape[2], n_class=self.num_classes,
            backend=self.model_backend, **self.tokenizer_kwargs, **(model_kwargs or {}),
        )
        task = create_gain_rtur_task(
            model, num_classes=self.num_classes, config=self.rtur_config,
            monitoring_config=self.monitoring_config,
            train_batch_transform=train_transform,
            val_batch_transform=eval_transform, test_batch_transform=eval_transform,
            schedule_total_steps_override=self.schedule_total_steps_override,
        )
        return LightningModelWrapper(model=model, task=task, lr=self.lr,
                                     weight_decay=self.weight_decay)
