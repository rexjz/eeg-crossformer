"""EEGCrossformer on pooled-session OpenBMI: [1, 3.5) s, NoAug, fixed budget."""
from argparse import ArgumentParser

from .OpenBMIPooledSessionKfoldExperiment import OpenBMIPaperMOABBPooledSessionKfoldExperiment
from eegcrossformer.experiments.by_dataset.openbmi_mi.kfold._paper_moabb_all_channels import parse_subjects_arg, parse_folds_arg, parse_sessions_arg
from eegcrossformer.experiments.utils.finite_training_guard import FiniteTrainingGuard
from eegcrossformer.experiments.utils.final_recipe import FinalRecipeMixin, add_common_args, recipe_kwargs_from_args, configure_run, run_kwargs, unified_tokenizer_spec


PROTOCOL = 'openbmi_within_subject'
TOKENIZER = unified_tokenizer_spec(250, 625)


class EEGCrossformerExperiment(FinalRecipeMixin, OpenBMIPaperMOABBPooledSessionKfoldExperiment):
    event_window_seconds = (1.0, 3.5)
    input_shape = [2, 62, 625]
    num_classes = 2
    include_session_id = False
    offline_transform_profile = 'crossformer_v7_2band'
    default_gain_radius = .1
    default_gain_tau = .1
    tokenizer_kwargs = TOKENIZER['model_kwargs']

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('two_stage_retrain', False)
        kwargs.setdefault('diagnostic_validation_as_test', False)
        self._finite_training_guard = None
        super().__init__(*args, **kwargs)
        self.schedule_total_steps_override = self.rtur_config.schedule_horizon_steps

    def get_protocol_settings(self):
        return {**super().get_protocol_settings(), **TOKENIZER['protocol_settings'], **self.final_protocol_settings(),
                'protocol_id': PROTOCOL, 'input_shape': self.input_shape,
                'sampling_rate_hz': 250, 'crossformer_branch_bands': [[1, 16], [16, 40]],
                'augmentation': 'none', 'augmentation_codename': 'none',
                'augmentation_config': None, 'mixup_partner_constraint': None,
                'session_id_model_input': False, 'validation_test_augmentation': 'none',
                'checkpoint_selection': 'none; evaluate final weights once',
                'outer_test_used_during_training': False}

    def get_retrain_callbacks(self):
        self._finite_training_guard = FiniteTrainingGuard(monitored_metrics=(
            'train_loss', 'train_loss_epoch', 'train_ce_main', 'train_ce_low',
            'train_ce_high', 'train_loss_rtur', 'train_loss_rtur_shape',
            'train_loss_rtur_ranking', 'grad_norm_l_branch', 'grad_norm_h_branch'))
        return [self._finite_training_guard]

    def get_fixed_training_metadata(self):
        return self._finite_training_guard.protocol_metadata() if self._finite_training_guard else {}

    def create_model(self, model_kwargs=None):
        return self.build_model(model_kwargs)


def get_parser():
    return add_common_args(ArgumentParser(description=__doc__), batch_size=16,
        max_epochs=250, patience=50, precision='bf16-mixed', num_workers=0,
        radius=.1, tau=.1, folds=True, sessions=True)


build_parser = get_parser


def build_experiment(args):
    experiment = EEGCrossformerExperiment(
        batch_size=16 if args.smoke_test else args.batch_size,
        max_epoches=1 if args.smoke_test else args.max_epochs,
        early_stop=False, patience=args.patience,
        subjects=['1'] if args.smoke_test else parse_subjects_arg(args.subjects),
        sessions=parse_sessions_arg(args.sessions),
        folds=[1] if args.smoke_test else parse_folds_arg(args.folds),
        training_seed=args.seed,
        # A fixed budget of max_epochs epochs; --max-steps > 0 switches to an exact step budget.
        fixed_training_steps=5 if args.smoke_test else (args.max_steps if args.max_steps > 0 else None),
        fixed_training_epochs=None if args.smoke_test or args.max_steps > 0 else args.max_epochs,
        **recipe_kwargs_from_args(args))
    return configure_run(experiment, args)


def run_from_args(args):
    return build_experiment(args).run(num_workers=args.num_workers, **run_kwargs(args))


def main(argv=None):
    return run_from_args(get_parser().parse_args(argv))


if __name__ == '__main__':
    main()
