"""EEGCrossformer on OpenBMI LOSO: cue-aligned [1, 3.5) s, NoAug."""
from argparse import ArgumentParser

from ._paper_moabb_all_channels import PaperMOABBAllChannelsOpenBMIMILOSOExperiment, parse_subjects_arg, parse_sessions_arg
from eegcrossformer.experiments.utils.final_recipe import FinalRecipeMixin, add_common_args, recipe_kwargs_from_args, configure_run, run_kwargs, unified_tokenizer_spec


PROTOCOL = 'openbmi_loso'
TOKENIZER = unified_tokenizer_spec(250, 625)


class EEGCrossformerExperiment(FinalRecipeMixin, PaperMOABBAllChannelsOpenBMIMILOSOExperiment):
    event_window_seconds = (1.0, 3.5)
    input_shape = [2, 62, 625]
    num_classes = 2
    offline_transform_profile = 'crossformer_v7_2band'
    default_gain_radius = .05
    default_gain_tau = .03
    tokenizer_kwargs = TOKENIZER['model_kwargs']

    def get_protocol_settings(self):
        return {**super().get_protocol_settings(), **TOKENIZER['protocol_settings'], **self.final_protocol_settings(),
                'protocol_id': PROTOCOL, 'input_shape': self.input_shape,
                'sampling_rate_hz': 250, 'crossformer_branch_bands': [[1, 16], [16, 40]],
                'augmentation': 'none', 'augmentation_codename': 'none',
                'augmentation_config': None,
                'validation_test_augmentation': 'none'}

    def create_model(self, model_kwargs=None):
        return self.build_model(model_kwargs)


def get_parser():
    return add_common_args(ArgumentParser(description=__doc__), batch_size=128,
        max_epochs=500, patience=50, precision='bf16-mixed', num_workers=0,
        radius=.05, tau=.03, sessions=True, val_subjects=5)


build_parser = get_parser


def build_experiment(args):
    experiment = EEGCrossformerExperiment(
        batch_size=16 if args.smoke_test else args.batch_size,
        max_epoches=1 if args.smoke_test else args.max_epochs,
        early_stop=not args.smoke_test, patience=3 if args.smoke_test else args.patience,
        random_seed=args.split_seed, training_seed=args.seed,
        num_val_subjects=1 if args.smoke_test else args.num_val_subjects,
        subjects=['1'] if args.smoke_test else parse_subjects_arg(args.subjects),
        sessions=parse_sessions_arg(args.sessions),
        source_subjects=['2', '3'] if args.smoke_test else None,
        num_workers=args.num_workers, deterministic='warn', **recipe_kwargs_from_args(args))
    return configure_run(experiment, args)


def run_from_args(args):
    return build_experiment(args).run(**run_kwargs(args))


def main(argv=None):
    return run_from_args(get_parser().parse_args(argv))


if __name__ == '__main__':
    main()
