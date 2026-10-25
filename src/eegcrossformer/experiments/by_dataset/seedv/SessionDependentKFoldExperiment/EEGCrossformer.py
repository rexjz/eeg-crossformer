"""EEGCrossformer on SEED-V: all subjects, sessions and within-session folds."""
from argparse import ArgumentParser
from functools import partial
import torcheeg.transforms as transforms

from eegcrossformer.data.dataset.seedV.SEEDVSessionDependentDataModule import SEEDVSessionDependentDataModule
from eegcrossformer.data.transforms.batch_transforms.synchronized_dual_band_augmentation import SDBAConfig, SynchronizedDualBandAugmentation
from eegcrossformer.data.transforms.fb_transform import TrailFilteringBankTransform
from .SEEDVSessionDependentKFoldExperiment import SEEDVSessionDependentKFoldExperiment, parse_seedv_subjects_arg, parse_seedv_sessions_arg, parse_seedv_folds_arg
from eegcrossformer.experiments.utils.final_recipe import FinalRecipeMixin, add_common_args, recipe_kwargs_from_args, configure_run, run_kwargs, unified_tokenizer_spec, center_crop_batch


PROTOCOL = 'seedv_within_session'
TOKENIZER = unified_tokenizer_spec(200, 200)


class EEGCrossformerExperiment(FinalRecipeMixin, SEEDVSessionDependentKFoldExperiment):
    input_shape = [2, 62, 200]
    num_classes = 5
    tokenizer_kwargs = TOKENIZER['model_kwargs']

    def __init__(self, *args, **kwargs):
        self.sdba_config = SDBAConfig(max_shift_points=20)
        self.train_augmentation = SynchronizedDualBandAugmentation(self.sdba_config)
        super().__init__(*args, **kwargs)

    def get_datamodule(self, subject_code, session_id, fold, num_workers=4):
        return SEEDVSessionDependentDataModule(
            batch_size=self.batch_size, num_workers=num_workers,
            subject_code=subject_code, session_id=session_id,
            kfold=self.num_folds, test_fold=fold, kfold_seed=self.experiment_seed,
            offline_transform=transforms.Compose([TrailFilteringBankTransform(freq_bands=[[1, 16], [16, 40]], sfreq=200)]),
            online_transform=transforms.ToTensor(), normalization='z2', get_item_style='tuple')

    def get_protocol_settings(self):
        return {**TOKENIZER['protocol_settings'], **self.final_protocol_settings(), 'protocol_id': PROTOCOL,
                'input_shape': self.input_shape, 'sampling_rate_hz': 200,
                'crossformer_branch_bands': [[1, 16], [16, 40]],
                'normalization': 'z2', 'augmentation': 'SDBA',
                'augmentation_config': self.sdba_config.to_dict(),
                'validation_test_augmentation': 'none'}

    def create_model(self, model_kwargs=None):
        def augment(batch):
            return (self.train_augmentation(batch[0], batch[1]), batch[1], *batch[2:])
        return self.build_model(model_kwargs, train_transform=augment,
                                eval_transform=partial(center_crop_batch, input_time=200))


def get_parser():
    return add_common_args(ArgumentParser(description=__doc__), batch_size=128,
        max_epochs=500, patience=50, precision='bf16-mixed', num_workers=4,
        radius=.025, tau=.01, folds=True, sessions=True)


build_parser = get_parser


def build_experiment(args):
    experiment = EEGCrossformerExperiment(
        batch_size=16 if args.smoke_test else args.batch_size,
        max_epochs=1 if args.smoke_test else args.max_epochs,
        patience=3 if args.smoke_test else args.patience,
        subjects=['1'] if args.smoke_test else parse_seedv_subjects_arg(args.subjects),
        sessions=['1'] if args.smoke_test else parse_seedv_sessions_arg(args.sessions),
        folds=[1] if args.smoke_test else parse_seedv_folds_arg(args.folds),
        training_seed=args.seed, deterministic='warn', **recipe_kwargs_from_args(args))
    return configure_run(experiment, args)


def run_from_args(args):
    return build_experiment(args).run(num_workers=args.num_workers, **run_kwargs(args))


def main(argv=None):
    return run_from_args(get_parser().parse_args(argv))


if __name__ == '__main__':
    main()
