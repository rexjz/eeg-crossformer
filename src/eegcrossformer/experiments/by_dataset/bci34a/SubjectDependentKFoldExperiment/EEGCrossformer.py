"""EEGCrossformer on BCIC34a: subject-dependent five-fold evaluation."""
from argparse import ArgumentParser
import torcheeg.transforms as transforms

from eegcrossformer.data.dataset.bci_competition.BCIC34aSessionDependentKFoldDataModule import BCIC34aSessionDependentKFoldDataModule
from eegcrossformer.data.transforms.batch_transforms.synchronized_dual_band_augmentation import SDBAConfig, SynchronizedDualBandAugmentation
from eegcrossformer.data.transforms.fb_transform import TrailFilteringBankTransform
from .BCIC34aSubjectDependentKfoldExperiment import BCIC34aSubjectDependentKfoldExperiment, parse_bci34a_subjects_arg, parse_bci34a_folds_arg
from eegcrossformer.experiments.utils.final_recipe import FinalRecipeMixin, add_common_args, recipe_kwargs_from_args, configure_run, run_kwargs, unified_tokenizer_spec


PROTOCOL = 'bcic34a_within_session'
SAMPLING_RATE_HZ = 100
# Cue-relative model window inside the loaded [0, 3.5) s trial.
WINDOW_SECONDS = (0.5, 3.5)
WINDOW = slice(round(WINDOW_SECONDS[0] * SAMPLING_RATE_HZ), round(WINDOW_SECONDS[1] * SAMPLING_RATE_HZ))
TOKENIZER = unified_tokenizer_spec(SAMPLING_RATE_HZ, WINDOW.stop - WINDOW.start)


def crop_window(batch):
    return (batch[0][..., WINDOW], batch[1], *batch[2:])


class EEGCrossformerExperiment(FinalRecipeMixin, BCIC34aSubjectDependentKfoldExperiment):
    monitor_metric = 'val_accuracy'
    input_shape = [2, 118, WINDOW.stop - WINDOW.start]
    num_classes = 2
    augmentation_protocol = 'sdba_shape_preserving'
    tokenizer_kwargs = TOKENIZER['model_kwargs']

    def __init__(self, *args, **kwargs):
        self.sdba_config = SDBAConfig(max_shift_points=10)
        self.train_augmentation = SynchronizedDualBandAugmentation(self.sdba_config)
        super().__init__(*args, **kwargs)

    def get_online_transform(self):
        return transforms.Compose([transforms.MeanStdNormalize(axis=0), TrailFilteringBankTransform(freq_bands=[[1, 16], [16, 40]], sfreq=100)])

    def get_datamodule(self, fold, subject_code, num_workers=0):
        return BCIC34aSessionDependentKFoldDataModule(
            fold=fold, batch_size=self.batch_size, subject_code=subject_code,
            kfold=self.num_folds, random_seed=self.experiment_seed,
            num_workers=num_workers, get_item_style='tuple',
            small_training_mode=False, transform=self.get_online_transform())

    def get_protocol_settings(self):
        return {**TOKENIZER['protocol_settings'], **self.final_protocol_settings(), 'protocol_id': PROTOCOL,
                'input_shape': self.input_shape, 'sampling_rate_hz': SAMPLING_RATE_HZ,
                'event_window_seconds': list(WINDOW_SECONDS),
                'temporal_crop': f'fixed cue-relative [{WINDOW_SECONDS[0]:g}, {WINDOW_SECONDS[1]:g}) s; '
                                 'filtering runs on the full [0, 3.5) s trial before the crop',
                'crossformer_branch_bands': [[1, 16], [16, 40]],
                'normalization': 'per-trial MeanStdNormalize(axis=0)',
                'augmentation': 'SDBA after the fixed window crop',
                'augmentation_config': self.sdba_config.to_dict(),
                'validation_test_augmentation': 'none'}

    def create_model(self, model_kwargs=None):
        def augment(batch):
            batch = crop_window(batch)
            return (self.train_augmentation(batch[0], batch[1]), batch[1], *batch[2:])
        return self.build_model(model_kwargs, train_transform=augment, eval_transform=crop_window)


def get_parser():
    return add_common_args(ArgumentParser(description=__doc__), batch_size=16,
        max_epochs=250, patience=251, precision='bf16-mixed', num_workers=0,
        seed=43, radius=.025, tau=.01, folds=True)


build_parser = get_parser


def build_experiment(args):
    experiment = EEGCrossformerExperiment(
        batch_size=16 if args.smoke_test else args.batch_size,
        max_epochs=1 if args.smoke_test else args.max_epochs,
        patience=2 if args.smoke_test else args.patience,
        subjects=['aa'] if args.smoke_test else parse_bci34a_subjects_arg(args.subjects),
        folds=[1] if args.smoke_test else parse_bci34a_folds_arg(args.folds),
        training_seed=args.seed, deterministic='warn', **recipe_kwargs_from_args(args))
    return configure_run(experiment, args)


def run_from_args(args):
    return build_experiment(args).run(num_workers=args.num_workers, **run_kwargs(args))


def main(argv=None):
    return run_from_args(get_parser().parse_args(argv))


if __name__ == '__main__':
    main()
