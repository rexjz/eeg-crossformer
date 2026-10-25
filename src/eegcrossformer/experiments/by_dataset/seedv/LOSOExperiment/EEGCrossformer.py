"""EEGCrossformer on SEED-V leave-one-subject-out: all sessions pooled, NoAug."""
from argparse import ArgumentParser
from functools import partial
import lightning as L
import torcheeg.transforms as transforms

from eegcrossformer.data.dataset.seedV.SEEDVLOSODataModule import SEEDVLOSODataModule
from eegcrossformer.data.transforms.fb_transform import TrailFilteringBankTransform
from .SEEDVLOSOExperiment import SEEDVLOSOExperiment
from eegcrossformer.experiments.by_dataset.seedv.SessionDependentKFoldExperiment.SEEDVSessionDependentKFoldExperiment import parse_seedv_subjects_arg
from eegcrossformer.experiments.utils.final_recipe import FinalRecipeMixin, add_common_args, recipe_kwargs_from_args, configure_run, run_kwargs, center_crop_batch, unified_tokenizer_spec


PROTOCOL = 'seedv_loso'
TOKENIZER = unified_tokenizer_spec(200, 200)


class EEGCrossformerExperiment(FinalRecipeMixin, SEEDVLOSOExperiment):
    input_shape = [2, 62, 200]
    num_classes = 5
    tokenizer_kwargs = TOKENIZER['model_kwargs']

    def __init__(self, *args, num_workers=4, **kwargs):
        self.num_workers = num_workers
        super().__init__(*args, **kwargs)

    def get_protocol_settings(self):
        return {**TOKENIZER['protocol_settings'], **self.final_protocol_settings(),
                'protocol_id': PROTOCOL, 'input_shape': self.input_shape,
                'sampling_rate_hz': 200, 'crossformer_branch_bands': [[1, 16], [16, 40]],
                'normalization': 'z2', 'augmentation': 'none',
                'evaluation_protocol': 'leave-one-subject-out; all three sessions pooled',
                'validation_subjects_per_fold': self.num_val_subjects,
                'validation_test_augmentation': 'none',
                'training_seed_policy': 'training_seed + leftout_subject - 1, set before each model'}

    def get_datamodule(self, subject_code):
        return SEEDVLOSODataModule(
            leftout_subject_code=subject_code, num_val_subjects=self.num_val_subjects,
            batch_size=self.batch_size, random_seed=self.random_seed,
            num_workers=self.num_workers,
            offline_transform=transforms.Compose([TrailFilteringBankTransform(freq_bands=[[1, 16], [16, 40]], sfreq=200)]),
            online_transform=transforms.ToTensor(), normalization='z2', get_item_style='tuple')

    def create_model(self, model_kwargs=None):
        return self.build_model(model_kwargs, eval_transform=partial(center_crop_batch, input_time=200))


def get_parser():
    return add_common_args(ArgumentParser(description=__doc__), batch_size=128,
        max_epochs=500, patience=50, precision='bf16-mixed', num_workers=4,
        radius=.025, tau=.01, val_subjects=3)


build_parser = get_parser


def build_experiment(args):
    experiment = EEGCrossformerExperiment(
        batch_size=16 if args.smoke_test else args.batch_size,
        max_epoches=1 if args.smoke_test else args.max_epochs,
        early_stop=not args.smoke_test, patience=3 if args.smoke_test else args.patience,
        random_seed=args.split_seed,
        num_val_subjects=1 if args.smoke_test else args.num_val_subjects,
        subjects=['1'] if args.smoke_test else parse_seedv_subjects_arg(args.subjects),
        num_workers=0 if args.smoke_test else args.num_workers,
        **recipe_kwargs_from_args(args))
    experiment.training_seed = args.seed
    return configure_run(experiment, args)


def run_from_args(args):
    L.seed_everything(args.seed, workers=True)
    return build_experiment(args).run(**run_kwargs(args))


def main(argv=None):
    return run_from_args(get_parser().parse_args(argv))


if __name__ == '__main__':
    main()
