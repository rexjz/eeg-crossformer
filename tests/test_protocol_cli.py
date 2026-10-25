"""Validate the public protocol inventory, dry-run recipes and portable paths."""

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from eegcrossformer.cli import describe_run
from eegcrossformer.protocols import PROTOCOLS


class ProtocolCLITests(unittest.TestCase):
    def test_five_full_cohorts_resolve_final_model_and_calibrated_gain(self):
        expected = {
            'bcic-within-session': (25, [2, 118, 300], .025, .01),
            'openbmi-within-subject': (270, [2, 62, 625], .1, .1),
            'openbmi-loso': (54, [2, 62, 625], .05, .03),
            'seedv-within-session': (240, [2, 62, 200], .025, .01),
            'seedv-loso': (16, [2, 62, 200], .025, .01),
        }
        self.assertEqual(set(PROTOCOLS), set(expected))
        for protocol, (units, shape, radius, tau) in expected.items():
            with self.subTest(protocol=protocol):
                entry = importlib.import_module(PROTOCOLS[protocol].module)
                plan = describe_run(protocol, entry, entry.get_parser().parse_args([]))
                self.assertEqual(plan['experiment_units'], units)
                recipe = plan['recipe']
                self.assertEqual(recipe['input_shape'], shape)
                self.assertEqual(recipe['fusion_norm'], 'none')
                self.assertEqual(recipe['router_type'], 'linear')
                self.assertTrue(recipe['rtur_enabled'])
                config = recipe['rtur_loss_config']
                self.assertEqual((config['candidate_radius'], config['gain_tau']), (radius, tau))
                self.assertEqual(config['auxiliary_ce_weight'], .3)
                self.assertEqual(config['schedule_horizon_steps'], 2500)

    def test_off_keeps_auxiliary_ce_and_loso_subset_keeps_validation_cohort(self):
        protocol = 'openbmi-loso'
        entry = importlib.import_module(PROTOCOLS[protocol].module)
        args = entry.get_parser().parse_args(['--subjects', '2,4', '--rtur', 'off'])
        plan = describe_run(protocol, entry, args)
        self.assertEqual(plan['experiment_units'], 2)
        self.assertEqual(plan['selected_subjects'], ['2', '4'])
        self.assertFalse(plan['recipe']['rtur_enabled'])
        self.assertEqual(plan['recipe']['rtur_loss_config']['auxiliary_ce_weight'], .3)
        self.assertEqual(entry.build_experiment(args).num_val_subjects, 5)

    def test_seedv_subset_count_and_smoke_unit(self):
        protocol = 'seedv-within-session'
        entry = importlib.import_module(PROTOCOLS[protocol].module)
        args = entry.get_parser().parse_args(['--subjects', '1,3', '--sessions', '1', '--folds', '2,4'])
        self.assertEqual(describe_run(protocol, entry, args)['experiment_units'], 4)
        args = entry.get_parser().parse_args(['--smoke-test'])
        plan = describe_run(protocol, entry, args)
        self.assertEqual(plan['experiment_units'], 1)
        self.assertEqual(plan['selected_sessions'], ['1'])
        self.assertEqual(plan['effective_training_options']['batch_size'], 16)
        self.assertEqual(plan['effective_training_options']['max_epochs'], 1)
        self.assertEqual(plan['effective_training_options']['max_steps'], 5)

    def test_listing_uses_no_third_party_imports(self):
        package = Path(importlib.import_module('eegcrossformer').__file__).parent.parent
        environment = dict(os.environ, PYTHONPATH=str(package))
        result = subprocess.run([sys.executable, '-S', '-m', 'eegcrossformer', '--list'],
                                env=environment, capture_output=True, text=True, check=True)
        self.assertEqual(len(result.stdout.splitlines()), 5)

    def test_dry_run_honors_cli_roots_without_creating_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = Path(importlib.import_module('eegcrossformer').__file__).parent.parent
            environment = dict(os.environ, PYTHONPATH=str(package))
            for key in ('DATA_DIR', 'OUTPUT_DIR', 'BCIC34A_ROOT', 'SEEDV_ROOT',
                        'OPENBMI_CACHE_ROOT', 'bcic3_dataset4a_path', 'seedV_eeg_path', 'openbmi_path'):
                environment.pop(key, None)
            result = subprocess.run([
                sys.executable, '-m', 'eegcrossformer', 'bcic-within-session', '--dry-run',
                '--data-dir', str(root / 'datasets'), '--output-dir', str(root / 'results'),
                '--subjects', 'aa', '--folds', '1',
            ], cwd=root, env=environment, capture_output=True, text=True, check=True)
            plan = json.loads(result.stdout)
            self.assertEqual(plan['experiment_units'], 1)
            self.assertEqual(plan['paths']['bcic34a_root'], str(root / 'datasets' / 'BCIC34a'))
            self.assertEqual(plan['paths']['output_dir'], str(root / 'results'))
            self.assertFalse((root / 'datasets').exists())
            self.assertFalse((root / 'results').exists())


if __name__ == '__main__':
    unittest.main()
