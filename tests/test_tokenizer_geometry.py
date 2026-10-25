"""The five entries share one 300-ms / 100-ms tokenizer geometry."""

import importlib
import os
import tempfile
import unittest
from unittest.mock import patch

from eegcrossformer.experiments.utils.final_recipe import unified_tokenizer_spec
from eegcrossformer.protocols import PROTOCOLS


EXPECTED = {  # protocol: (sampling rate, input samples, K, P, S, tokens, deployed parameters)
    'bcic-within-session': (100, 300, 21, 10, 10, 28, 22243),
    'openbmi-within-subject': (250, 625, 51, 25, 25, 23, 21003),
    'openbmi-loso': (250, 625, 51, 25, 25, 23, 21003),
    'seedv-within-session': (200, 200, 41, 20, 20, 8, 20486),
    'seedv-loso': (200, 200, 41, 20, 20, 8, 20486),
}
TRAINING_ONLY = ('_ema_teacher.', 'head_l.', 'head_h.')


class TokenizerGeometryTests(unittest.TestCase):
    def test_every_entry_uses_identical_branch_grids_at_300_100_ms(self):
        for protocol, (rate, samples, kernel, pool, stride, tokens, _) in EXPECTED.items():
            with self.subTest(protocol=protocol):
                entry = importlib.import_module(PROTOCOLS[protocol].module)
                experiment = entry.build_experiment(entry.get_parser().parse_args(['--model-backend', 'gla_cpu']))
                settings = experiment.get_protocol_settings()
                self.assertEqual(experiment.tokenizer_kwargs,
                                 {'kernel_size': kernel, 'pool_size': pool, 'pool_stride': stride})
                self.assertEqual(experiment.input_shape[-1], samples)
                self.assertEqual(settings['sampling_rate_hz'], rate)
                self.assertEqual(1000 * (kernel + pool - 1) / rate, 300)
                self.assertEqual(1000 * stride / rate, 100)
                self.assertEqual(settings['token_receptive_field_ms'], 300)
                self.assertEqual(settings['token_hop_ms'], 100)
                self.assertEqual(settings['num_aligned_tokens'], tokens)
                self.assertEqual((samples - (kernel + pool - 1)) // stride + 1, tokens)

    def test_deployed_parameter_counts_match_documentation(self):
        for protocol, (*_, tokens, deployed) in EXPECTED.items():
            with self.subTest(protocol=protocol):
                entry = importlib.import_module(PROTOCOLS[protocol].module)
                experiment = entry.build_experiment(entry.get_parser().parse_args(['--model-backend', 'gla_cpu']))
                model = experiment.create_model().model
                count = sum(p.numel() for name, p in model.named_parameters() if not name.startswith(TRAINING_ONLY))
                self.assertEqual(count, deployed)
                self.assertEqual(model.head.conv.kernel_size, (1, tokens))

    def test_spec_rejects_windows_shorter_than_one_receptive_field(self):
        with self.assertRaises(ValueError):
            unified_tokenizer_spec(200, 59)

    def test_seedv_loso_seeds_each_held_out_subject_independently(self):
        entry = importlib.import_module(PROTOCOLS['seedv-loso'].module)
        module = importlib.import_module(
            'eegcrossformer.experiments.by_dataset.seedv.LOSOExperiment.SEEDVLOSOExperiment')

        class Stop(Exception):
            pass

        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'OUTPUT_DIR': directory}):
            experiment = entry.build_experiment(entry.get_parser().parse_args(['--subjects', '7']))
            experiment.log_base_dir = directory
            seeds = []
            with patch.object(module.L, 'seed_everything', side_effect=lambda seed, workers=False: seeds.append(seed)), \
                    patch.object(type(experiment), 'create_model', side_effect=Stop):
                with self.assertRaises(Stop):
                    experiment.run()
            self.assertEqual(seeds, [42 + 7 - 1])


if __name__ == '__main__':
    unittest.main()
