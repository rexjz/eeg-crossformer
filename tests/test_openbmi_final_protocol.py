"""Regression checks for the final cue-relative OpenBMI window and cache identity."""
import importlib
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from mne.filter import resample

from eegcrossformer.protocols import PROTOCOLS
from eegcrossformer.data.dataset.OpenBMI.mi.paper_moabb_transform_cache import transform_cache_key


class OpenBMIFinalProtocolTests(unittest.TestCase):
    def test_late_window_reaches_both_actual_datamodules(self):
        for protocol in ('openbmi-within-subject', 'openbmi-loso'):
            with self.subTest(protocol=protocol):
                entry = importlib.import_module(PROTOCOLS[protocol].module)
                args = entry.get_parser().parse_args([])
                experiment = entry.build_experiment(args)
                settings = experiment.get_protocol_settings()
                self.assertEqual(settings['event_window_seconds'], [1.0, 3.5])
                self.assertEqual(settings['input_shape'], [2, 62, 625])
                self.assertEqual(settings['num_aligned_tokens'], 23)
                self.assertEqual(settings['augmentation'], 'none')
                self.assertIsNone(settings['augmentation_config'])
                self.assertEqual(args.precision, 'bf16-mixed')
                if protocol == 'openbmi-within-subject':
                    dm = experiment.get_datamodule('1', 'pooled', 1)
                    self.assertFalse(dm.include_session_id)
                else:
                    dm = experiment.get_datamodule('1')
                self.assertEqual(dm.event_window_seconds, (1.0, 3.5))
                self.assertEqual(dm.offline_transform_profile, 'crossformer_v7_2band')
                self.assertEqual(dm.channels, 'all')

    def test_late_window_cache_uses_1_16_band_and_separates_old_window(self):
        late, payload = transform_cache_key('crossformer_v7_2band', 'all', event_window_seconds=(1.0, 3.5))
        old, _ = transform_cache_key('crossformer_v7_2band', 'all', event_window_seconds=(0.0, 3.0))
        self.assertEqual(late, 'a8012b43c8fd3d0212d4edb6dc9e4213e3f8615b8f51b622b665484da94e8717')
        self.assertIn('[[1, 16], [16, 40]]', json.dumps(payload['transform']))
        self.assertNotEqual(late, old)
        self.assertEqual(payload['expected_shape'], [2, 62, 625])
        self.assertEqual(payload['event_window_seconds'], [1.0, 3.5])

    def test_raw_moabb_builder_crops_correct_samples_before_resampling(self):
        dataset_module = importlib.import_module('eegcrossformer.data.dataset.OpenBMI.mi.OpenBMIMIDataset')
        # Index-valued epochs make an off-by-one or a center crop observable.
        epoch = np.broadcast_to(np.arange(4000, dtype=float), (62, 4000)).copy()
        trials = [(epoch, 0), (epoch + 10000, 1)]
        raw_dataset = SimpleNamespace(datasets=[SimpleNamespace(raw=SimpleNamespace(info={'sfreq': 1000}))])
        windows = Mock()
        windows.split.return_value = {'0': trials}
        api = (Mock(return_value=raw_dataset), Mock(), Mock(), Mock(), Mock(return_value=windows))
        with patch.object(dataset_module, '_get_braindecode_moabb_api', return_value=api), \
             patch.object(dataset_module, '_select_moabb_session_split', return_value=trials), \
             patch('mne.filter.resample', wraps=resample) as resampler:
            data, labels = dataset_module._build_openbmi_paper_moabb_subject_session('1', 1, 'all', (1.0, 3.5))
        cropped = resampler.call_args.args[0]
        self.assertEqual(cropped.shape, (2, 62, 2500))
        self.assertEqual(cropped[0, 0, 0], 1000)
        self.assertEqual(cropped[0, 0, -1], 3499)
        self.assertEqual(data.shape, (2, 1, 62, 625))
        np.testing.assert_array_equal(labels, [0, 1])
        self.assertEqual(data.dtype, np.float32)


if __name__ == '__main__':
    unittest.main()
