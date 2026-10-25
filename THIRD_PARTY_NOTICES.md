# Dependency and retained-code notes

The experiment framework uses PyTorch, Lightning, TorchEEG, Flash Linear Attention, Braindecode, MOABB, MNE, NumPy, SciPy, pandas, scikit-learn, and supporting libraries declared in `pyproject.toml`. These are installed dependencies and retain their own licenses.

The source was extracted from the EEGCrossformer research repository. Dataset modules and preprocessing profiles retain their original implementations. The constrained convolution used by the classification head retains the research implementation's max-norm operation. Historical baseline experiment entries and their model implementations are excluded from this distribution.

The CPU GLA implementation (`CPUGatedLinearAttention` in `models/eegcrossformer.py`) follows Flash Linear Attention's `fla/ops/gla/naive.py` recurrence and `fla/layers/gla.py` projections, normalization, and output gate. The upstream MIT notice is included in [FLA_LICENSE.txt](src/eegcrossformer/models/FLA_LICENSE.txt), including its copyright attribution to Songlin Yang, Yu Zhang, and Zhiyuan Li.

Dataset access and redistribution are governed by the respective dataset providers. No raw EEG data, trained checkpoints, or private experiment logs are included.

The project-wide license has not been assigned in this checkout.
