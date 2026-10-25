# EEGCrossformer

EEGCrossformer decodes EEG through low- and high-frequency branches and learns a scalar fusion gate for each token within a trial. The model uses a linear router, applies no normalization after fusion, and is trained with Gain RTUR. The complete model and its training objective are in a single file, [`src/eegcrossformer/models/eegcrossformer.py`](src/eegcrossformer/models/eegcrossformer.py).

This repository contains the final model, shared data/training framework, and five main experiment protocols:

| Command | Dataset | Evaluation | Full run |
| --- | --- | --- | --- |
| `bcic-within-session` | BCI Competition III, dataset IVa | Within-session / subject-dependent five-fold | 5 subjects × 5 folds |
| `openbmi-within-subject` | OpenBMI / Lee2019 MI | Both sessions pooled; subject-dependent five-fold | 54 subjects × 5 folds |
| `openbmi-loso` | OpenBMI / Lee2019 MI | Leave-one-subject-out, both sessions | 54 held-out subjects |
| `seedv-within-session` | SEED-V | Separate subject/session five-fold | 16 subjects × 3 sessions × 5 folds |
| `seedv-loso` | SEED-V | Leave-one-subject-out, all sessions pooled | 16 held-out subjects |

All five entries share one tokenizer geometry: a 300 ms receptive field and a 100 ms hop, so adjacent tokens overlap by 200 ms, with identical valid-convolution grids in the low- and high-band branches. The deployed model has 20,486–22,243 parameters depending on the protocol. See [docs/tokenizer_geometry.md](docs/tokenizer_geometry.md) for the per-protocol grids, token spans, parameter counts, and MACs. Protocol definitions and defaults are in [docs/protocols.md](docs/protocols.md).

Both OpenBMI entries follow the final cue-aligned **[1, 3.5) s** protocol: **625 samples at 250 Hz, NoAug, bf16 mixed**. Within-subject pools both sessions for joint session/class-stratified five-fold evaluation; LOSO uses 48 training, 5 validation, and 1 held-out subject. Model dropout, auxiliary CE, and Gain RTUR remain active.

## Installation

Use Python 3.11. The GPU implementation uses PyTorch 2.7.1 and Flash Linear Attention 0.5.0. Install the PyTorch build appropriate for your CUDA environment before the project:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[gpu]'
```

For CPU validation, install `python -m pip install -e .` and select `--model-backend gla_cpu --accelerator cpu --precision 32-true`. The CPU implementation is intended for short functional checks.

## Data paths

Copy `.env.example` to `.env` and set the locations of your datasets and outputs. Paths are resolved from the current working directory. Dataset files and preprocessing caches stay outside the Python package.

```dotenv
DATA_DIR=./data
OUTPUT_DIR=./outputs
MNE_DATA=./data/moabb
```

The default layout is:

```text
data/
├── BCIC34a/
│   ├── data_set_IVa_aa.mat
│   ├── true_labels_aa.mat
│   └── ...                    # aa, al, av, aw, ay: data and true labels
├── SEED-V/
│   └── EEG_raw/*.cnt
├── moabb/                     # OpenBMI raw data managed by MNE/MOABB
└── OpenBMI/                   # derived OpenBMI preprocessing cache
```

BCIC uses all true-labeled trials from dataset IVa. OpenBMI is loaded through MOABB's `Lee2019_MI` dataset; its loader may download missing raw files into `MNE_DATA`. `OPENBMI_CACHE_ROOT` controls the derived cache, rather than the raw-download directory. Obtain SEED-V's raw EEG files through the dataset provider.

For existing data locations, set `BCIC34A_ROOT`, `SEEDV_ROOT`, and `OPENBMI_CACHE_ROOT` individually. You can override the two shared roots for one command with `--data-dir PATH --output-dir PATH`.

## Run the main experiments

```bash
eegcrossformer --list
eegcrossformer bcic-within-session
eegcrossformer openbmi-within-subject
eegcrossformer openbmi-loso
eegcrossformer seedv-within-session
eegcrossformer seedv-loso
```

Each command runs its full cohort by default. `python -m eegcrossformer` provides the same interface. Use `eegcrossformer PROTOCOL --help` for protocol options, and `--dry-run` to inspect the resolved model, loss, selected units, and paths without loading data.

```bash
eegcrossformer bcic-within-session --dry-run
eegcrossformer openbmi-within-subject --subjects 1,2 --folds 1 --run-tag example
eegcrossformer seedv-within-session --subjects 1,2,3,4,5 --sessions 1 --folds all
eegcrossformer openbmi-loso --subjects 1,2 --devices '[0]'
```

In LOSO, `--subjects` selects the held-out folds to run; it leaves the training/validation cohort intact. `--devices 1` selects one GPU, and `--devices '[0]'` selects GPU index 0. `--subjects`, `--sessions`, and `--folds` accept comma-separated IDs where applicable.

`--smoke-test` runs the first experimental unit with a short training budget using real data. For a CPU check:

```bash
eegcrossformer bcic-within-session --smoke-test \
  --model-backend gla_cpu --accelerator cpu --precision 32-true --num-workers 0
```

Use `--rtur off` for a Gain RTUR switch ablation. It keeps the architecture and auxiliary branch cross-entropies and sets the routing-shape weight to zero. The default schedule horizon is 2,500 updates; `--rtur-horizon 0` scales the schedule to the planned training budget.

## Outputs and monitoring

Runs write to `OUTPUT_DIR`, including per-unit `metrics.csv`, protocol metadata, checkpoints, test summaries, and metric plots. The fixed-budget OpenBMI within-subject protocol evaluates the final training weights and does not select a checkpoint on its outer test set.

RTUR monitoring is enabled by default. Each unit records `rtur_monitoring/steps.csv`, `epochs.csv`, `probe.csv`, `metadata.json`, and `probe_tokens.csv`. These capture gate statistics, Gain targets/confidence, loss components, router-gradient diagnostics, and deterministic training probes. Token vectors are recorded at the first epoch and every ten epochs. `--no-monitoring` disables these additional records.

## Code and validation

```text
src/eegcrossformer/
├── models/                    # final model, GLA backends, Lightning wrapper
├── data/                      # dataset modules, preprocessing, normalization, SDBA
├── experiments/by_dataset/    # five final entries and their protocol bases
├── experiments/utils/         # shared final recipe, RTUR task and monitoring
├── cli.py                     # unified command line
└── project_config.py          # portable paths and cohort definitions
```

See [docs/architecture.md](docs/architecture.md) for the forward and training paths.

```bash
python -m unittest discover -s tests -v
```

## License

The project license is pending selection. Dependency and retained-code notes are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
