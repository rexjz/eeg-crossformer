# Main experiment protocols

All five entries use the released EEGCrossformer: two log-power tokenizers, two two-layer GLA branch encoders with width 20, a linear scalar token router, and no post-fusion LayerNorm. Adam uses learning rate `3e-4` and weight decay `0.05`. Each experimental unit trains a new model.

## Data and token grids

Shapes below omit the batch dimension and follow `[band, channel, time]`. The two branches receive separately filtered views of the same trial.

| Protocol | Shape | Sampling rate | Low / high bands (Hz) | Temporal kernels | Pool / stride | Tokens |
| --- | --- | --- | --- | --- | --- | --- |
| BCIC within-session | `[2,118,300]` | 100 Hz | 1–16 / 16–40 | 21 / 21 | 10 / 10 | 28 |
| OpenBMI within-subject | `[2,62,625]` | 250 Hz | 1–16 / 16–40 | 51 / 51 | 25 / 25 | 23 |
| OpenBMI LOSO | `[2,62,625]` | 250 Hz | 1–16 / 16–40 | 51 / 51 | 25 / 25 | 23 |
| SEED-V within-session | `[2,62,200]` | 200 Hz | 1–16 / 16–40 | 41 / 41 | 20 / 20 | 8 |
| SEED-V LOSO | `[2,62,200]` | 200 Hz | 1–16 / 16–40 | 41 / 41 | 20 / 20 | 8 |

All entries share one tokenizer geometry: valid temporal convolution, a 300 ms receptive field, and a 100 ms hop, so adjacent tokens overlap by 200 ms. Both branches use identical kernel, pool, and stride settings. [docs/tokenizer_geometry.md](tokenizer_geometry.md) gives the derivation, token time spans, parameter counts, and MACs.

## Splits and training

| Protocol | Training / validation / test | Augmentation | Batch | Budget | Precision | Selection |
| --- | --- | --- | --- | --- | --- | --- |
| BCIC within-session | Outer five-fold within each subject; train/validation split inside outer training pool | SDBA | 16 | 250 epochs; no early stopping | bf16 mixed | Best validation accuracy |
| OpenBMI within-subject | Joint session/class-stratified five-fold, both sessions pooled; 160 train / 40 test trials per fold | None | 16 | 250 epochs (10 updates per epoch); no early stopping | bf16 mixed | Final weights; no validation |
| OpenBMI LOSO | 48 train / 5 validation / 1 test subjects; both sessions | None | 128 | Up to 500 epochs; patience 50 | bf16 mixed | Best validation accuracy |
| SEED-V within-session | Five-fold windows within each subject/session; validation split from outer training pool | SDBA | 128 | Up to 500 epochs; patience 50 | bf16 mixed | Best validation accuracy |
| SEED-V LOSO | 12 train / 3 validation / 1 test subjects; all three sessions | None | 128 | Up to 500 epochs; patience 50 | bf16 mixed | Best whole-epoch validation accuracy |

BCIC is **BCI Competition III IVa**, with subjects `aa,al,av,aw,ay`. All 280 true-labeled trials per subject participate in the random five-fold evaluation. The outer training pool is split 80/20 into training and validation. Common-average referencing and per-trial `MeanStdNormalize(axis=0)` precede the model. Each trial is loaded as the cue-aligned [0, 3.5) s segment (350 samples) and filtered as a whole; the model then receives the fixed cue-relative **[0.5, 3.5) s** window (300 samples) in training, validation and test. There is no random cropping.

OpenBMI processes its two sessions separately before pooling them. Both released entries use the fixed cue-aligned half-open window **[1, 3.5) seconds**, yielding 625 samples at 250 Hz, with **NoAug** and **bf16 mixed** precision. NoAug removes SDBA, temporal shift, channel dropout, gain perturbation, mixup, and random cropping; model dropout and auxiliary losses remain active. LOSO keeps all trials from the held-out subject outside training and validation. The raw-data loader is MOABB `Lee2019_MI`: raw continuous EEG, all 62 EEG channels, μV conversion, 0–40 Hz filtering, continuous exponential standardization, cue-relative crop, and resampling to 250 Hz. It does not use z2 normalization.

Both OpenBMI entries use the shared 300/100 ms grid: 51-sample kernels and 23 aligned tokens. The window settings are part of the preprocessing cache key, so caches for different windows do not collide.

SEED-V uses non-overlapping one-second EEG windows and z2 normalization. Within-session folds split windows, so windows from a movie can occur in multiple folds. The training/validation pool shares its normalization statistics, while the test pool has its own unlabeled statistics. LOSO computes normalization per subject, including unlabeled statistics from the held-out subject. These definitions preserve the research protocols.

## Gain RTUR defaults

| Protocol | Candidate radius | Gain temperature `tau` |
| --- | --- | --- |
| BCIC within-session | 0.025 | 0.01 |
| OpenBMI within-subject | 0.10 | 0.10 |
| OpenBMI LOSO | 0.05 | 0.03 |
| SEED-V within-session | 0.025 | 0.01 |
| SEED-V LOSO | 0.025 | 0.01 |

These are per-protocol defaults; `--gain-radius` and `--gain-tau` override them.

Common settings are three candidate updates, candidate step size `0.1`, gate epsilon `1e-4`, auxiliary CE weight `0.3`, routing-shape weight `0.1`, Smooth L1 beta `0.25`, and ranking weight `0`. Supervision uses all trials and aligned token targets. The schedule has 10% warmup and 10% ramp over 2,500 updates; EMA decay moves from `0.99` to `0.999`. The loss is:

```text
CE_main + 0.3 × (CE_low + CE_high) + schedule × 0.1 × Gain_shape
```

The candidate gate is searched with the frozen EMA teacher. Gain RTUR supplies a direct gradient to the router through detached student branch features. Auxiliary branch CE remains active when `--rtur off`.

## Seeds and subsets

The split seed defaults to 42. The training base seed is 43 for BCIC and 42 for the other protocols. These are one deterministic run per unit, rather than repeated-seed experiments.

| Protocol | Training seed |
| --- | --- |
| BCIC | `base + canonical_subject_index × 5 + fold − 1` |
| OpenBMI within-subject | `base + (subject − 1) × 5 + fold − 1` |
| OpenBMI LOSO | `base + held_out_subject − 1` |
| SEED-V within-session | `base + (subject − 1) × 15 + (session − 1) × 5 + fold − 1` |
| SEED-V LOSO | `base + held_out_subject − 1` |

`--subjects` selects experiment units. LOSO still constructs the full remaining cohort for training/validation. Fold IDs are 1–5; OpenBMI session IDs are 1–2 and SEED-V IDs are 1–3. OpenBMI LOSO defaults to both sessions and accepts `--sessions` to select a single session; SEED-V LOSO always pools all three sessions. `--dry-run` reports the selected unit count before training.

## Entry modules

| CLI protocol | Python module under `eegcrossformer.experiments.by_dataset` |
| --- | --- |
| `bcic-within-session` | `bci34a.SubjectDependentKFoldExperiment.EEGCrossformer` |
| `openbmi-within-subject` | `openbmi_mi.pooled_session_kfold.EEGCrossformer` |
| `openbmi-loso` | `openbmi_mi.loso.EEGCrossformer` |
| `seedv-within-session` | `seedv.SessionDependentKFoldExperiment.EEGCrossformer` |
| `seedv-loso` | `seedv.LOSOExperiment.EEGCrossformer` |

Every entry exposes `get_parser`, `build_experiment`, `run_from_args`, and `main`. Dataset splitting and logging remain in generic protocol base classes; model and loss construction are shared in `experiments/utils/final_recipe.py`.
