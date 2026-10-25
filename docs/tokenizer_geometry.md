# Tokenizer geometry and model size

Every protocol uses one physical tokenizer geometry: each token summarizes a
**300 ms** receptive field, consecutive tokens advance by a **100 ms** hop, and
adjacent tokens therefore overlap by **200 ms**. The low- and high-band branches
use identical sample-domain settings, so the paired tokens at position `n`
cover exactly the same interval with the same center. This is what the scalar
router compares when it weights the two bands token by token.

## How a token is formed

The log-power tokenizer (`LogPowerTokenizer`) maps one band of a
trial `[1, C, T]` to a token sequence `[N, D]` with `D = 20`:

1. a 1×1 spatial convolution mixes the `C` electrodes into `D` features;
2. a depthwise temporal convolution of `K` samples filters each feature, with
   **valid** padding (no samples are padded);
3. BatchNorm and GELU;
4. the squared response is averaged over a sliding window of `P` samples with
   stride `S`, then log-compressed;
5. LayerNorm and dropout.

Token `n` (zero-based) depends on samples `nS, …, nS + K + P − 2` of the
filtered input. Hence

```text
receptive field  = K + P − 1 samples
hop              = S samples
overlap          = K + P − 1 − S samples
tokens           N = floor((T − (K + P − 1)) / S) + 1
```

The geometry is fixed in time, not in samples: `K` spans about 205 ms of
convolution context and `P` is a 100 ms power-averaging window in every
protocol.

## Per-protocol settings

| Protocol | Rate | Input `T` | `K` | `P` | `S` | Receptive field | Hop | Overlap | Tokens `N` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BCIC within-session | 100 Hz | 300 (3.0 s) | 21 (210 ms) | 10 (100 ms) | 10 | 30 samples = 300 ms | 100 ms | 200 ms | 28 |
| OpenBMI within-subject | 250 Hz | 625 (2.5 s) | 51 (204 ms) | 25 (100 ms) | 25 | 75 samples = 300 ms | 100 ms | 200 ms | 23 |
| OpenBMI LOSO | 250 Hz | 625 (2.5 s) | 51 (204 ms) | 25 (100 ms) | 25 | 75 samples = 300 ms | 100 ms | 200 ms | 23 |
| SEED-V within-session | 200 Hz | 200 (1.0 s) | 41 (205 ms) | 20 (100 ms) | 20 | 60 samples = 300 ms | 100 ms | 200 ms | 8 |
| SEED-V LOSO | 200 Hz | 200 (1.0 s) | 41 (205 ms) | 20 (100 ms) | 20 | 60 samples = 300 ms | 100 ms | 200 ms | 8 |

All values are exact at the three sampling rates; no rounding is involved.

Token time spans, relative to the start of the model input window:

| Protocol | First token | Last token | Coverage |
| --- | --- | --- | --- |
| BCIC within-session | 0–300 ms (cue + 0.5–0.8 s) | 2,700–3,000 ms (cue + 3.2–3.5 s) | whole `[0.5, 3.5)` s window |
| OpenBMI (both) | 0–300 ms (cue + 1.0–1.3 s) | 2,200–2,500 ms (cue + 3.2–3.5 s) | whole `[1, 3.5)` s window |
| SEED-V (both) | 0–300 ms | 700–1,000 ms | whole 1.0 s window |

`EEGCrossformer` keyword arguments (shared by the two branches):

| Protocol | `kernel_size` | `pool_size` | `pool_stride` |
| --- | ---: | ---: | ---: |
| BCIC within-session | 21 | 10 | 10 |
| OpenBMI within-subject / LOSO | 51 | 25 | 25 |
| SEED-V within-session / LOSO | 41 | 20 | 20 |

## Model size

The deployed inference graph contains the two tokenizers, the two two-layer
GLA encoders (width 20), the linear scalar router, and the main temporal
readout. The auxiliary branch heads and the EMA teacher are used only during
training. The main readout is a max-norm temporal convolution whose kernel
spans all `N` tokens, so it is the only part whose size depends on the token
count.

### Deployed parameters

| Component | BCIC | OpenBMI (both) | SEED-V (both) |
| --- | ---: | ---: | ---: |
| Tokenizer, per branch | 2,860 | 2,340 | 2,140 |
| GLA encoder (2 layers), per branch | 7,680 | 7,680 | 7,680 |
| Linear scalar router | 41 | 41 | 41 |
| Main temporal readout (`20·N·classes + classes`) | 1,122 | 922 | 805 |
| **Deployed total** | **22,243** | **21,003** | **20,486** |

The tokenizer count is `20·C` (1×1 convolution, no bias) + `20·K` (depthwise
convolution, no bias) + 40 (BatchNorm) + 40 (LayerNorm). BCIC has `C = 118`;
OpenBMI and SEED-V have `C = 62`. The router is one `Linear(40, 1)` shared by
all tokens.

### Training-time footprint

| | BCIC | OpenBMI (both) | SEED-V (both) |
| --- | ---: | ---: | ---: |
| Auxiliary heads (low + high) | 502 + 502 | 502 + 502 | 565 + 565 |
| Trainable student | 23,247 | 22,007 | 21,616 |
| Frozen EMA teacher (a full student copy) | 23,247 | 22,007 | 21,616 |

### Inference cost per trial or window

| | BCIC | OpenBMI (both) | SEED-V (both) |
| --- | ---: | ---: | ---: |
| Convolution MACs | 1.6523 M | 2.7239 M | 0.7592 M |
| Linear MACs | 0.4133 M | 0.3395 M | 0.1181 M |
| GLA recurrent core MACs | 0.0224 M | 0.0184 M | 0.0064 M |
| **Total MACs** | **2.0880 M** | **3.0818 M** | **0.8837 M** |
| **FLOPs** (1 MAC = 2 FLOPs) | **4.1760 M** | **6.1636 M** | **1.7674 M** |

MACs count convolution and linear layers plus the GLA matrix core
(`2 · tokens · heads · d_k · d_v` per layer). Normalization, activations,
pooling, elementwise operations, filter-bank preprocessing, and the RTUR loss
are excluded. Counts were obtained with forward hooks on the released model
for a single input; the parameter totals also match the component formulas
above.
