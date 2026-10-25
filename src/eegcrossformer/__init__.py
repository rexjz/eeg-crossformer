"""EEGCrossformer: dual-band EEG decoding with token-wise routing."""

__version__ = "0.1.0"
__all__ = ["EEGCrossformer", "GainRTURLossConfig", "__version__"]


def __getattr__(name):
    if name in ("EEGCrossformer", "GainRTURLossConfig"):
        from eegcrossformer.models.eegcrossformer import EEGCrossformer, GainRTURLossConfig
        return {"EEGCrossformer": EEGCrossformer, "GainRTURLossConfig": GainRTURLossConfig}[name]
    raise AttributeError(name)

