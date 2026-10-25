"""Inventory of the five public experiment protocols; no dataset imports."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Protocol:
    key: str
    label: str
    module: str
    subjects: tuple[str, ...]
    sessions: tuple[str, ...] = ()
    folds: tuple[str, ...] = ()


PREFIX = "eegcrossformer.experiments.by_dataset."
PROTOCOLS = {
    "bcic-within-session": Protocol(
        "bcic-within-session", "BCIC III IVa: within-session / subject-dependent",
        PREFIX + "bci34a.SubjectDependentKFoldExperiment.EEGCrossformer",
        ("aa", "al", "av", "aw", "ay"), folds=("1", "2", "3", "4", "5")),
    "openbmi-within-subject": Protocol(
        "openbmi-within-subject", "OpenBMI: within-subject, sessions pooled",
        PREFIX + "openbmi_mi.pooled_session_kfold.EEGCrossformer",
        tuple(str(i) for i in range(1, 55)), ("1", "2"), ("1", "2", "3", "4", "5")),
    "openbmi-loso": Protocol(
        "openbmi-loso", "OpenBMI: leave-one-subject-out",
        PREFIX + "openbmi_mi.loso.EEGCrossformer",
        tuple(str(i) for i in range(1, 55)), ("1", "2")),
    "seedv-within-session": Protocol(
        "seedv-within-session", "SEED-V: within-session",
        PREFIX + "seedv.SessionDependentKFoldExperiment.EEGCrossformer",
        tuple(str(i) for i in range(1, 17)), ("1", "2", "3"), ("1", "2", "3", "4", "5")),
    "seedv-loso": Protocol(
        "seedv-loso", "SEED-V: leave-one-subject-out",
        PREFIX + "seedv.LOSOExperiment.EEGCrossformer",
        tuple(str(i) for i in range(1, 17)), ("1", "2", "3")),
}

