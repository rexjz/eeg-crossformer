"""Unified command line for the five EEGCrossformer experiments."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys

from eegcrossformer.protocols import PROTOCOLS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eegcrossformer", add_help=False,
        description="EEGCrossformer: dual-band token-wise routing with Gain RTUR.",
        epilog="Use 'eegcrossformer PROTOCOL --help' for experiment options.")
    parser.add_argument("protocol", nargs="?", choices=tuple(PROTOCOLS))
    parser.add_argument("--list", action="store_true", help="list the five protocols")
    parser.add_argument("--dry-run", action="store_true", help="print the resolved recipe without loading data or training")
    parser.add_argument("--data-dir", type=Path, help="override DATA_DIR for this process")
    parser.add_argument("--output-dir", type=Path, help="override OUTPUT_DIR for this process")
    return parser


def _subset(value, available, label):
    if value is None or str(value).lower() in ("all", "both"):
        return list(available)
    values = [str(v) for v in value] if isinstance(value, (tuple, list)) else str(value).split(",")
    values = [v.strip() for v in values]
    unknown = set(values) - set(available)
    if unknown or len(set(values)) != len(values):
        raise ValueError(f"Invalid {label}: {value}; choices are {','.join(available)}")
    return values


def describe_run(protocol, module, args):
    """Resolve actual experiment defaults without constructing a datamodule."""
    spec = PROTOCOLS[protocol]
    experiment = module.build_experiment(args)
    subjects = _subset(args.subjects, spec.subjects, "subjects")
    folds = _subset(getattr(args, "folds", None), spec.folds, "folds") if spec.folds else []
    sessions = _subset(getattr(args, "sessions", None), spec.sessions, "sessions") if spec.sessions else []
    if args.smoke_test:
        subjects, folds = [spec.subjects[0]], (["1"] if spec.folds else [])
        if protocol == "seedv-within-session":
            sessions = ["1"]
    unit_count = len(subjects) * max(len(folds), 1)
    if protocol == "seedv-within-session":
        unit_count *= len(sessions)
    from eegcrossformer import project_config
    return {
        "protocol": protocol, "description": spec.label,
        "selected_subjects": subjects, "selected_sessions": sessions,
        "selected_folds": folds, "experiment_units": unit_count,
        "recipe": experiment.get_protocol_settings(),
        "training_options": vars(args),
        "effective_training_options": {
            "batch_size": experiment.batch_size,
            "max_epochs": getattr(experiment, "max_epochs", getattr(experiment, "max_epoches", args.max_epochs)),
            "max_steps": getattr(experiment, "fixed_training_steps", None) or experiment.training_max_steps,
            "patience": experiment.patience,
            "num_workers": getattr(experiment, "num_workers", args.num_workers),
            "accelerator": experiment.accelerator,
            "precision": args.precision,
        },
        "paths": {
            "data_dir": str(project_config.DATA_DIR), "output_dir": str(project_config.OUTPUT_DIR),
            "bcic34a_root": str(project_config.BCIC34A_ROOT), "seedv_root": str(project_config.SEEDV_ROOT),
            "openbmi_cache_root": str(project_config.OPENBMI_CACHE_ROOT),
        },
    }


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        return 0
    global_args, remaining = parser.parse_known_args(argv)
    if global_args.list:
        for key, protocol in PROTOCOLS.items():
            print(f"{key:25s} {protocol.label}")
        return 0
    if global_args.protocol is None:
        parser.error("choose a protocol, or use --list")
    for value, name in ((global_args.data_dir, "DATA_DIR"), (global_args.output_dir, "OUTPUT_DIR")):
        if value is not None:
            os.environ[name] = str(value.expanduser().resolve())
    module = importlib.import_module(PROTOCOLS[global_args.protocol].module)
    entry_parser = module.get_parser()
    entry_parser.prog = f"eegcrossformer {global_args.protocol}"
    args = entry_parser.parse_args(remaining)
    if global_args.dry_run:
        try:
            plan = describe_run(global_args.protocol, module, args)
        except ValueError as exc:
            entry_parser.error(str(exc))
        print(json.dumps(plan, indent=2))
        return 0
    module.run_from_args(args)
    return 0

