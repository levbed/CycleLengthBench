#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.data import (
    DEFAULT_MAX_CYCLE_LENGTH,
    DEFAULT_MIN_CYCLE_LENGTH,
    build_cycle_examples,
    count_by_participant,
    load_combined_csv,
    load_mcphases_data,
    summarize_loaded_data,
)
from src.evaluate import evaluate_feature_table, write_csv
from src.features import build_feature_table
from src.plots import save_mae_by_track, save_predicted_vs_observed


def _print_summary(summary: dict) -> None:
    print(f"Participants: {summary['participants']}")
    print(f"Inferred complete cycles: {summary['cycles']}")
    print(f"Eligible source/target examples: {summary['eligible_examples']}")
    print("Tables inspected:")
    for table, rows in sorted(summary["table_rows"].items()):
        print(f"  - {table}: {rows} rows")
    print("Variables used:")
    if summary["tables_used"]:
        for table, variables in sorted(summary["tables_used"].items()):
            print(f"  - {table}: {', '.join(variables)}")
    else:
        print("  - none")


def _load_and_build(args: argparse.Namespace, demo: bool = False):
    loaded = load_combined_csv(args.data_file) if demo else load_mcphases_data(args.data_dir)
    examples = build_cycle_examples(
        loaded.flow_rows,
        min_cycle_length=args.min_cycle_length,
        max_cycle_length=args.max_cycle_length,
    )
    feature_table = build_feature_table(examples, loaded.measurements)
    summary = summarize_loaded_data(loaded, examples)
    return loaded, examples, feature_table, summary


def inspect_command(args: argparse.Namespace) -> None:
    _, examples, _, summary = _load_and_build(args, demo=False)
    _print_summary(summary)
    counts = count_by_participant(examples)
    if counts:
        print("Eligible examples per participant:")
        print(f"  min={min(counts.values())}, median={sorted(counts.values())[len(counts)//2]}, max={max(counts.values())}")


def evaluate_command(args: argparse.Namespace, demo: bool = False) -> None:
    _, examples, feature_table, summary = _load_and_build(args, demo=demo)
    _print_summary(summary)
    print(f"Feature variables used: {feature_table.variables_used}")
    scores, fold_scores, predictions = evaluate_feature_table(feature_table)
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "scores.csv", scores)
    write_csv(output_dir / "fold_scores.csv", fold_scores)
    write_csv(output_dir / "predictions.csv", predictions)
    save_mae_by_track(output_dir / "mae_by_track.png", scores)
    save_predicted_vs_observed(output_dir / "predicted_vs_observed.png", predictions)
    print(f"Saved results to {output_dir.resolve()}")
    print("Scores:")
    for row in scores:
        print(
            f"  - {row['track']}: MAE={float(row['mae']):.3f}, "
            f"RMSE={float(row['rmse']):.3f}, within_7_days={float(row['within_7_days_pct']):.1f}%"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="mcPHASES CycleBench")
    parser.add_argument("--min-cycle-length", type=int, default=DEFAULT_MIN_CYCLE_LENGTH)
    parser.add_argument("--max-cycle-length", type=int, default=DEFAULT_MAX_CYCLE_LENGTH)
    parser.add_argument("--output-dir", default="results")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect", help="Inspect local mcPHASES data and inferred examples.")
    inspect.add_argument("--data-dir", required=True)
    inspect.set_defaults(func=inspect_command)

    evaluate = subparsers.add_parser("evaluate", help="Run participant-disjoint benchmark on local mcPHASES data.")
    evaluate.add_argument("--data-dir", required=True)
    evaluate.set_defaults(func=lambda args: evaluate_command(args, demo=False))

    demo = subparsers.add_parser("demo", help="Run benchmark on the included synthetic CSV.")
    demo.add_argument("--data-file", required=True)
    demo.set_defaults(func=lambda args: evaluate_command(args, demo=True))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
