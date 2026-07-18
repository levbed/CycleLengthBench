from __future__ import annotations

import csv
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FeatureTable, feature_tracks


METRIC_NAMES = [
    "mae",
    "rmse",
    "median_absolute_error",
    "within_3_days_pct",
    "within_7_days_pct",
    "mean_signed_error",
]

DEFAULT_RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
DEFAULT_BOOTSTRAP_REPLICATES = 2000
DEFAULT_RANDOM_SEED = 2026


def median(values: Iterable[float]) -> float:
    clean = [float(value) for value in values if not math.isnan(float(value))]
    return statistics.median(clean) if clean else 0.0


def _split_count(group_count: int) -> int:
    if group_count < 2:
        raise ValueError("Participant-disjoint evaluation requires at least two participants.")
    if group_count >= 5:
        return 5
    if group_count >= 3:
        return 3
    return group_count


def group_kfold_indices(
    rows: list[dict[str, Any]], group_key: str = "participant_id"
) -> list[tuple[list[int], list[int]]]:
    groups = np.asarray([str(row[group_key]) for row in rows])
    splitter = GroupKFold(n_splits=_split_count(len(set(groups))))
    placeholder = np.zeros(len(rows))
    return [
        (train_indices.tolist(), test_indices.tolist())
        for train_indices, test_indices in splitter.split(placeholder, groups=groups)
    ]


def assert_participant_disjoint(
    rows: list[dict[str, Any]], folds: list[tuple[list[int], list[int]]]
) -> None:
    for fold_idx, (train_indices, test_indices) in enumerate(folds, start=1):
        train_groups = {rows[idx]["participant_id"] for idx in train_indices}
        test_groups = {rows[idx]["participant_id"] for idx in test_indices}
        overlap = train_groups & test_groups
        if overlap:
            raise AssertionError(f"Fold {fold_idx} has overlapping participants: {sorted(overlap)}")


def _as_float(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _matrix(rows: list[dict[str, Any]], indices: list[int], feature_names: list[str]) -> np.ndarray:
    return np.asarray(
        [[_as_float(rows[idx].get(name)) for name in feature_names] for idx in indices],
        dtype=float,
    )


def _pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def select_ridge_alpha(
    rows: list[dict[str, Any]],
    train_indices: list[int],
    feature_names: list[str],
    alphas: Iterable[float] = DEFAULT_RIDGE_ALPHAS,
) -> tuple[float, dict[float, float]]:
    groups = [str(rows[idx]["participant_id"]) for idx in train_indices]
    unique_groups = sorted(set(groups))
    candidates = sorted({float(alpha) for alpha in alphas})
    if not candidates:
        raise ValueError("At least one Ridge alpha is required.")
    if len(unique_groups) < 2:
        return candidates[-1], {candidates[-1]: math.nan}

    inner_splits = min(3, len(unique_groups))
    splitter = GroupKFold(n_splits=inner_splits)
    x_all = _matrix(rows, train_indices, feature_names)
    y_all = np.asarray([float(rows[idx]["target_cycle_length"]) for idx in train_indices])
    group_array = np.asarray(groups)
    mean_mae: dict[float, float] = {}

    for alpha in candidates:
        fold_mae: list[float] = []
        for inner_train, inner_validation in splitter.split(x_all, y_all, groups=group_array):
            model = _pipeline(alpha)
            model.fit(x_all[inner_train], y_all[inner_train])
            predictions = model.predict(x_all[inner_validation])
            fold_mae.append(float(np.mean(np.abs(predictions - y_all[inner_validation]))))
        mean_mae[alpha] = statistics.mean(fold_mae)

    selected = min(candidates, key=lambda alpha: (mean_mae[alpha], -alpha))
    return selected, mean_mae


def compute_metrics(y_true: list[float], y_pred: list[float]) -> dict[str, float]:
    if not y_true:
        return {name: math.nan for name in METRIC_NAMES}
    errors = [pred - truth for truth, pred in zip(y_true, y_pred)]
    abs_errors = [abs(error) for error in errors]
    return {
        "mae": statistics.mean(abs_errors),
        "rmse": math.sqrt(statistics.mean(error * error for error in errors)),
        "median_absolute_error": statistics.median(abs_errors),
        "within_3_days_pct": 100.0 * sum(error <= 3 for error in abs_errors) / len(abs_errors),
        "within_7_days_pct": 100.0 * sum(error <= 7 for error in abs_errors) / len(abs_errors),
        "mean_signed_error": statistics.mean(errors),
    }


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def participant_bootstrap_mae(
    predictions: list[dict[str, Any]],
    reference_track: str = "history_only",
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, dict[str, float]]:
    errors: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in predictions:
        errors[str(row["track"])][str(row["participant_id"])].append(abs(float(row["error_days"])))

    tracks = sorted(errors)
    participants = sorted(errors.get(reference_track, {}))
    if not participants:
        return {}
    rng = random.Random(seed)
    bootstrap_mae: dict[str, list[float]] = {track: [] for track in tracks}
    bootstrap_delta: dict[str, list[float]] = {track: [] for track in tracks}

    for _ in range(replicates):
        sampled = [rng.choice(participants) for _ in participants]
        replicate_values: dict[str, float] = {}
        for track in tracks:
            sampled_errors = [
                error
                for participant in sampled
                for error in errors[track].get(participant, [])
            ]
            replicate_values[track] = statistics.mean(sampled_errors) if sampled_errors else math.nan
            bootstrap_mae[track].append(replicate_values[track])
        reference_mae = replicate_values[reference_track]
        for track in tracks:
            bootstrap_delta[track].append(replicate_values[track] - reference_mae)

    intervals: dict[str, dict[str, float]] = {}
    for track in tracks:
        intervals[track] = {
            "mae_ci_low": _percentile(bootstrap_mae[track], 0.025),
            "mae_ci_high": _percentile(bootstrap_mae[track], 0.975),
            "delta_mae_ci_low": _percentile(bootstrap_delta[track], 0.025),
            "delta_mae_ci_high": _percentile(bootstrap_delta[track], 0.975),
        }
    return intervals


def evaluate_feature_table(
    feature_table: FeatureTable,
    ridge_alphas: Iterable[float] = DEFAULT_RIDGE_ALPHAS,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = feature_table.rows
    if not rows:
        raise ValueError("No eligible examples were built.")
    folds = group_kfold_indices(rows)
    assert_participant_disjoint(rows, folds)
    tracks = feature_tracks(feature_table)
    fold_scores: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []

    for fold_number, (train_indices, test_indices) in enumerate(folds, start=1):
        y_train = [float(rows[idx]["target_cycle_length"]) for idx in train_indices]
        y_test = [float(rows[idx]["target_cycle_length"]) for idx in test_indices]
        median_pred = median(y_train)
        track_predictions: dict[str, list[float]] = {
            "global_median": [median_pred for _ in test_indices],
            "previous_cycle": [float(rows[idx]["history_previous_cycle_length"]) for idx in test_indices],
        }

        for track_name, feature_names in tracks.items():
            available_features = [
                name
                for name in feature_names
                if any(not math.isnan(_as_float(rows[idx].get(name))) for idx in train_indices)
            ]
            selected_alpha: float | str = ""
            inner_mae: dict[float, float] = {}
            if not available_features:
                track_predictions[track_name] = [median_pred for _ in test_indices]
                model_name = "training_fold_median_fallback"
            else:
                selected_alpha, inner_mae = select_ridge_alpha(
                    rows, train_indices, available_features, ridge_alphas
                )
                model = _pipeline(float(selected_alpha))
                model.fit(_matrix(rows, train_indices, available_features), np.asarray(y_train))
                predicted = model.predict(_matrix(rows, test_indices, available_features))
                track_predictions[track_name] = [float(value) for value in predicted]
                model_name = "ridge_nested_group_cv"

            metrics = compute_metrics(y_test, track_predictions[track_name])
            fold_scores.append(
                {
                    "fold": fold_number,
                    "track": track_name,
                    "model": model_name,
                    "n_train": len(train_indices),
                    "n_test": len(test_indices),
                    "n_train_participants": len({rows[idx]["participant_id"] for idx in train_indices}),
                    "n_test_participants": len({rows[idx]["participant_id"] for idx in test_indices}),
                    "feature_count": len(available_features),
                    "selected_alpha": selected_alpha,
                    "inner_cv_mae_by_alpha": ";".join(
                        f"{alpha:g}:{score:.6f}" for alpha, score in sorted(inner_mae.items())
                    ),
                    "features_used": ";".join(available_features),
                    **metrics,
                }
            )

        for track_name in ["global_median", "previous_cycle"]:
            metrics = compute_metrics(y_test, track_predictions[track_name])
            fold_scores.append(
                {
                    "fold": fold_number,
                    "track": track_name,
                    "model": "training_fold_median" if track_name == "global_median" else "previous_cycle_length",
                    "n_train": len(train_indices),
                    "n_test": len(test_indices),
                    "n_train_participants": len({rows[idx]["participant_id"] for idx in train_indices}),
                    "n_test_participants": len({rows[idx]["participant_id"] for idx in test_indices}),
                    "feature_count": 0,
                    "selected_alpha": "",
                    "inner_cv_mae_by_alpha": "",
                    "features_used": "",
                    **metrics,
                }
            )

        for track_name, predicted_values in track_predictions.items():
            for idx, predicted in zip(test_indices, predicted_values):
                row = rows[idx]
                predictions.append(
                    {
                        "fold": fold_number,
                        "track": track_name,
                        "example_id": row["example_id"],
                        "participant_id": row["participant_id"],
                        "study_interval": row["study_interval"],
                        "source_start_day": row["source_start_day"],
                        "target_start_day": row["target_start_day"],
                        "target_end_day": row["target_end_day"],
                        "observed_cycle_length": row["target_cycle_length"],
                        "predicted_cycle_length": predicted,
                        "error_days": predicted - float(row["target_cycle_length"]),
                    }
                )

    intervals = participant_bootstrap_mae(
        predictions,
        replicates=bootstrap_replicates,
        seed=random_seed,
    )
    reference_predictions = [row for row in predictions if row["track"] == "history_only"]
    reference_mae = compute_metrics(
        [float(row["observed_cycle_length"]) for row in reference_predictions],
        [float(row["predicted_cycle_length"]) for row in reference_predictions],
    )["mae"]

    scores: list[dict[str, Any]] = []
    for track in sorted({str(row["track"]) for row in predictions}):
        track_rows = [row for row in predictions if row["track"] == track]
        y_true = [float(row["observed_cycle_length"]) for row in track_rows]
        y_pred = [float(row["predicted_cycle_length"]) for row in track_rows]
        metrics = compute_metrics(y_true, y_pred)
        scores.append(
            {
                "track": track,
                "n": len(track_rows),
                **metrics,
                "delta_mae_vs_history": metrics["mae"] - reference_mae,
                **intervals.get(track, {}),
            }
        )
    scores.sort(key=lambda row: float(row["mae"]))
    return scores, fold_scores, predictions


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_value(row.get(key)) for key in fieldnames})


def _format_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.6f}"
    return value
