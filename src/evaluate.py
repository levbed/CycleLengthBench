from __future__ import annotations

import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .features import FeatureTable, feature_tracks


METRIC_NAMES = [
    "mae",
    "rmse",
    "median_absolute_error",
    "within_3_days_pct",
    "within_7_days_pct",
    "mean_signed_error",
]


def median(values: list[float]) -> float:
    clean = [float(value) for value in values if not math.isnan(float(value))]
    return statistics.median(clean) if clean else 0.0


def group_kfold_indices(rows: list[dict[str, Any]], group_key: str = "participant_id") -> list[tuple[list[int], list[int]]]:
    counts = Counter(str(row[group_key]) for row in rows)
    groups = sorted(counts, key=lambda group: (-counts[group], group))
    if len(groups) < 2:
        raise ValueError("Participant-disjoint evaluation requires at least two participants.")
    if len(groups) >= 5 and len(rows) >= 5:
        n_splits = 5
    elif len(groups) >= 3 and len(rows) >= 3:
        n_splits = 3
    else:
        n_splits = len(groups)

    fold_groups: list[list[str]] = [[] for _ in range(n_splits)]
    fold_sizes = [0 for _ in range(n_splits)]
    for group in groups:
        fold_index = min(range(n_splits), key=lambda idx: (fold_sizes[idx], idx))
        fold_groups[fold_index].append(group)
        fold_sizes[fold_index] += counts[group]

    folds: list[tuple[list[int], list[int]]] = []
    all_indices = set(range(len(rows)))
    for groups_in_fold in fold_groups:
        test_groups = set(groups_in_fold)
        test_indices = [idx for idx, row in enumerate(rows) if str(row[group_key]) in test_groups]
        train_indices = sorted(all_indices - set(test_indices))
        if train_indices and test_indices:
            folds.append((train_indices, test_indices))
    return folds


def assert_participant_disjoint(rows: list[dict[str, Any]], folds: list[tuple[list[int], list[int]]]) -> None:
    for fold_idx, (train_indices, test_indices) in enumerate(folds, start=1):
        train_groups = {rows[idx]["participant_id"] for idx in train_indices}
        test_groups = {rows[idx]["participant_id"] for idx in test_indices}
        overlap = train_groups & test_groups
        if overlap:
            raise AssertionError(f"Fold {fold_idx} has overlapping participants: {sorted(overlap)}")


def _column_medians(rows: list[dict[str, Any]], indices: list[int], feature_names: list[str]) -> dict[str, float]:
    return {name: median([_as_float(rows[idx].get(name)) for idx in indices]) for name in feature_names}


def _as_float(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if not math.isnan(number) and not math.isinf(number) else math.nan


def _prepare_matrix(
    rows: list[dict[str, Any]],
    indices: list[int],
    feature_names: list[str],
    medians: dict[str, float],
    means: dict[str, float] | None = None,
    scales: dict[str, float] | None = None,
) -> tuple[list[list[float]], dict[str, float], dict[str, float]]:
    raw: list[list[float]] = []
    for idx in indices:
        raw_row: list[float] = []
        for name in feature_names:
            value = _as_float(rows[idx].get(name))
            raw_row.append(medians[name] if math.isnan(value) else value)
        raw.append(raw_row)

    if means is None or scales is None:
        means = {}
        scales = {}
        for col_idx, name in enumerate(feature_names):
            column = [row[col_idx] for row in raw]
            avg = sum(column) / len(column) if column else 0.0
            variance = sum((value - avg) ** 2 for value in column) / len(column) if column else 0.0
            scale = math.sqrt(variance)
            means[name] = avg
            scales[name] = scale if scale > 1e-12 else 1.0

    transformed = [
        [(value - means[name]) / scales[name] for value, name in zip(row, feature_names)]
        for row in raw
    ]
    return transformed, means, scales


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    augmented = [matrix[i][:] + [vector[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-12:
            augmented[pivot][col] = 1e-12
        if pivot != col:
            augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        divisor = augmented[col][col]
        augmented[col] = [value / divisor for value in augmented[col]]
        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            if factor:
                augmented[row] = [
                    current - factor * pivot_value
                    for current, pivot_value in zip(augmented[row], augmented[col])
                ]
    return [augmented[i][-1] for i in range(n)]


def _fit_ridge(x_train: list[list[float]], y_train: list[float], alpha: float = 1.0) -> list[float]:
    p = len(x_train[0]) if x_train else 0
    size = p + 1
    xtx = [[0.0 for _ in range(size)] for _ in range(size)]
    xty = [0.0 for _ in range(size)]
    for features, target in zip(x_train, y_train):
        values = [1.0] + features
        for i in range(size):
            xty[i] += values[i] * target
            for j in range(size):
                xtx[i][j] += values[i] * values[j]
    for i in range(1, size):
        xtx[i][i] += alpha
    return _solve_linear_system(xtx, xty)


def _predict_ridge(coefs: list[float], x_rows: list[list[float]]) -> list[float]:
    return [coefs[0] + sum(weight * value for weight, value in zip(coefs[1:], row)) for row in x_rows]


def compute_metrics(y_true: list[float], y_pred: list[float]) -> dict[str, float]:
    if not y_true:
        return {name: math.nan for name in METRIC_NAMES}
    errors = [pred - truth for truth, pred in zip(y_true, y_pred)]
    abs_errors = [abs(error) for error in errors]
    return {
        "mae": sum(abs_errors) / len(abs_errors),
        "rmse": math.sqrt(sum(error * error for error in errors) / len(errors)),
        "median_absolute_error": statistics.median(abs_errors),
        "within_3_days_pct": 100.0 * sum(error <= 3 for error in abs_errors) / len(abs_errors),
        "within_7_days_pct": 100.0 * sum(error <= 7 for error in abs_errors) / len(abs_errors),
        "mean_signed_error": sum(errors) / len(errors),
    }


def evaluate_feature_table(feature_table: FeatureTable) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
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
        track_predictions = {
            "global_median": [median_pred for _ in test_indices],
            "previous_cycle": [float(rows[idx]["history_previous_cycle_length"]) for idx in test_indices],
        }

        for track_name, feature_names in tracks.items():
            available_features = [
                name
                for name in feature_names
                if any(not math.isnan(_as_float(rows[idx].get(name))) for idx in train_indices)
            ]
            if not available_features:
                track_predictions[track_name] = [median_pred for _ in test_indices]
                model_name = "training_fold_median_fallback"
            else:
                medians = _column_medians(rows, train_indices, available_features)
                x_train, means, scales = _prepare_matrix(rows, train_indices, available_features, medians)
                x_test, _, _ = _prepare_matrix(rows, test_indices, available_features, medians, means, scales)
                coefs = _fit_ridge(x_train, y_train, alpha=1.0)
                track_predictions[track_name] = _predict_ridge(coefs, x_test)
                model_name = "ridge"

            metrics = compute_metrics(y_test, track_predictions[track_name])
            fold_scores.append(
                {
                    "fold": fold_number,
                    "track": track_name,
                    "model": model_name,
                    "n_train": len(train_indices),
                    "n_test": len(test_indices),
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
                    "features_used": "",
                    **metrics,
                }
            )

        for track_name, preds in track_predictions.items():
            for idx, pred in zip(test_indices, preds):
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
                        "predicted_cycle_length": pred,
                        "error_days": pred - float(row["target_cycle_length"]),
                    }
                )

    scores: list[dict[str, Any]] = []
    for track in sorted({row["track"] for row in predictions}):
        track_rows = [row for row in predictions if row["track"] == track]
        y_true = [float(row["observed_cycle_length"]) for row in track_rows]
        y_pred = [float(row["predicted_cycle_length"]) for row in track_rows]
        scores.append({"track": track, "n": len(track_rows), **compute_metrics(y_true, y_pred)})
    scores.sort(key=lambda row: row["mae"])
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
