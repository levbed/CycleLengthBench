from __future__ import annotations

import csv
import json
import math
import os
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

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
DEFAULT_MODEL_IDS = ("ridge", "rbf_svr", "hist_gradient_boosting")
MODEL_LABELS = {
    "baseline": "Baseline",
    "ridge": "Ridge",
    "rbf_svr": "RBF-SVR",
    "hist_gradient_boosting": "HistGradientBoosting",
}
DEFAULT_BOOTSTRAP_REPLICATES = 2000
DEFAULT_RANDOM_SEED = 2026


def default_model_grids(
    ridge_alphas: Iterable[float] = DEFAULT_RIDGE_ALPHAS,
) -> dict[str, list[dict[str, Any]]]:
    # Simpler or more strongly regularized candidates appear first for deterministic ties.
    return {
        "ridge": [
            {"alpha": float(alpha)}
            for alpha in sorted({float(value) for value in ridge_alphas}, reverse=True)
        ],
        "rbf_svr": [
            {"C": c, "epsilon": epsilon, "gamma": "scale"}
            for c in (1.0, 10.0, 100.0)
            for epsilon in (3.0, 1.0)
        ],
        "hist_gradient_boosting": [
            {
                "learning_rate": 0.05,
                "max_iter": 150,
                "max_leaf_nodes": leaves,
                "min_samples_leaf": minimum,
                "l2_regularization": regularization,
            }
            for leaves, minimum, regularization in (
                (3, 10, 10.0),
                (3, 5, 10.0),
                (7, 10, 10.0),
                (7, 5, 10.0),
                (3, 10, 1.0),
                (7, 5, 1.0),
            )
        ],
    }


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


def _pipeline(model_id: str, parameters: dict[str, Any], seed: int) -> Pipeline:
    if model_id == "ridge":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=float(parameters["alpha"]))),
            ]
        )
    if model_id == "rbf_svr":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                (
                    "regressor",
                    SVR(
                        kernel="rbf",
                        C=float(parameters["C"]),
                        epsilon=float(parameters["epsilon"]),
                        gamma=parameters["gamma"],
                    ),
                ),
            ]
        )
    if model_id == "hist_gradient_boosting":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                (
                    "regressor",
                    HistGradientBoostingRegressor(
                        learning_rate=float(parameters["learning_rate"]),
                        max_iter=int(parameters["max_iter"]),
                        max_leaf_nodes=int(parameters["max_leaf_nodes"]),
                        min_samples_leaf=int(parameters["min_samples_leaf"]),
                        l2_regularization=float(parameters["l2_regularization"]),
                        early_stopping=False,
                        random_state=seed,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown model: {model_id}")


def _parameter_key(parameters: dict[str, Any]) -> str:
    return json.dumps(parameters, sort_keys=True, separators=(",", ":"))


def select_model_parameters(
    rows: list[dict[str, Any]],
    train_indices: list[int],
    feature_names: list[str],
    model_id: str,
    candidates: list[dict[str, Any]],
    seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[dict[str, Any], dict[str, float]]:
    if not candidates:
        raise ValueError(f"At least one hyperparameter candidate is required for {model_id}.")
    groups = np.asarray([str(rows[idx]["participant_id"]) for idx in train_indices])
    unique_groups = sorted(set(groups))
    if len(unique_groups) < 2:
        return candidates[0], {_parameter_key(candidates[0]): math.nan}

    splitter = GroupKFold(n_splits=min(3, len(unique_groups)))
    x_all = _matrix(rows, train_indices, feature_names)
    y_all = np.asarray([float(rows[idx]["target_cycle_length"]) for idx in train_indices])
    mean_mae: dict[str, float] = {}

    for candidate in candidates:
        fold_mae: list[float] = []
        for inner_train, inner_validation in splitter.split(x_all, y_all, groups=groups):
            model = _pipeline(model_id, candidate, seed)
            model.fit(x_all[inner_train], y_all[inner_train])
            predictions = model.predict(x_all[inner_validation])
            fold_mae.append(float(np.mean(np.abs(predictions - y_all[inner_validation]))))
        mean_mae[_parameter_key(candidate)] = statistics.mean(fold_mae)

    selected = min(candidates, key=lambda item: mean_mae[_parameter_key(item)])
    return selected, mean_mae


def select_ridge_alpha(
    rows: list[dict[str, Any]],
    train_indices: list[int],
    feature_names: list[str],
    alphas: Iterable[float] = DEFAULT_RIDGE_ALPHAS,
) -> tuple[float, dict[float, float]]:
    candidates = default_model_grids(alphas)["ridge"]
    selected, scores = select_model_parameters(
        rows, train_indices, feature_names, "ridge", candidates
    )
    by_alpha = {float(json.loads(key)["alpha"]): value for key, value in scores.items()}
    return float(selected["alpha"]), by_alpha


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


def _result_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("model", "ridge")), str(row["track"])


def participant_bootstrap_mae(
    predictions: list[dict[str, Any]],
    primary_reference: tuple[str, str] = ("ridge", "history_only"),
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_RANDOM_SEED,
) -> dict[tuple[str, str], dict[str, float]]:
    errors: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in predictions:
        errors[_result_key(row)][str(row["participant_id"])].append(abs(float(row["error_days"])))

    if primary_reference not in errors:
        primary_reference = next((key for key in errors if key[1] == "history_only"), next(iter(errors), primary_reference))
    participants = sorted(errors.get(primary_reference, {}))
    if not participants:
        return {}

    keys = sorted(errors)
    references = {
        key: (key[0], "history_only") if (key[0], "history_only") in errors else primary_reference
        for key in keys
    }
    ridge_references = {
        key: ("ridge", key[1]) if ("ridge", key[1]) in errors else primary_reference
        for key in keys
    }
    rng = random.Random(seed)
    bootstrap_mae: dict[tuple[str, str], list[float]] = {key: [] for key in keys}
    bootstrap_delta: dict[tuple[str, str], list[float]] = {key: [] for key in keys}
    bootstrap_model_delta: dict[tuple[str, str], list[float]] = {key: [] for key in keys}

    for _ in range(replicates):
        sampled = [rng.choice(participants) for _ in participants]
        replicate_values: dict[tuple[str, str], float] = {}
        for key in keys:
            sampled_errors = [
                error
                for participant in sampled
                for error in errors[key].get(participant, [])
            ]
            replicate_values[key] = statistics.mean(sampled_errors) if sampled_errors else math.nan
            bootstrap_mae[key].append(replicate_values[key])
        for key in keys:
            bootstrap_delta[key].append(replicate_values[key] - replicate_values[references[key]])
            bootstrap_model_delta[key].append(
                replicate_values[key] - replicate_values[ridge_references[key]]
            )

    return {
        key: {
            "mae_ci_low": _percentile(bootstrap_mae[key], 0.025),
            "mae_ci_high": _percentile(bootstrap_mae[key], 0.975),
            "delta_mae_ci_low": _percentile(bootstrap_delta[key], 0.025),
            "delta_mae_ci_high": _percentile(bootstrap_delta[key], 0.975),
            "delta_mae_vs_ridge_ci_low": _percentile(bootstrap_model_delta[key], 0.025),
            "delta_mae_vs_ridge_ci_high": _percentile(bootstrap_model_delta[key], 0.975),
        }
        for key in keys
    }


def evaluate_feature_table(
    feature_table: FeatureTable,
    ridge_alphas: Iterable[float] = DEFAULT_RIDGE_ALPHAS,
    model_ids: Iterable[str] = DEFAULT_MODEL_IDS,
    model_grids: dict[str, list[dict[str, Any]]] | None = None,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = feature_table.rows
    if not rows:
        raise ValueError("No eligible examples were built.")
    folds = group_kfold_indices(rows)
    assert_participant_disjoint(rows, folds)
    tracks = feature_tracks(feature_table)
    grids = model_grids or default_model_grids(ridge_alphas)
    requested_models = [str(model_id) for model_id in model_ids]
    unknown = [model_id for model_id in requested_models if model_id not in grids]
    if unknown:
        raise ValueError(f"No tuning grid configured for: {', '.join(unknown)}")

    fold_scores: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []

    for fold_number, (train_indices, test_indices) in enumerate(folds, start=1):
        y_train = [float(rows[idx]["target_cycle_length"]) for idx in train_indices]
        y_test = [float(rows[idx]["target_cycle_length"]) for idx in test_indices]
        median_pred = median(y_train)
        fold_predictions: dict[tuple[str, str], list[float]] = {
            ("baseline", "global_median"): [median_pred for _ in test_indices],
            ("baseline", "previous_cycle"): [
                float(rows[idx]["history_previous_cycle_length"]) for idx in test_indices
            ],
        }

        for model_id in requested_models:
            for track_name, feature_names in tracks.items():
                available_features = [
                    name
                    for name in feature_names
                    if any(not math.isnan(_as_float(rows[idx].get(name))) for idx in train_indices)
                ]
                selected: dict[str, Any] = {}
                inner_mae: dict[str, float] = {}
                estimator = "training_fold_median_fallback"
                if available_features:
                    selected, inner_mae = select_model_parameters(
                        rows,
                        train_indices,
                        available_features,
                        model_id,
                        grids[model_id],
                        seed=random_seed,
                    )
                    model = _pipeline(model_id, selected, random_seed)
                    model.fit(_matrix(rows, train_indices, available_features), np.asarray(y_train))
                    predicted = model.predict(_matrix(rows, test_indices, available_features))
                    fold_predictions[(model_id, track_name)] = [float(value) for value in predicted]
                    estimator = f"{model_id}_nested_group_cv"
                else:
                    fold_predictions[(model_id, track_name)] = [median_pred for _ in test_indices]

                metrics = compute_metrics(y_test, fold_predictions[(model_id, track_name)])
                fold_scores.append(
                    {
                        "fold": fold_number,
                        "model": model_id,
                        "model_label": MODEL_LABELS[model_id],
                        "track": track_name,
                        "estimator": estimator,
                        "n_train": len(train_indices),
                        "n_test": len(test_indices),
                        "n_train_participants": len({rows[idx]["participant_id"] for idx in train_indices}),
                        "n_test_participants": len({rows[idx]["participant_id"] for idx in test_indices}),
                        "feature_count": len(available_features),
                        "selected_hyperparameters": _parameter_key(selected) if selected else "",
                        "selected_alpha": selected.get("alpha", ""),
                        "inner_cv_mae_by_candidate": json.dumps(inner_mae, sort_keys=True),
                        "features_used": ";".join(available_features),
                        **metrics,
                    }
                )

        for track_name in ("global_median", "previous_cycle"):
            metrics = compute_metrics(y_test, fold_predictions[("baseline", track_name)])
            fold_scores.append(
                {
                    "fold": fold_number,
                    "model": "baseline",
                    "model_label": MODEL_LABELS["baseline"],
                    "track": track_name,
                    "estimator": "training_fold_median" if track_name == "global_median" else "previous_cycle_length",
                    "n_train": len(train_indices),
                    "n_test": len(test_indices),
                    "n_train_participants": len({rows[idx]["participant_id"] for idx in train_indices}),
                    "n_test_participants": len({rows[idx]["participant_id"] for idx in test_indices}),
                    "feature_count": 0,
                    "selected_hyperparameters": "",
                    "selected_alpha": "",
                    "inner_cv_mae_by_candidate": "",
                    "features_used": "",
                    **metrics,
                }
            )

        for (model_id, track_name), predicted_values in fold_predictions.items():
            for idx, predicted in zip(test_indices, predicted_values):
                row = rows[idx]
                predictions.append(
                    {
                        "fold": fold_number,
                        "model": model_id,
                        "model_label": MODEL_LABELS[model_id],
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

    primary_reference = ("ridge", "history_only")
    intervals = participant_bootstrap_mae(
        predictions,
        primary_reference=primary_reference,
        replicates=bootstrap_replicates,
        seed=random_seed,
    )
    grouped_predictions: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped_predictions[_result_key(row)].append(row)

    scores: list[dict[str, Any]] = []
    for key, result_rows in grouped_predictions.items():
        model_id, track = key
        y_true = [float(row["observed_cycle_length"]) for row in result_rows]
        y_pred = [float(row["predicted_cycle_length"]) for row in result_rows]
        metrics = compute_metrics(y_true, y_pred)
        reference_key = (
            (model_id, "history_only")
            if (model_id, "history_only") in grouped_predictions
            else primary_reference
        )
        reference_rows = grouped_predictions[reference_key]
        reference_mae = compute_metrics(
            [float(row["observed_cycle_length"]) for row in reference_rows],
            [float(row["predicted_cycle_length"]) for row in reference_rows],
        )["mae"]
        ridge_reference_key = (
            ("ridge", track) if ("ridge", track) in grouped_predictions else primary_reference
        )
        ridge_reference_rows = grouped_predictions[ridge_reference_key]
        ridge_reference_mae = compute_metrics(
            [float(row["observed_cycle_length"]) for row in ridge_reference_rows],
            [float(row["predicted_cycle_length"]) for row in ridge_reference_rows],
        )["mae"]
        scores.append(
            {
                "model": model_id,
                "model_label": MODEL_LABELS[model_id],
                "track": track,
                "reference_model": reference_key[0],
                "reference_track": reference_key[1],
                "n": len(result_rows),
                **metrics,
                "delta_mae_vs_history": metrics["mae"] - reference_mae,
                "delta_mae_vs_ridge_same_track": metrics["mae"] - ridge_reference_mae,
                **intervals.get(key, {}),
            }
        )

    model_order = {model_id: index for index, model_id in enumerate((*requested_models, "baseline"))}
    scores.sort(key=lambda row: (model_order.get(str(row["model"]), 99), float(row["mae"])))
    return scores, fold_scores, predictions


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        destination.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_value(row.get(key)) for key in fieldnames})


def _format_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.6f}"
    return value
