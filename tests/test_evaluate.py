from __future__ import annotations

import unittest
from pathlib import Path

from src.data import build_cycle_examples, load_combined_csv, safe_ordinal
from src.evaluate import DEFAULT_RIDGE_ALPHAS, evaluate_feature_table, participant_bootstrap_mae
from src.features import build_feature_table, feature_tracks


ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC = ROOT / "examples" / "synthetic_data.csv"


class EvaluationTests(unittest.TestCase):
    def test_self_report_ordinal_mapping(self) -> None:
        self.assertEqual(safe_ordinal("Not at all"), 0.0)
        self.assertEqual(safe_ordinal("Very Low/Little"), 1.0)
        self.assertEqual(safe_ordinal("Moderate"), 3.0)
        self.assertEqual(safe_ordinal("5"), 5.0)
        self.assertIsNone(safe_ordinal("unknown"))

    def test_nested_ridge_records_only_prespecified_alphas(self) -> None:
        loaded = load_combined_csv(SYNTHETIC)
        examples = build_cycle_examples(loaded.flow_rows)
        feature_table = build_feature_table(examples, loaded.measurements)
        _, fold_scores, _ = evaluate_feature_table(feature_table, bootstrap_replicates=100)
        selected = {
            float(row["selected_alpha"])
            for row in fold_scores
            if row["model"] == "ridge_nested_group_cv"
        }
        self.assertTrue(selected)
        self.assertTrue(selected.issubset(set(DEFAULT_RIDGE_ALPHAS)))

    def test_demo_exercises_optional_multimodal_tracks(self) -> None:
        loaded = load_combined_csv(SYNTHETIC)
        examples = build_cycle_examples(loaded.flow_rows)
        tracks = feature_tracks(build_feature_table(examples, loaded.measurements))
        self.assertIn("history_plus_symptoms", tracks)
        self.assertIn("history_plus_glucose_stress", tracks)

    def test_participant_bootstrap_is_paired_by_participant(self) -> None:
        predictions = []
        for participant in ["A", "B", "C"]:
            for track, errors in {"history_only": [2.0, 2.0], "better": [1.0, 1.0]}.items():
                for error in errors:
                    predictions.append(
                        {
                            "track": track,
                            "participant_id": participant,
                            "error_days": error,
                        }
                    )
        intervals = participant_bootstrap_mae(predictions, replicates=200, seed=7)
        self.assertAlmostEqual(intervals["better"]["delta_mae_ci_low"], -1.0)
        self.assertAlmostEqual(intervals["better"]["delta_mae_ci_high"], -1.0)


if __name__ == "__main__":
    unittest.main()
