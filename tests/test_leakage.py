from __future__ import annotations

import unittest
from pathlib import Path

from src.data import build_cycle_examples, find_data_dir, load_combined_csv
from src.evaluate import assert_participant_disjoint, group_kfold_indices
from src.features import build_feature_table


ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC = ROOT / "examples" / "synthetic_data.csv"


class LeakageTests(unittest.TestCase):
    def test_missing_data_directory_has_actionable_error(self) -> None:
        missing = ROOT / "does-not-exist"
        with self.assertRaisesRegex(FileNotFoundError, "Replace /path/to/mcphases"):
            find_data_dir(missing)

    def test_group_folds_are_participant_disjoint(self) -> None:
        loaded = load_combined_csv(SYNTHETIC)
        examples = build_cycle_examples(loaded.flow_rows)
        feature_table = build_feature_table(examples, loaded.measurements)
        folds = group_kfold_indices(feature_table.rows)
        assert_participant_disjoint(feature_table.rows, folds)
        self.assertGreaterEqual(len(folds), 3)

    def test_feature_days_are_before_target_cycle_start(self) -> None:
        loaded = load_combined_csv(SYNTHETIC)
        examples = build_cycle_examples(loaded.flow_rows)
        feature_table = build_feature_table(examples, loaded.measurements)
        self.assertGreater(len(feature_table.rows), 0)
        for row in feature_table.rows:
            self.assertLess(row["feature_max_day"], row["target_start_day"])
            observed = row["observed_feature_max_day"]
            if observed == observed:
                self.assertLess(observed, row["target_start_day"])


if __name__ == "__main__":
    unittest.main()
