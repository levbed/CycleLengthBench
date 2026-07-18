from __future__ import annotations

import unittest
from pathlib import Path

from src.data import CycleExample, find_data_dir
from src.evaluate import assert_participant_disjoint, group_kfold_indices
from src.features import build_feature_table


ROOT = Path(__file__).resolve().parents[1]


class LeakageTests(unittest.TestCase):
    def test_missing_data_directory_has_actionable_error(self) -> None:
        missing = ROOT / "does-not-exist"
        with self.assertRaisesRegex(FileNotFoundError, "Replace /path/to/mcphases"):
            find_data_dir(missing)

    def test_group_folds_are_participant_disjoint(self) -> None:
        rows = [{"participant_id": f"P{index}"} for index in range(6)]
        folds = group_kfold_indices(rows)
        assert_participant_disjoint(rows, folds)
        self.assertGreaterEqual(len(folds), 3)

    def test_feature_days_are_before_target_cycle_start(self) -> None:
        example = CycleExample(
            example_id="test-example",
            participant_id="test-participant",
            study_interval="test-interval",
            source_cycle_index=1,
            source_start_day=10,
            target_start_day=40,
            target_end_day=70,
            previous_cycle_length=30.0,
            target_cycle_length=30.0,
            history_lengths=(29.0, 30.0),
        )
        measurements = {
            ("test-participant", "test-interval", 39): {"steps": [8000.0]},
            ("test-participant", "test-interval", 40): {"steps": [999999.0]},
        }
        row = build_feature_table([example], measurements).rows[0]
        self.assertLess(row["feature_max_day"], row["target_start_day"])
        self.assertLess(row["observed_feature_max_day"], row["target_start_day"])
        self.assertEqual(row["steps_max"], 8000.0)


if __name__ == "__main__":
    unittest.main()
