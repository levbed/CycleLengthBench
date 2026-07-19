from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.report import assert_aggregate_payload_safe


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SUMMARY = ROOT / "docs" / "data" / "benchmark_summary.json"


class PublicArtifactTests(unittest.TestCase):
    def test_public_summary_is_aggregate_and_complete(self) -> None:
        payload = json.loads(PUBLIC_SUMMARY.read_text())
        assert_aggregate_payload_safe(payload)
        self.assertEqual(payload["benchmark"], "mcPHASES CycleBench")
        self.assertEqual(payload["protocol_version"], "2.0")
        self.assertEqual(len(payload["scores"]), 20)
        serialized = json.dumps(payload).lower()
        for forbidden in ("participant_id", "example_id", "source_start_day", "predictions"):
            self.assertNotIn(forbidden, serialized)


if __name__ == "__main__":
    unittest.main()
