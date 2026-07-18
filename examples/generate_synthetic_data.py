#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


FIELDNAMES = [
    "id",
    "study_interval",
    "day_in_study",
    "phase",
    "flow_volume",
    "lh",
    "estrogen",
    "pdg",
    "fsh",
    "sleep_minutes",
    "resting_hr",
    "steps",
    "active_minutes",
    "glucose",
    "stress_score",
    "fatigue",
    "cramps",
    "stress",
    "sleepissue",
    "moodswing",
]

ORDINAL = ["Not at all", "Very Low/Little", "Low", "Moderate", "High", "Very High"]


def _level(value: float) -> str:
    return ORDINAL[max(0, min(5, round(value)))]


def generate_rows(participants: int = 18, complete_cycles: int = 6, seed: int = 2026):
    rows: list[dict[str, object]] = []
    for participant_index in range(participants):
        rng = random.Random(seed + participant_index)
        participant_id = f"S{participant_index + 1:03d}"
        baseline = 25 + participant_index % 8
        resting_offset = (participant_index % 5) - 2
        stress = [rng.uniform(-1.5, 1.5) for _ in range(complete_cycles)]
        lengths = [max(20, min(40, round(baseline + rng.gauss(0, 0.8))))]
        for cycle_index in range(1, complete_cycles):
            next_length = baseline + 2.4 * stress[cycle_index - 1] + rng.gauss(0, 0.8)
            lengths.append(max(20, min(40, round(next_length))))

        starts = [1]
        for length in lengths:
            starts.append(starts[-1] + length)

        for cycle_index, start in enumerate(starts):
            if cycle_index == complete_cycles:
                rows.append(
                    {
                        "id": participant_id,
                        "study_interval": "synthetic",
                        "day_in_study": start,
                        "phase": "Menstrual",
                        "flow_volume": "Moderate",
                    }
                )
                continue

            latent = stress[cycle_index]
            symptom_level = 2.5 + latent
            rows.append(
                {
                    "id": participant_id,
                    "study_interval": "synthetic",
                    "day_in_study": start,
                    "phase": "Menstrual",
                    "flow_volume": "Moderate",
                    "lh": f"{4.5 + 0.4 * latent:.2f}",
                    "estrogen": f"{95 + 7 * latent:.1f}",
                    "pdg": f"{2.2 - 0.1 * latent:.2f}",
                    "fsh": f"{5.0 + 0.2 * latent:.2f}",
                    "sleep_minutes": round(425 - 20 * latent + rng.gauss(0, 6)),
                    "resting_hr": f"{62 + resting_offset + 2.2 * latent:.1f}",
                    "steps": round(7900 - 650 * latent + rng.gauss(0, 180)),
                    "active_minutes": round(55 - 5 * latent + rng.gauss(0, 2)),
                    "glucose": f"{96 + 5 * latent + rng.gauss(0, 1):.1f}",
                    "stress_score": f"{50 + 11 * latent + rng.gauss(0, 2):.1f}",
                    "fatigue": _level(symptom_level),
                    "cramps": _level(3.0 + 0.5 * latent),
                    "stress": _level(symptom_level),
                    "sleepissue": _level(2.0 + latent),
                    "moodswing": _level(2.0 + 0.7 * latent),
                }
            )
            rows.append(
                {
                    "id": participant_id,
                    "study_interval": "synthetic",
                    "day_in_study": start + 7,
                    "phase": "Follicular",
                    "flow_volume": "Not at all",
                    "lh": f"{7.5 + 0.6 * latent:.2f}",
                    "estrogen": f"{145 + 12 * latent:.1f}",
                    "pdg": f"{2.8 - 0.1 * latent:.2f}",
                    "fsh": f"{4.4 + 0.2 * latent:.2f}",
                    "sleep_minutes": round(435 - 18 * latent + rng.gauss(0, 6)),
                    "resting_hr": f"{61 + resting_offset + 2 * latent:.1f}",
                    "steps": round(8500 - 600 * latent + rng.gauss(0, 180)),
                    "active_minutes": round(62 - 5 * latent + rng.gauss(0, 2)),
                    "glucose": f"{101 + 8 * latent + rng.gauss(0, 1):.1f}",
                    "stress_score": f"{48 + 10 * latent + rng.gauss(0, 2):.1f}",
                    "fatigue": _level(2.0 + latent),
                    "cramps": _level(0.5 + 0.3 * latent),
                    "stress": _level(2.0 + latent),
                    "sleepissue": _level(1.5 + latent),
                    "moodswing": _level(1.5 + 0.7 * latent),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic fabricated CycleBench demo data.")
    parser.add_argument("--output", default="examples/synthetic_data.csv")
    args = parser.parse_args()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(generate_rows())
    print(f"Wrote fabricated demo data to {destination.resolve()}")


if __name__ == "__main__":
    main()
