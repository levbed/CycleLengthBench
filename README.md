# mcPHASES CycleBench

mcPHASES CycleBench is a participant-disjoint benchmark for next-cycle length forecasting. It tests whether hormone and wearable summaries improve prediction beyond simple menstrual-history baselines.

This repository does not redistribute raw mcPHASES data. Keep the downloaded mcPHASES release outside version control, or in a path ignored by `.gitignore`.

## Research Question

For participants with complete consecutive menstrual cycles, do source-cycle hormone or wearable summaries improve prediction of the next menstrual-cycle length over simple cycle-history baselines?

This benchmark is exploratory and is not intended for diagnosis, fertility planning, treatment, or perimenopause prediction.

## Dataset Access

Download mcPHASES through the official dataset access channel and place the extracted release locally. The benchmark expects a directory containing `hormones_and_selfreport.csv`; it can also accept a parent directory with exactly one child release directory containing that file.

Do not commit the downloaded data, participant-level predictions, or raw-derived participant outputs. The included `.gitignore` ignores the local mcPHASES release directory pattern and generated result CSV/PNG files under `results/`.

## Target Definition

For source cycle `t`, predict the length in days of cycle `t+1`.

Cycle starts are inferred from menstrual-flow evidence in `hormones_and_selfreport.csv`: a day is treated as menstrual evidence when `flow_volume` is positive or `phase` is `Menstrual`. Adjacent menstrual-evidence days separated by at most two days are merged into one episode, and the episode start is used as the cycle start.

An eligible example requires three consecutive inferred starts within the same participant and study interval:

- source start: start of cycle `t`
- target start: start of cycle `t+1`
- target end: start of cycle `t+2`

Both the source-cycle length and target-cycle length must fall in the plausible default range of 10 to 90 days. Features are summarized only from source-cycle days `source_start_day <= day < target_start_day`. The target-cycle end date is used only to compute the label.

## Feature Tracks

The benchmark automatically skips unavailable variables and reports those used.

- `global_median`: training-fold median target length.
- `previous_cycle`: source-cycle length.
- `history_only`: previous cycle length, historical mean, median, standard deviation, and number of prior complete cycles available before the target starts.
- `history_plus_wearables`: history plus available source-cycle sleep, resting heart rate, steps, active minutes, and coverage summaries.
- `history_plus_hormones`: history plus available source-cycle hormone means, maxima, standard deviations, counts, and coverage for E3G/estrogen, LH, PdG, and FSH when present.
- `full_multimodal`: history, wearables, and hormones.

## Participant-Safe Evaluation

Evaluation uses participant-disjoint grouped cross-validation, grouped by participant ID. Five folds are used when there are at least five participants; otherwise three folds are used when feasible.

Median imputation, standard scaling, and ridge regression fitting are performed independently inside each training fold. No preprocessing statistics are fitted on held-out participants.

Leakage checks in `tests/test_leakage.py` confirm:

- training and testing participants do not overlap within a fold
- feature days occur before the target cycle starts

## Outputs

Running `evaluate` or `demo` saves:

- `results/scores.csv`
- `results/fold_scores.csv`
- `results/predictions.csv`
- `results/mae_by_track.png`
- `results/predicted_vs_observed.png`

Generated CSVs and PNGs in `results/` are ignored by default to avoid accidentally committing participant-level outputs.

## Reproduction Commands

Create an environment with Python 3.10 or newer. There are no mandatory third-party packages.
Replace `/path/to/mcphases` below with the actual directory where you downloaded and
extracted the dataset. The directory may either contain
`hormones_and_selfreport.csv` directly or contain one extracted dataset directory.

```bash
python run_benchmark.py inspect --data-dir /path/to/mcphases
python run_benchmark.py evaluate --data-dir /path/to/mcphases
python run_benchmark.py demo --data-file examples/synthetic_data.csv
python -m unittest tests/test_leakage.py
```

On systems where `python` is not available, use `python3`.

## Baseline Results

Executed locally with:

```bash
python3 run_benchmark.py evaluate --data-dir mcphases-a-dataset-of-physiological-hormonal-and-self-reported-events-and-symptoms-for-menstrual-health-tracking-with-wearables-1.0.0
```

Local mcPHASES v1.0.0 inspection found 42 participants, 142 inferred complete cycles, and 82 eligible examples.

| Track | MAE | RMSE | Within 7 days |
| --- | ---: | ---: | ---: |
| history_only | 5.090 | 6.834 | 72.0% |
| global_median | 5.171 | 7.375 | 76.8% |
| previous_cycle | 5.683 | 8.212 | 70.7% |
| history_plus_hormones | 5.787 | 7.800 | 69.5% |
| history_plus_wearables | 6.257 | 12.276 | 78.0% |
| full_multimodal | 10.511 | 39.396 | 68.3% |

Variables used in that run:

- Hormones/self-report: `hormone_e3g`, `hormone_lh`, `hormone_pdg`
- Sleep: `sleep_efficiency`, `sleep_minutes_asleep`, `sleep_minutes_awake`, `sleep_time_in_bed`
- Wearables: `resting_heart_rate`, `steps`, `active_minutes`

The included synthetic demo is fabricated and exists only to verify the pipeline end to end.

## Limitations

Cycle starts are inferred from available flow and phase fields, not manually adjudicated. Missing or inconsistent self-reporting can change eligibility and labels.

The benchmark uses simple source-cycle summaries and ridge regression only. It intentionally skips deep learning, dashboards, extensive tuning, and additional prediction tasks.

The participant count is modest after requiring complete consecutive cycles, so fold-level estimates can be unstable. The multimodal tracks can overfit when source-cycle wearable or hormone coverage is sparse.

The target is next-cycle length only. The benchmark should not be interpreted as a clinical model or as evidence for diagnosis, fertility planning, treatment decisions, or perimenopause prediction.
