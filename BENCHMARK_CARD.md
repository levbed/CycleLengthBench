# mcPHASES CycleBench Benchmark Card

## Research Question

For participants with complete consecutive menstrual cycles, do source-cycle
hormone, wearable, symptom, glucose, or stress summaries improve prediction of
the next cycle's length beyond menstrual-history baselines?

## Task

For source cycle `t`, predict the length in days of cycle `t+1`. Cycle `t+1`
begins at the next inferred menstrual start and ends at the following inferred
menstrual start.

This is an exploratory research task. It is not intended for diagnosis,
fertility planning, treatment, individual guidance, or perimenopause prediction.

## Tracks

1. `global_median`: target median fitted on the outer training fold.
2. `previous_cycle`: the complete source-cycle length.
3. `history_only`: previous length, historical mean, median, standard deviation,
   and prior-cycle count.
4. `history_plus_wearables`: history plus sleep, resting heart rate, activity,
   steps, and coverage summaries.
5. `history_plus_hormones`: history plus E3G, LH, PdG, FSH, and coverage
   summaries when available.
6. `history_plus_symptoms`: history plus ordinal self-report summaries.
7. `history_plus_glucose_stress`: history plus CGM and Fitbit stress summaries.
8. `full_multimodal`: all available feature families.

Unavailable variables and unavailable optional tracks are skipped and reported.

## Models And Preprocessing

- Training-fold median baseline.
- Previous-cycle baseline.
- Scikit-learn Ridge regression.
- Median imputation and standard scaling in a scikit-learn `Pipeline`.
- Ridge `alpha` selected from `0.1, 1, 10, 100` using inner
  participant-disjoint GroupKFold.

Feature availability, imputation, scaling, alpha selection, and model fitting
occur without access to the outer test fold.

## Evaluation

- Outer participant-disjoint `GroupKFold`, grouped by participant ID.
- Five folds when at least five participants are available; otherwise three
  folds when feasible.
- MAE, RMSE, median absolute error, percentage within 3 days, percentage within
  7 days, and mean signed error.
- Paired MAE difference versus `history_only`.
- 95% participant-clustered bootstrap intervals using 2,000 deterministic
  replicates.

## Leakage Controls

- Training and test participant sets must be disjoint in every outer fold.
- Feature timestamps must be strictly before the target cycle begins.
- No target-cycle measurement, target end date, or future-cycle summary is a
  model feature.
- All learned preprocessing and hyperparameter selection is contained within
  training data.

Automated tests enforce participant separation, temporal boundaries, aggregate
OpenAI payload safety, and the use of `store=False` for OpenAI requests.

## v1.1 Aggregate Results

Local evaluation of mcPHASES v1.0.0 produced 82 examples from 42 participants
and 142 inferred complete cycles.

| Track | MAE (95% CI) | Delta MAE vs history (95% CI) | Within 7 days |
| --- | ---: | ---: | ---: |
| `history_only` | 4.84 (3.57, 6.23) | 0.000 (0.000, 0.000) | 76.8% |
| `history_plus_hormones` | 5.17 (3.91, 6.59) | +0.334 (+0.002, +0.649) | 73.2% |
| `history_plus_glucose_stress` | 5.23 (3.95, 6.66) | +0.390 (+0.039, +0.826) | 72.0% |
| `history_plus_symptoms` | 5.28 (3.97, 6.72) | +0.439 (+0.104, +0.816) | 74.4% |
| `global_median` | 5.29 (3.60, 7.24) | +0.452 (-0.308, +1.235) | 78.0% |
| `history_plus_wearables` | 5.39 (3.86, 7.14) | +0.551 (-0.345, +1.724) | 74.4% |
| `previous_cycle` | 5.68 (4.47, 7.21) | +0.842 (-0.577, +2.107) | 70.7% |
| `full_multimodal` | 5.83 (4.38, 7.51) | +0.994 (+0.117, +2.164) | 68.3% |

No added modality improved over history-only in this evaluation. Hormone,
glucose/stress, symptom, and full-multimodal tracks had participant-bootstrap
intervals for MAE difference above zero. The wearable comparison was
inconclusive. These are benchmark comparisons, not causal or clinical findings.

## Outputs

The pipeline writes aggregate scores, fold diagnostics, participant-level
predictions, aggregate reports, and plots. Generated outputs remain ignored by
default. Only aggregate figures intentionally copied into documentation should
be committed.
