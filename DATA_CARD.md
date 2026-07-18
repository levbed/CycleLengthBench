# mcPHASES CycleBench Data Card

## Source

CycleBench uses mcPHASES v1.0.0:

> Lin, B., Li, J. Y., Kalani, K., Truong, K., & Mariakakis, A. (2025).
> mcPHASES: A Dataset of Physiological, Hormonal, and Self-reported Events and
> Symptoms for Menstrual Health Tracking with Wearables. PhysioNet.
> https://doi.org/10.13026/zx6a-2c81

The source dataset contains longitudinal data from 42 Canadian young adult
participants across two study intervals. Modalities include Fitbit data,
continuous glucose monitoring, urine hormone measurements, menstrual reports,
and daily self-reported experiences.

Official dataset page: https://physionet.org/content/mcphases/1.0.0/

## Access And License

mcPHASES is a restricted-access PhysioNet resource governed by the PhysioNet
Restricted Health Data License and Data Use Agreement 1.5.0. Each researcher
must obtain access directly from PhysioNet and comply with its terms.

This repository does not redistribute mcPHASES files, participant records,
participant identifiers, or participant-level derived outputs. The repository's
MIT license applies to CycleBench code and documentation, not to mcPHASES data.

## Tables Used

CycleBench automatically uses available variables from:

| Source table | Variables |
| --- | --- |
| `hormones_and_selfreport.csv` | Menstrual evidence, E3G/estrogen, LH, PdG, FSH when present, and ordinal self-reports |
| `sleep.csv` | Minutes asleep, minutes awake, time in bed, sleep efficiency |
| `resting_heart_rate.csv` | Resting heart rate |
| `active_minutes.csv` | Light, moderate, and vigorous active minutes |
| `steps.csv` | Daily total steps |
| `glucose.csv` | CGM glucose values |
| `stress_score.csv` | Fitbit stress score |

Self-reports include appetite, exercise level, headaches, cramps, sore breasts,
fatigue, sleep issues, mood swings, stress, food cravings, indigestion, and
bloating. Ordered responses are mapped from 0 (`Not at all`) through 5
(`Very High`). Unknown values are treated as missing.

## Unit Of Analysis

One example represents a complete source cycle `t` and its immediately
following complete target cycle `t+1`, within one participant and study
interval. The label is the target cycle's length in days.

Features use only source-cycle measurements with
`source_start_day <= day < target_start_day`. The target-cycle end is used only
to calculate the label and is never included in model features.

## Eligibility

- At least three consecutive inferred cycle starts are required.
- Source and target cycle lengths must each be between 10 and 90 days inclusive.
- Menstrual episodes are inferred from positive flow or `Menstrual` phase
  reports; evidence days separated by at most two days are merged.
- Missing modalities do not exclude an otherwise eligible cycle pair.

## Known Limitations

- Cycle boundaries are inferred from self-report rather than adjudicated.
- The cohort is small and does not represent all ages, geographies, health
  conditions, devices, or menstrual experiences.
- Hormone and sensor coverage differs across participants and cycles.
- Ordinal self-report mappings assume consistent interpretation of response
  categories.
- CycleBench does not establish clinical validity or causal relationships.

## Privacy

Raw data and generated participant-level files are ignored by Git. OpenAI
summarization accepts only `benchmark_summary.json`, which contains aggregate
cohort counts, coverage, model metrics, and uncertainty intervals. It rejects
participant IDs, example IDs, dates, and predictions.
