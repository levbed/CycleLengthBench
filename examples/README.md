# Synthetic Demo Data

`synthetic_data.csv` is fully fabricated and contains no mcPHASES records. It is
generated deterministically by `generate_synthetic_data.py`.

The generator creates source-cycle stress-related hormone, wearable, symptom,
and glucose signals that influence the following synthetic cycle length. This
known signal lets the demo verify that multimodal tracks can recover incremental
information when it exists. Demo rankings are not scientific results and should
never be presented as evidence about real physiology or clinical performance.

Regenerate the fixture with:

```bash
python examples/generate_synthetic_data.py --output examples/synthetic_data.csv
```
