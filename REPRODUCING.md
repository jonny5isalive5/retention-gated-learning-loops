# Reproducing the existing conditions

Run from the repository root. These commands perform training and are supplied for future reproduction; publication itself ran no experiments. Use Python 3.12 (observed: 3.12.10). Installation needs network access. `requirements.txt` is the untouched, unpinned original; `requirements-observed.txt` records installed versions at publication, not a historical pre-run lock. Dependency availability on other platforms is not guaranteed. The runners set PyTorch to two CPU threads.

## Setup (PowerShell)

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements-observed.txt
```

## Strict conditions (seeds 0 through 4)

```powershell
.venv/Scripts/python -X utf8 run_strict.py --seeds 5 --facts 100 --hidden 128 --budget 200 --out repro_strict_100
.venv/Scripts/python -X utf8 run_strict.py --seeds 5 --facts 1000 --hidden 128 --budget 200 --out repro_strict_1000_h128
.venv/Scripts/python -X utf8 run_strict.py --seeds 5 --facts 1000 --hidden 256 --budget 200 --out repro_strict_1000_h256
.venv/Scripts/python -X utf8 run_strict.py --seeds 5 --facts 1000 --hidden 512 --budget 200 --out repro_strict_1000_h512
.venv/Scripts/python -X utf8 run_strict.py --seeds 5 --facts 1000 --hidden 128 --budget 1000 --out repro_strict_1000_b1000
```

`-X utf8` avoids the saved Windows console encoding failure. Each output contains the configuration and code hashes, per-fact CSVs, summary JSON/Markdown and plot. Do not reuse the historical `results*` destinations. Default width and budget are 128 and 200; the current runner added explicit flags after the first experiments. Older runner source matching every frozen hash was not present.

## Modified conditions and final-check ablation

```powershell
.venv/Scripts/python -X utf8 run.py --seeds 5 --facts 100 --out repro_modified_main
.venv/Scripts/python -X utf8 run.py --hard --seeds 5 --facts 100 --out repro_modified_hard_ablation
```

The current `run.py` includes the revisit baselines. It covers the main setting and calibrated stress ablation; it does not recreate the pre-ablation code or the abandoned uncalibrated stress attempts. Configurations for modified results are printed in each saved `summary.md`, rather than separate pre-run configuration files.

## Inspect without training

```powershell
.venv/Scripts/python -B verify_evidence.py
```

This uses only the standard library: checks publication hashes, original-to-public mappings, parses source syntax, compares strict CSV/JSON acceptance counts and checks the stored ablation finding. It does not import or train the learner.

`strict_long_report.py <directory>` reads CSVs and writes a report: use a copy of evidence if regenerating it. `strict_separability.py <directory>` regenerates the synthetic world and evaluates a true-key classifier without training. Neither was run during publication. Strict JSONs contain Python-style `NaN` for unavailable first-fact fields; some strict JSON readers will need to handle these.
