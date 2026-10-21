# Retention-Gated Learning Loops (RGLL) v1

A small, synthetic continual-learning research prototype. **Concept and original strict hypothesis: Jonathan Harriss (20 October 2026). Experimental implementation: Claude Opus (21 October 2026).** Codex prepared this publication from the existing files; no new experiments were run for publication.

> Learn the new fact, verify it alongside every previously accepted fact, and advance only when all gates pass. Otherwise keep learning; if the budget expires without a pass, stop.

This repository records an exploratory hypothesis and its tests. It does not claim a novel learning algorithm, a literature-established priority claim, a solution to catastrophic forgetting, or an absolute measurement of network capacity.

## Two different procedures

**STRICT RGLL** is the original hypothesis: after each loop, the new fact must score at least 90% and **every** accepted fact at least 80% on their probe sets. A loop uses 10 gradient updates, replay and repair of failed facts. The default budget is 200 loops per fact. There are no deferrals, skipped facts or flagged progression. A failure at the budget ends the run. Code: [`rgll/strict.py`](rgll/strict.py), [`run_strict.py`](run_strict.py).

**MODIFIED RGLL** is an earlier implementation: sampled retention audits with periodic full sweeps, capped loops, one deferral, then flagged commitment even if gates fail. It can progress without demonstrated retention. Its stress setting uses 80%/70% gates and 15 loops. Code: [`rgll/methods.py`](rgll/methods.py), [`run.py`](run.py). Results labelled simply “RGLL” in the historical modified reports refer to this variant.

## Existing findings

**Strict, 100 facts:** all five seeds accepted **100/100**. This means every probe gate passed at each acceptance, not that every fact passed an independent test. On 8–26% of acceptances, at least one accepted fact scored below 80% on the separate test samples. See [`results_strict/`](results_strict/summary.md).

**Strict, 1,000-fact streams:** accepted counts below; all seeds stopped. Width comparisons use the same world and order within each seed, but different initial weights. Budget is loops per fact.

| Seed | Width 128, budget 200 | Width 256, budget 200 | Width 512, budget 200 | Width 128, budget 1,000 |
|---|---:|---:|---:|---:|
| 0 | 224 | 272 | 238 | 226 |
| 1 | 106 | 147 | 149 | 128 |
| 2 | 225 | 128 | 326 | 232 |
| 3 | 204 | 226 | 211 | 204 |
| 4 | 252 | 158 | 291 | 254 |
| Mean | 202.2 | 186.2 | 243.0 | 208.8 |
| Median | 224 | 158 | 238 | 226 |

Four times the width increased the mean accepted count by about 20%, but the trend was not monotonic. Five times the budget at width 128 added 33 accepted facts across five seeds (+3.3%), at about 2.8 times the recorded compute. Every long-stream stop had an old-fact retention failure while the new fact passed. These are **ceilings under this method, architecture, thresholds and budget**. They do not establish absolute network capacity or a probability law for stopping.

Detailed records: [128/200](results_strict_1000/long_stream.md), [256/200](results_strict_1000_h256/long_stream.md), [512/200](results_strict_1000_h512/long_stream.md), [128/1,000](results_strict_1000_b1000/long_stream.md). Each directory includes configuration, historical code hashes, per-fact CSVs, JSON and plots. The 100-fact and 1,000-fact worlds are different draws, not one shared prefix.

**Fair baseline ablation, modified variant:** in the calibrated stress condition, replay with one final probe audit and one revisit of weak facts beat modified RGLL on **all five seeds**: 89.3% versus 86.9% mean test accuracy, using 2.27 M versus 3.94 M recorded compute units (about 42% less). The budget allocation was derived from RGLL, but actual expenditure differed; “steps-matched” is the historical allocation label, not identical realized steps. The compute-allocated revisit baseline also won on all five seeds (88.5%). This is evidence against the benefit of continuous gating in this tested modified condition, not a matched strict-RGLL ablation. [Full ablation](results_hard_ablation/summary.md).

## Reproduce and inspect

Python 3.12, CPU PyTorch, NumPy and Matplotlib. Use new output directories to preserve the published evidence:

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements-observed.txt
.venv/Scripts/python -X utf8 run_strict.py --seeds 5 --facts 100 --out repro_strict_100
```

[Reproduction commands](REPRODUCING.md) cover every condition and evidence verification. Commands are provided for readers; they were **not executed** during publication. Historical runner revisions are unavailable, and the environment was recorded only at publication, so bit-for-bit reproduction of earlier runs is not promised.

## Evidence, limitations and rights

Read [limitations](LIMITATIONS.md) and [provenance](PROVENANCE.md). The [original README](README.original.md), all eight available result directories and all eight available logs are retained. Historical reports contain stronger interpretations and sometimes misleading descriptions of modified progression; this README and the limitations supply the corrected scope. The interrupted stress attempts survive as narrative in the original README; separate raw files for those attempts were not found.

The original MS Paint conceptual drawing is pending the owner's exact file path; no substitute has been created. Its absence does not change the experimental record.

No license was supplied. Public availability does not grant a reuse license; see [NOTICE](NOTICE). Third-party dependencies retain their own licenses and are not bundled.
