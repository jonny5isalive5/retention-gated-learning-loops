# RGLL prototype results

The fixed-step methods move to the next fact because their step counter ran out. RGLL moves on only when the network has **demonstrated** that it knows the new fact (acquisition gate) and still knows the earlier ones (retention gate).

5 seeds · 100 facts · 10 possible values · 10 key groups. All accuracies are on held-out test samples that neither training nor the gates ever saw.

- **Retention accuracy**: mean held-out accuracy over all facts at the end of the stream (± = sd across seeds).
- **Facts retained**: share of facts still recalled at ≥80% held-out accuracy at the end.
- **Forgetting**: average drop from a fact's accuracy just after it was learned to its accuracy at the end.
- **Learning loops**: learn→check cycles. Fixed-step methods do one unchecked cycle per fact.
- **Total compute**: forward-pass equivalents (training sample = 3, verification probe sample = 1), so RGLL pays for its checking.

## Results (mean over seeds)

| Method | Advances when… | Retention accuracy | Facts retained (≥80%) | Forgetting | Learning loops | Gradient steps | Total compute | Seed consistency (sd · worst seed) |
|---|---|---|---|---|---|---|---|---|
| Naive fine-tuning | 20 steps done | 20.8% ± 2.4 | 19.0% | 73.8% | 100 (1/fact, unchecked) | 2,000 | 0.05 M | 2.4 pts · 18.6% |
| Replay | 20 steps done | 93.6% ± 1.2 | 93.8% | 5.4% | 100 (1/fact, unchecked) | 2,000 | 0.24 M | 1.2 pts · 92.1% |
| Replay, same gradient steps as RGLL | 34 steps done | 96.6% ± 0.8 | 97.0% | 3.2% | 100 (1/fact, unchecked) | 3,280 | 0.39 M | 0.8 pts · 95.5% |
| Replay, same total compute as RGLL | 42 steps done | 97.3% ± 0.4 | 98.8% | 2.5% | 100 (1/fact, unchecked) | 4,120 | 0.49 M | 0.4 pts · 96.6% |
| RGLL, acquisition gate only | probe check passes | 91.2% ± 1.8 | 87.6% | 8.8% | 131 (1.31/fact) | 1,803 | 0.21 M | 1.8 pts · 89.3% |
| RGLL (both gates) | **it has shown it remembers** | 96.9% ± 1.3 | 98.4% | 2.3% | 231 (2.31/fact) | 3,273 | 0.50 M | 1.3 pts · 94.8% |
| Joint shuffled (upper bound) | n/a (not sequential) | 99.5% ± 0.1 | 100.0% | n/a | n/a | 3,273 | 0.39 M | 0.1 pts · 99.4% |

## Consistency across seeds (retention accuracy)

| Seed | Naive | Replay | Replay (=steps) | Replay (=compute) | RGLL (acq. only) | RGLL | Joint (upper bd) |
|---|---|---|---|---|---|---|---|
| 0 | 19.4% | 92.1% | 95.5% | 96.6% | 90.6% | 97.3% | 99.5% |
| 1 | 22.8% | 94.3% | 97.1% | 97.3% | 89.3% | 97.2% | 99.6% |
| 2 | 23.8% | 93.4% | 97.6% | 97.5% | 93.4% | 94.8% | 99.5% |
| 3 | 18.6% | 95.2% | 96.2% | 97.3% | 89.7% | 96.9% | 99.5% |
| 4 | 19.1% | 93.1% | 96.7% | 97.6% | 92.8% | 98.1% | 99.4% |

RGLL beat replay given the same total compute on **2 of 5 seeds** (mean difference -0.4 pts, per-seed: +0.6, -0.1, -2.7, -0.4, +0.5).

## RGLL per seed

| Seed | Retention acc. | Facts retained | Forgetting | Loops (total) | Loops/fact: mean · max | First-try passes | Old facts repaired | Deferred / flagged | Gradient steps | Compute | of which checking |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 97.3% | 99.0% | 2.1% | 241 | 2.41 · 7 | 22.0% | 167 | 0 / 0 | 3,360 | 0.51 M | 17.0% |
| 1 | 97.2% | 99.0% | 1.8% | 234 | 2.34 · 6 | 15.0% | 185 | 0 / 0 | 3,330 | 0.51 M | 17.4% |
| 2 | 94.8% | 96.0% | 4.3% | 234 | 2.34 · 6 | 24.0% | 169 | 0 / 0 | 3,290 | 0.50 M | 17.3% |
| 3 | 96.9% | 98.0% | 2.5% | 219 | 2.19 · 5 | 16.0% | 168 | 0 / 0 | 3,130 | 0.47 M | 18.0% |
| 4 | 98.1% | 100.0% | 0.9% | 229 | 2.29 · 6 | 19.0% | 161 | 0 / 0 | 3,255 | 0.49 M | 17.3% |

Average loops per fact: **2.08** for the first 50 facts vs **2.54** for the last 50. New facts took more loops as memory filled (interference grows).

## Configs
```
world: {'n_facts': 100, 'n_values': 10, 'dim': 32, 'n_groups': 10, 'group_spread': 0.6, 'noise': 0.35, 'n_train': 8, 'n_probe': 20, 'n_test': 50}
train: {'hidden': 128, 'lr': 0.001, 'replay_bs': 32, 'mem_per_item': 4, 'k': 20}
rgll:  {'k': 10, 'consolidate': 5, 'tau_acq': 0.9, 'tau_ret': 0.8, 'max_loops': 40, 'max_defers': 1, 'audit_similar': 5, 'audit_fragile': 5, 'audit_stale': 5, 'audit_random': 5, 'full_sweep_every': 10, 'audit': True}
```