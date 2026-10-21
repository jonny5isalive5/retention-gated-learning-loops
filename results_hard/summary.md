# RGLL prototype results

The fixed-step methods move to the next fact because their step counter ran out. RGLL moves on only when the network has **demonstrated** that it knows the new fact (acquisition gate) and still knows the earlier ones (retention gate).

5 seeds · 100 facts · 10 possible values · 5 key groups. All accuracies are on held-out test samples that neither training nor the gates ever saw.

- **Retention accuracy**: mean held-out accuracy over all facts at the end of the stream (± = sd across seeds).
- **Facts retained**: share of facts still recalled at ≥80% held-out accuracy at the end.
- **Forgetting**: average drop from a fact's accuracy just after it was learned to its accuracy at the end.
- **Learning loops**: learn→check cycles. Fixed-step methods do one unchecked cycle per fact.
- **Total compute**: forward-pass equivalents (training sample = 3, verification probe sample = 1), so RGLL pays for its checking.

## Results (mean over seeds)

| Method | Advances when… | Retention accuracy | Facts retained (≥80%) | Forgetting | Learning loops | Gradient steps | Total compute | Seed consistency (sd · worst seed) |
|---|---|---|---|---|---|---|---|---|
| Naive fine-tuning | 20 steps done | 15.0% ± 1.8 | 13.6% | 58.0% | 100 (1/fact, unchecked) | 2,000 | 0.05 M | 1.8 pts · 11.9% |
| Replay | 20 steps done | 74.1% ± 2.2 | 53.8% | 17.4% | 100 (1/fact, unchecked) | 2,000 | 0.24 M | 2.2 pts · 70.8% |
| Replay, same gradient steps as RGLL | 232 steps done | 84.8% ± 1.3 | 74.2% | 13.9% | 100 (1/fact, unchecked) | 22,460 | 2.67 M | 1.3 pts · 83.4% |
| Replay, same total compute as RGLL | 339 steps done | 83.3% ± 1.5 | 70.0% | 15.4% | 100 (1/fact, unchecked) | 32,840 | 3.91 M | 1.5 pts · 80.7% |
| RGLL, acquisition gate only | probe check passes | 75.2% ± 1.8 | 56.0% | 23.8% | 152 (1.52/fact) | 2,017 | 0.23 M | 1.8 pts · 73.1% |
| RGLL (both gates) | **it has shown it remembers** | 86.9% ± 0.7 | 79.0% | 10.3% | 1516 (15.16/fact) | 22,458 | 3.94 M | 0.7 pts · 86.1% |
| Joint shuffled (upper bound) | n/a (not sequential) | 95.1% ± 1.2 | 98.2% | n/a | n/a | 22,458 | 2.69 M | 1.2 pts · 93.3% |

## Consistency across seeds (retention accuracy)

| Seed | Naive | Replay | Replay (=steps) | Replay (=compute) | RGLL (acq. only) | RGLL | Joint (upper bd) |
|---|---|---|---|---|---|---|---|
| 0 | 15.7% | 74.5% | 83.4% | 80.7% | 77.0% | 87.0% | 96.2% |
| 1 | 15.2% | 74.7% | 85.1% | 83.6% | 76.6% | 88.0% | 95.9% |
| 2 | 16.5% | 73.5% | 85.1% | 84.8% | 73.5% | 86.5% | 95.7% |
| 3 | 11.9% | 70.8% | 83.6% | 83.4% | 73.1% | 86.1% | 93.3% |
| 4 | 15.8% | 76.9% | 86.8% | 83.9% | 75.7% | 86.9% | 94.5% |

RGLL beat replay given the same total compute on **5 of 5 seeds** (mean difference +3.6 pts, per-seed: +6.3, +4.5, +1.7, +2.7, +3.0).

## RGLL per seed

| Seed | Retention acc. | Facts retained | Forgetting | Loops (total) | Loops/fact: mean · max | First-try passes | Old facts repaired | Deferred / flagged | Gradient steps | Compute | of which checking |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 87.0% | 82.0% | 10.5% | 1564 | 8.89 · 15 | 3.0% | 8134 | 45 / 45 | 23,225 | 4.06 M | 18.9% |
| 1 | 88.0% | 81.0% | 8.6% | 1073 | 6.53 · 15 | 7.0% | 3609 | 28 / 28 | 15,790 | 2.57 M | 18.6% |
| 2 | 86.5% | 79.0% | 9.9% | 1302 | 7.47 · 15 | 11.0% | 5579 | 37 / 37 | 19,260 | 3.25 M | 18.6% |
| 3 | 86.1% | 73.0% | 11.7% | 1942 | 10.42 · 15 | 6.0% | 11086 | 60 / 60 | 28,840 | 5.40 M | 22.5% |
| 4 | 86.9% | 80.0% | 10.6% | 1697 | 9.32 · 15 | 5.0% | 9159 | 51 / 51 | 25,175 | 4.41 M | 18.3% |

Average loops per fact: **3.70** for the first 50 facts vs **13.36** for the last 50. New facts took more loops as memory filled (interference grows).

## Configs
```
world: {'n_facts': 100, 'n_values': 10, 'dim': 32, 'n_groups': 5, 'group_spread': 0.5, 'noise': 0.35, 'n_train': 8, 'n_probe': 20, 'n_test': 50}
train: {'hidden': 64, 'lr': 0.001, 'replay_bs': 32, 'mem_per_item': 2, 'k': 20}
rgll:  {'k': 10, 'consolidate': 5, 'tau_acq': 0.8, 'tau_ret': 0.7, 'flag_slack': 0.1, 'max_loops': 15, 'max_defers': 1, 'audit_similar': 5, 'audit_fragile': 5, 'audit_stale': 5, 'audit_random': 5, 'full_sweep_every': 10, 'audit': True}
```