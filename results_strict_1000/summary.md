# Strict RGLL: results

**Hypothesis (Jonathan Harriss):** present one fact at a time. The network may advance only after it has
demonstrated retention of the new fact **and every previously accepted fact**. If it cannot, it stops.

**Rules applied:** no deferral, no skipping, no recovery pass. After every learning loop, every fact is
checked: new fact ≥ 90% and every accepted fact ≥ 80% on probe samples it never
trained on. Budget: 200 loops (2,000 training updates) per fact;
reaching it means STOP. Retention is also verified on a third, separate test set at every acceptance.

5 seeds · stream of 1000 facts · main-setting world and network (unchanged from `results/`).

## Outcome

| Seed | Facts accepted | Outcome | Total loops | Training updates | Total compute | Verified retention at last acceptance: mean · min · share ≥80% |
|---|---|---|---|---|---|---|
| 0 | **224 / 1000** | **STOPPED at fact 225** | 1,486 | 14,860 | 7.28 M | 96.7% · 76.0% · 99.1% |
| 1 | **106 / 1000** | **STOPPED at fact 107** | 801 | 8,010 | 2.50 M | 98.5% · 74.0% · 98.1% |
| 2 | **225 / 1000** | **STOPPED at fact 226** | 1,339 | 13,390 | 6.60 M | 96.8% · 74.0% · 98.7% |
| 3 | **204 / 1000** | **STOPPED at fact 205** | 1,120 | 11,200 | 5.03 M | 96.6% · 82.0% · 100.0% |
| 4 | **252 / 1000** | **STOPPED at fact 253** | 1,572 | 15,720 | 7.92 M | 95.9% · 74.0% · 98.8% |

Facts accepted: 224, 106, 225, 204, 252 (min 106, max 252).

## Stopping points

- Seed 0: budget of 200 loops reached at fact 225: 2 accepted fact(s) below 80%.
- Seed 1: budget of 200 loops reached at fact 107: 1 accepted fact(s) below 80%.
- Seed 2: budget of 200 loops reached at fact 226: 1 accepted fact(s) below 80%.
- Seed 3: budget of 200 loops reached at fact 205: 3 accepted fact(s) below 80%.
- Seed 4: budget of 200 loops reached at fact 253: 1 accepted fact(s) below 80%.

A stopping point is the ceiling **under this architecture, training method, thresholds and budget**.
It is not an absolute capacity limit of the network.

## Cost per fact as the accepted set grows (all seeds)

| Facts | Mean loops per fact | Mean training updates per fact | Mean compute per fact |
|---|---|---|---|
| 1–10 | 1.78 | 18 | 2.3 k |
| 11–25 | 1.93 | 19 | 3.1 k |
| 26–50 | 2.36 | 24 | 4.9 k |
| 51–75 | 2.66 | 27 | 7.1 k |
| 76–100 | 3.38 | 34 | 10.8 k |

Per-fact records (loops, updates, compute, gate results, verified retention): `facts_seed<N>.csv`.

## Config (frozen in config.json before training)
```
world:  {'n_facts': 1000, 'n_values': 10, 'dim': 32, 'n_groups': 10, 'group_spread': 0.6, 'noise': 0.35, 'n_train': 8, 'n_probe': 20, 'n_test': 50}
train:  {'hidden': 128, 'lr': 0.001, 'replay_bs': 32, 'mem_per_item': 4, 'k': 20}
strict: {'k': 10, 'tau_new': 0.9, 'tau_old': 0.8, 'max_loops_per_fact': 200}
```