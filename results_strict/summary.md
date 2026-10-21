# Strict RGLL: results

**Hypothesis (Jonathan Harriss):** present one fact at a time. The network may advance only after it has
demonstrated retention of the new fact **and every previously accepted fact**. If it cannot, it stops.

**Rules applied:** no deferral, no skipping, no recovery pass. After every learning loop, every fact is
checked: new fact ≥ 90% and every accepted fact ≥ 80% on probe samples it never
trained on. Budget: 200 loops (2,000 training updates) per fact;
reaching it means STOP. Retention is also verified on a third, separate test set at every acceptance.

5 seeds · stream of 100 facts · main-setting world and network (unchanged from `results/`).

## Outcome

| Seed | Facts accepted | Outcome | Total loops | Training updates | Total compute | Verified retention at last acceptance: mean · min · share ≥80% |
|---|---|---|---|---|---|---|
| 0 | **100 / 100** | completed all facts | 244 | 2,440 | 0.61 M | 97.2% · 64.0% · 98.0% |
| 1 | **100 / 100** | completed all facts | 261 | 2,610 | 0.65 M | 97.5% · 78.0% · 98.0% |
| 2 | **100 / 100** | completed all facts | 253 | 2,530 | 0.62 M | 97.6% · 78.0% · 99.0% |
| 3 | **100 / 100** | completed all facts | 247 | 2,470 | 0.61 M | 98.0% · 80.0% · 100.0% |
| 4 | **100 / 100** | completed all facts | 261 | 2,610 | 0.65 M | 97.8% · 82.0% · 100.0% |

Facts accepted: 100, 100, 100, 100, 100 (min 100, max 100).

## Stopping points

- No seed stopped.

A stopping point is the ceiling **under this architecture, training method, thresholds and budget**.
It is not an absolute capacity limit of the network.

## Cost per fact as the accepted set grows (all seeds)

| Facts | Mean loops per fact | Mean training updates per fact | Mean compute per fact |
|---|---|---|---|
| 1–10 | 1.70 | 17 | 2.1 k |
| 11–25 | 2.15 | 21 | 3.5 k |
| 26–50 | 2.37 | 24 | 5.0 k |
| 51–75 | 2.62 | 26 | 7.0 k |
| 76–100 | 3.17 | 32 | 10.2 k |

Per-fact records (loops, updates, compute, gate results, verified retention): `facts_seed<N>.csv`.

## Config (frozen in config.json before training)
```
world:  {'n_facts': 100, 'n_values': 10, 'dim': 32, 'n_groups': 10, 'group_spread': 0.6, 'noise': 0.35, 'n_train': 8, 'n_probe': 20, 'n_test': 50}
train:  {'hidden': 128, 'lr': 0.001, 'replay_bs': 32, 'mem_per_item': 4, 'k': 20}
strict: {'k': 10, 'tau_new': 0.9, 'tau_old': 0.8, 'max_loops_per_fact': 200}
```