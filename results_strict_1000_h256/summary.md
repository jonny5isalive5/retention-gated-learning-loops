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
| 0 | **272 / 1000** | **STOPPED at fact 273** | 1,670 | 16,700 | 9.17 M | 96.0% · 66.0% · 98.5% |
| 1 | **147 / 1000** | **STOPPED at fact 148** | 717 | 7,170 | 2.57 M | 97.7% · 72.0% · 99.3% |
| 2 | **128 / 1000** | **STOPPED at fact 129** | 533 | 5,330 | 1.73 M | 98.9% · 88.0% · 100.0% |
| 3 | **226 / 1000** | **STOPPED at fact 227** | 1,240 | 12,400 | 6.02 M | 97.7% · 74.0% · 99.6% |
| 4 | **158 / 1000** | **STOPPED at fact 159** | 994 | 9,940 | 3.84 M | 98.7% · 84.0% · 100.0% |

Facts accepted: 272, 147, 128, 226, 158 (min 128, max 272).

## Stopping points

- Seed 0: budget of 200 loops reached at fact 273: 1 accepted fact(s) below 80%.
- Seed 1: budget of 200 loops reached at fact 148: 1 accepted fact(s) below 80%.
- Seed 2: budget of 200 loops reached at fact 129: 1 accepted fact(s) below 80%.
- Seed 3: budget of 200 loops reached at fact 227: 1 accepted fact(s) below 80%.
- Seed 4: budget of 200 loops reached at fact 159: 1 accepted fact(s) below 80%.

A stopping point is the ceiling **under this architecture, training method, thresholds and budget**.
It is not an absolute capacity limit of the network.

## Cost per fact as the accepted set grows (all seeds)

| Facts | Mean loops per fact | Mean training updates per fact | Mean compute per fact |
|---|---|---|---|
| 1–10 | 1.62 | 16 | 2.1 k |
| 11–25 | 1.89 | 19 | 3.0 k |
| 26–50 | 2.30 | 23 | 4.8 k |
| 51–75 | 2.48 | 25 | 6.5 k |
| 76–100 | 2.60 | 26 | 8.3 k |

Per-fact records (loops, updates, compute, gate results, verified retention): `facts_seed<N>.csv`.

## Config (frozen in config.json before training)
```
world:  {'n_facts': 1000, 'n_values': 10, 'dim': 32, 'n_groups': 10, 'group_spread': 0.6, 'noise': 0.35, 'n_train': 8, 'n_probe': 20, 'n_test': 50}
train:  {'hidden': 256, 'lr': 0.001, 'replay_bs': 32, 'mem_per_item': 4, 'k': 20}
strict: {'k': 10, 'tau_new': 0.9, 'tau_old': 0.8, 'max_loops_per_fact': 200}
```