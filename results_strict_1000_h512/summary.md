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
| 0 | **238 / 1000** | **STOPPED at fact 239** | 1,636 | 16,360 | 8.16 M | 97.3% · 64.0% · 98.7% |
| 1 | **149 / 1000** | **STOPPED at fact 150** | 612 | 6,120 | 2.22 M | 98.3% · 80.0% · 100.0% |
| 2 | **326 / 1000** | **STOPPED at fact 327** | 1,807 | 18,070 | 11.19 M | 97.2% · 64.0% · 99.1% |
| 3 | **211 / 1000** | **STOPPED at fact 212** | 825 | 8,250 | 3.68 M | 98.2% · 74.0% · 99.5% |
| 4 | **291 / 1000** | **STOPPED at fact 292** | 1,605 | 16,050 | 9.24 M | 98.2% · 80.0% · 100.0% |

Facts accepted: 238, 149, 326, 211, 291 (min 149, max 326).

## Stopping points

- Seed 0: budget of 200 loops reached at fact 239: 1 accepted fact(s) below 80%.
- Seed 1: budget of 200 loops reached at fact 150: 1 accepted fact(s) below 80%.
- Seed 2: budget of 200 loops reached at fact 327: 1 accepted fact(s) below 80%.
- Seed 3: budget of 200 loops reached at fact 212: 1 accepted fact(s) below 80%.
- Seed 4: budget of 200 loops reached at fact 292: 1 accepted fact(s) below 80%.

A stopping point is the ceiling **under this architecture, training method, thresholds and budget**.
It is not an absolute capacity limit of the network.

## Cost per fact as the accepted set grows (all seeds)

| Facts | Mean loops per fact | Mean training updates per fact | Mean compute per fact |
|---|---|---|---|
| 1–10 | 1.26 | 13 | 1.6 k |
| 11–25 | 1.73 | 17 | 2.8 k |
| 26–50 | 1.97 | 20 | 4.1 k |
| 51–75 | 2.28 | 23 | 6.0 k |
| 76–100 | 2.33 | 23 | 7.4 k |

Per-fact records (loops, updates, compute, gate results, verified retention): `facts_seed<N>.csv`.

## Config (frozen in config.json before training)
```
world:  {'n_facts': 1000, 'n_values': 10, 'dim': 32, 'n_groups': 10, 'group_spread': 0.6, 'noise': 0.35, 'n_train': 8, 'n_probe': 20, 'n_test': 50}
train:  {'hidden': 512, 'lr': 0.001, 'replay_bs': 32, 'mem_per_item': 4, 'k': 20}
strict: {'k': 10, 'tau_new': 0.9, 'tau_old': 0.8, 'max_loops_per_fact': 200}
```