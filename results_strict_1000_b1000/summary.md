# Strict RGLL: results

**Hypothesis (Jonathan Harriss):** present one fact at a time. The network may advance only after it has
demonstrated retention of the new fact **and every previously accepted fact**. If it cannot, it stops.

**Rules applied:** no deferral, no skipping, no recovery pass. After every learning loop, every fact is
checked: new fact ≥ 90% and every accepted fact ≥ 80% on probe samples it never
trained on. Budget: 1000 loops (10,000 training updates) per fact;
reaching it means STOP. Retention is also verified on a third, separate test set at every acceptance.

5 seeds · stream of 1000 facts · main-setting world and network (unchanged from `results/`).

## Outcome

| Seed | Facts accepted | Outcome | Total loops | Training updates | Total compute | Verified retention at last acceptance: mean · min · share ≥80% |
|---|---|---|---|---|---|---|
| 0 | **226 / 1000** | **STOPPED at fact 227** | 2,889 | 28,890 | 15.61 M | 97.1% · 80.0% · 100.0% |
| 1 | **128 / 1000** | **STOPPED at fact 129** | 4,131 | 41,310 | 15.39 M | 97.7% · 80.0% · 100.0% |
| 2 | **232 / 1000** | **STOPPED at fact 233** | 4,008 | 40,080 | 22.98 M | 96.4% · 74.0% · 99.1% |
| 3 | **204 / 1000** | **STOPPED at fact 205** | 1,920 | 19,200 | 9.55 M | 96.6% · 82.0% · 100.0% |
| 4 | **254 / 1000** | **STOPPED at fact 255** | 3,151 | 31,510 | 18.12 M | 97.2% · 80.0% · 100.0% |

Facts accepted: 226, 128, 232, 204, 254 (min 128, max 254).

## Stopping points

- Seed 0: budget of 1000 loops reached at fact 227: 2 accepted fact(s) below 80%.
- Seed 1: budget of 1000 loops reached at fact 129: 2 accepted fact(s) below 80%.
- Seed 2: budget of 1000 loops reached at fact 233: 3 accepted fact(s) below 80%.
- Seed 3: budget of 1000 loops reached at fact 205: 3 accepted fact(s) below 80%.
- Seed 4: budget of 1000 loops reached at fact 255: 2 accepted fact(s) below 80%.

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
strict: {'k': 10, 'tau_new': 0.9, 'tau_old': 0.8, 'max_loops_per_fact': 1000}
```