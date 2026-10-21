# Strict RGLL, 1000-fact stream: long-stream report

Seeds run: 0, 1, 2, 3, 4. Thresholds: new fact ≥90%, accepted facts ≥80% (probe samples). Budget 200 loops per fact; reaching it means STOP.

## Per seed

| Seed | Facts accepted | Outcome | At the stop | Loops | Updates | Compute | Most loops (accepted fact) | Verified retention at last acceptance (test): mean · min · share ≥80% | Wall time |
|---|---|---|---|---|---|---|---|---|---|
| 0 | **224 / 1000** | **STOPPED at fact 225** | new fact passed (100.0%); 2 of 224 accepted facts below 80% (worst 75.0%) | 1,486 | 14,860 | 7.3 M | 105 | 96.7% · 76.0% · 99.1% | 1.8 min |
| 1 | **106 / 1000** | **STOPPED at fact 107** | new fact passed (100.0%); 1 of 106 accepted facts below 80% (worst 75.0%) | 801 | 8,010 | 2.5 M | 186 | 98.5% · 74.0% · 98.1% | 0.6 min |
| 2 | **225 / 1000** | **STOPPED at fact 226** | new fact passed (100.0%); 1 of 225 accepted facts below 80% (worst 75.0%) | 1,339 | 13,390 | 6.6 M | 86 | 96.8% · 74.0% · 98.7% | 1.6 min |
| 3 | **204 / 1000** | **STOPPED at fact 205** | new fact passed (100.0%); 3 of 204 accepted facts below 80% (worst 65.0%) | 1,120 | 11,200 | 5.0 M | 116 | 96.6% · 82.0% · 100.0% | 1.3 min |
| 4 | **252 / 1000** | **STOPPED at fact 253** | new fact passed (95.0%); 1 of 252 accepted facts below 80% (worst 75.0%) | 1,572 | 15,720 | 7.9 M | 113 | 95.9% · 74.0% · 98.8% | 2.1 min |

## Cost per accepted fact by position (all seeds run)

| Facts | Facts in band | Mean loops | Max loops | Mean updates | Mean compute | Verified at acceptance (test): mean · min |
|---|---|---|---|---|---|---|
| 1–10 | 50 | 1.78 | 3 | 18 | 2.3 k | 99.8% · 88.0% |
| 11–100 | 450 | 2.66 | 15 | 27 | 6.8 k | 98.7% · 56.0% |
| 101–200 | 406 | 5.45 | 186 | 54 | 24.4 k | 97.0% · 56.0% |
| 201–300 | 105 | 17.35 | 116 | 174 | 102.3 k | 96.4% · 42.0% |
| 301–1000 | 0 | (not reached by any seed) | | | | |

A stopping point is the ceiling **under this architecture, training method, thresholds and budget**, not an absolute capacity limit of the network.
