# Strict RGLL, 1000-fact stream: long-stream report

Seeds run: 0, 1, 2, 3, 4. Thresholds: new fact ≥90%, accepted facts ≥80% (probe samples). Budget 200 loops per fact; reaching it means STOP.

## Per seed

| Seed | Facts accepted | Outcome | At the stop | Loops | Updates | Compute | Most loops (accepted fact) | Verified retention at last acceptance (test): mean · min · share ≥80% | Wall time |
|---|---|---|---|---|---|---|---|---|---|
| 0 | **272 / 1000** | **STOPPED at fact 273** | new fact passed (100.0%); 1 of 272 accepted facts below 80% (worst 75.0%) | 1,670 | 16,700 | 9.2 M | 111 | 96.0% · 66.0% · 98.5% | 2.8 min |
| 1 | **147 / 1000** | **STOPPED at fact 148** | new fact passed (100.0%); 1 of 147 accepted facts below 80% (worst 70.0%) | 717 | 7,170 | 2.6 M | 50 | 97.7% · 72.0% · 99.3% | 0.8 min |
| 2 | **128 / 1000** | **STOPPED at fact 129** | new fact passed (100.0%); 1 of 128 accepted facts below 80% (worst 70.0%) | 533 | 5,330 | 1.7 M | 22 | 98.9% · 88.0% · 100.0% | 0.6 min |
| 3 | **226 / 1000** | **STOPPED at fact 227** | new fact passed (100.0%); 1 of 226 accepted facts below 80% (worst 70.0%) | 1,240 | 12,400 | 6.0 M | 102 | 97.7% · 74.0% · 99.6% | 1.6 min |
| 4 | **158 / 1000** | **STOPPED at fact 159** | new fact passed (100.0%); 1 of 158 accepted facts below 80% (worst 75.0%) | 994 | 9,940 | 3.8 M | 101 | 98.7% · 84.0% · 100.0% | 0.9 min |

## Cost per accepted fact by position (all seeds run)

| Facts | Facts in band | Mean loops | Max loops | Mean updates | Mean compute | Verified at acceptance (test): mean · min |
|---|---|---|---|---|---|---|
| 1–10 | 50 | 1.62 | 3 | 16 | 2.1 k | 99.8% · 76.0% |
| 11–100 | 450 | 2.36 | 6 | 24 | 5.9 k | 99.1% · 40.0% |
| 101–200 | 333 | 5.20 | 101 | 52 | 22.8 k | 98.0% · 54.0% |
| 201–300 | 98 | 13.02 | 111 | 130 | 80.3 k | 97.3% · 58.0% |
| 301–1000 | 0 | (not reached by any seed) | | | | |

A stopping point is the ceiling **under this architecture, training method, thresholds and budget**, not an absolute capacity limit of the network.
