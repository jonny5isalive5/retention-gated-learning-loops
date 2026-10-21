# Strict RGLL, 1000-fact stream: long-stream report

Seeds run: 0, 1, 2, 3, 4. Thresholds: new fact ≥90%, accepted facts ≥80% (probe samples). Budget 200 loops per fact; reaching it means STOP.

## Per seed

| Seed | Facts accepted | Outcome | At the stop | Loops | Updates | Compute | Most loops (accepted fact) | Verified retention at last acceptance (test): mean · min · share ≥80% | Wall time |
|---|---|---|---|---|---|---|---|---|---|
| 0 | **238 / 1000** | **STOPPED at fact 239** | new fact passed (100.0%); 1 of 238 accepted facts below 80% (worst 75.0%) | 1,636 | 16,360 | 8.2 M | 102 | 97.3% · 64.0% · 98.7% | 3.5 min |
| 1 | **149 / 1000** | **STOPPED at fact 150** | new fact passed (100.0%); 1 of 149 accepted facts below 80% (worst 65.0%) | 612 | 6,120 | 2.2 M | 59 | 98.3% · 80.0% · 100.0% | 1.0 min |
| 2 | **326 / 1000** | **STOPPED at fact 327** | new fact passed (100.0%); 1 of 326 accepted facts below 80% (worst 60.0%) | 1,807 | 18,070 | 11.2 M | 105 | 97.2% · 64.0% · 99.1% | 4.2 min |
| 3 | **211 / 1000** | **STOPPED at fact 212** | new fact passed (100.0%); 1 of 211 accepted facts below 80% (worst 75.0%) | 825 | 8,250 | 3.7 M | 72 | 98.2% · 74.0% · 99.5% | 1.6 min |
| 4 | **291 / 1000** | **STOPPED at fact 292** | new fact passed (100.0%); 1 of 291 accepted facts below 80% (worst 75.0%) | 1,605 | 16,050 | 9.2 M | 180 | 98.2% · 80.0% · 100.0% | 3.4 min |

## Cost per accepted fact by position (all seeds run)

| Facts | Facts in band | Mean loops | Max loops | Mean updates | Mean compute | Verified at acceptance (test): mean · min |
|---|---|---|---|---|---|---|
| 1–10 | 50 | 1.26 | 3 | 13 | 1.6 k | 99.8% · 80.0% |
| 11–100 | 450 | 2.12 | 5 | 21 | 5.3 k | 99.3% · 52.0% |
| 101–200 | 449 | 4.34 | 102 | 43 | 19.7 k | 98.4% · 54.0% |
| 201–300 | 240 | 8.40 | 180 | 84 | 54.1 k | 97.8% · 50.0% |
| 301–400 | 26 | 19.38 | 105 | 194 | 151.8 k | 97.3% · 60.0% |
| 401–1000 | 0 | (not reached by any seed) | | | | |

A stopping point is the ceiling **under this architecture, training method, thresholds and budget**, not an absolute capacity limit of the network.
