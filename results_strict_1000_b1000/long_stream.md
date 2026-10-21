# Strict RGLL, 1000-fact stream: long-stream report

Seeds run: 0, 1, 2, 3, 4. Thresholds: new fact ≥90%, accepted facts ≥80% (probe samples). Budget 1000 loops per fact; reaching it means STOP.

## Per seed

| Seed | Facts accepted | Outcome | At the stop | Loops | Updates | Compute | Most loops (accepted fact) | Verified retention at last acceptance (test): mean · min · share ≥80% | Wall time |
|---|---|---|---|---|---|---|---|---|---|
| 0 | **226 / 1000** | **STOPPED at fact 227** | new fact passed (100.0%); 2 of 226 accepted facts below 80% (worst 75.0%) | 2,889 | 28,890 | 15.6 M | 305 | 97.1% · 80.0% · 100.0% | 4.0 min |
| 1 | **128 / 1000** | **STOPPED at fact 129** | new fact passed (100.0%); 2 of 128 accepted facts below 80% (worst 70.0%) | 4,131 | 41,310 | 15.4 M | 489 | 97.7% · 80.0% · 100.0% | 3.4 min |
| 2 | **232 / 1000** | **STOPPED at fact 233** | new fact passed (100.0%); 3 of 232 accepted facts below 80% (worst 70.0%) | 4,008 | 40,080 | 23.0 M | 687 | 96.4% · 74.0% · 99.1% | 5.3 min |
| 3 | **204 / 1000** | **STOPPED at fact 205** | new fact passed (100.0%); 3 of 204 accepted facts below 80% (worst 70.0%) | 1,920 | 19,200 | 9.6 M | 116 | 96.6% · 82.0% · 100.0% | 2.2 min |
| 4 | **254 / 1000** | **STOPPED at fact 255** | new fact passed (100.0%); 2 of 254 accepted facts below 80% (worst 75.0%) | 3,151 | 31,510 | 18.1 M | 499 | 97.2% · 80.0% · 100.0% | 4.3 min |

## Cost per accepted fact by position (all seeds run)

| Facts | Facts in band | Mean loops | Max loops | Mean updates | Mean compute | Verified at acceptance (test): mean · min |
|---|---|---|---|---|---|---|
| 1–10 | 50 | 1.78 | 3 | 18 | 2.3 k | 99.8% · 88.0% |
| 11–100 | 450 | 2.66 | 15 | 27 | 6.8 k | 98.7% · 56.0% |
| 101–200 | 428 | 11.08 | 489 | 111 | 45.6 k | 97.1% · 56.0% |
| 201–300 | 116 | 43.73 | 687 | 437 | 263.6 k | 96.5% · 42.0% |
| 301–1000 | 0 | (not reached by any seed) | | | | |

A stopping point is the ceiling **under this architecture, training method, thresholds and budget**, not an absolute capacity limit of the network.
