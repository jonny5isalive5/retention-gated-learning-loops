"""Long-stream report for a strict RGLL run (reads <out>/facts_seed<N>.csv; trains nothing).

    .venv/Scripts/python strict_long_report.py results_strict_1000

run_strict.py's own summary bins cost per fact only up to fact 100. This adds bins across the whole
stream and the details of any stop, and writes <out>/long_stream.md. The run's code is untouched, so
its config.json hashes still match results_strict/.
"""

import csv
import json
import statistics as st
import sys
from pathlib import Path


def load(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for r in rows:
        r["accepted"] = r["accepted"] == "True"
        for k in ("position", "loops", "grad_steps", "train_samples", "probe_samples", "compute", "old_facts_failing"):
            r[k] = int(r[k])
        for k in ("seconds", "new_fact_probe_acc", "min_old_probe_acc", "test_mean", "test_min", "test_share_ge_80"):
            r[k] = float(r[k])
    return rows


def pct(x):
    return "n/a" if x != x else f"{x * 100:.1f}%"


def main():
    out = Path(sys.argv[1])
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    n, budget = cfg["world"]["n_facts"], cfg["strict"]["max_loops_per_fact"]
    tau_new, tau_old = cfg["strict"]["tau_new"], cfg["strict"]["tau_old"]
    seeds = {s: load(out / f"facts_seed{s}.csv") for s in cfg["seeds"] if (out / f"facts_seed{s}.csv").exists()}

    md = [f"# Strict RGLL, {n}-fact stream: long-stream report", "",
          f"Seeds run: {', '.join(map(str, seeds))}. Thresholds: new fact ≥{tau_new:.0%}, accepted facts ≥{tau_old:.0%} "
          f"(probe samples). Budget {budget} loops per fact; reaching it means STOP.", "",
          "## Per seed", "",
          "| Seed | Facts accepted | Outcome | At the stop | Loops | Updates | Compute | Most loops (accepted fact) "
          "| Verified retention at last acceptance (test): mean · min · share ≥80% | Wall time |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for s, rows in seeds.items():
        acc = [r for r in rows if r["accepted"]]
        last, lv = rows[-1], (acc[-1] if acc else None)
        stopped = not last["accepted"]
        if stopped:
            parts = []
            if last["new_fact_probe_acc"] < tau_new:
                parts.append(f"new fact at {pct(last['new_fact_probe_acc'])}")
            else:
                parts.append(f"new fact passed ({pct(last['new_fact_probe_acc'])})")
            parts.append(f"{last['old_facts_failing']} of {len(acc)} accepted facts below {tau_old:.0%} "
                         f"(worst {pct(last['min_old_probe_acc'])})")
            at_stop = "; ".join(parts)
        else:
            at_stop = "n/a"
        md.append(
            f"| {s} | **{len(acc)} / {n}** | {'**STOPPED at fact ' + str(last['position']) + '**' if stopped else 'completed'} "
            f"| {at_stop} | {sum(r['loops'] for r in rows):,} | {sum(r['grad_steps'] for r in rows):,} "
            f"| {sum(r['compute'] for r in rows) / 1e6:.1f} M | {max((r['loops'] for r in acc), default=0)} "
            f"| {pct(lv['test_mean']) if lv else 'n/a'} · {pct(lv['test_min']) if lv else 'n/a'} · "
            f"{pct(lv['test_share_ge_80']) if lv else 'n/a'} | {sum(r['seconds'] for r in rows) / 60:.1f} min |")

    step = max(1, n // 10)
    bands = [(1, 10), (11, step)] + [(lo, lo + step - 1) for lo in range(step + 1, n + 1, step)]
    md += ["", "## Cost per accepted fact by position (all seeds run)", "",
           "| Facts | Facts in band | Mean loops | Max loops | Mean updates | Mean compute | "
           "Verified at acceptance (test): mean · min |", "|---|---|---|---|---|---|---|"]
    for lo, hi in bands:
        recs = [r for rows in seeds.values() for r in rows if lo <= r["position"] <= hi and r["accepted"]]
        if not recs:
            md.append(f"| {lo}–{n} | 0 | (not reached by any seed) | | | | |")
            break
        md.append(f"| {lo}–{hi} | {len(recs)} | {st.mean(r['loops'] for r in recs):.2f} | {max(r['loops'] for r in recs)} "
                  f"| {st.mean(r['grad_steps'] for r in recs):.0f} | {st.mean(r['compute'] for r in recs) / 1e3:.1f} k "
                  f"| {pct(st.mean(r['test_mean'] for r in recs))} · {pct(min(r['test_min'] for r in recs))} |")

    md += ["", "A stopping point is the ceiling **under this architecture, training method, thresholds and budget**, "
               "not an absolute capacity limit of the network."]
    (out / "long_stream.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
