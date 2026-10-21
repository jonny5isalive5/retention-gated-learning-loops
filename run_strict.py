"""Strict RGLL experiment: the original hypothesis, nothing else.

    .venv/Scripts/python run_strict.py            # 5 seeds, 100 facts, main-setting world
    .venv/Scripts/python run_strict.py --quick    # smoke test

Learn it. Retain it. Prove it. Then move on. If it cannot retain everything, stop.

Before any training, the configuration and a SHA-256 of the code are written to
<out>/config.json, so the setup is fixed before results exist. Then writes
<out>/facts_seed<N>.csv, <out>/summary.md, <out>/summary.json and <out>/strict_results.png.
"""

import argparse
import csv
import hashlib
import json
import statistics as st
import time
from dataclasses import asdict, fields
from pathlib import Path

import torch

from rgll.data import FactWorld, WorldConfig
from rgll.methods import TrainConfig
from rgll.strict import FactRecord, StrictConfig, run_strict

CODE = ["rgll/strict.py", "rgll/data.py", "rgll/learner.py", "run_strict.py"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pct(x):
    return "n/a" if x != x else f"{x * 100:.1f}%"


def summarise(results, wc, tc, sc, outdir):
    n_seeds = len(results)
    rows = ["| Seed | Facts accepted | Outcome | Total loops | Training updates | Total compute "
            "| Verified retention at last acceptance: mean · min · share ≥80% |",
            "|---|---|---|---|---|---|---|"]
    for r in results:
        last = [x for x in r.records if x.accepted]
        lv = last[-1] if last else None
        outcome = f"**STOPPED at fact {r.facts_presented}**" if r.stopped else "completed all facts"
        rows.append(
            f"| {r.seed} | **{r.facts_accepted} / {wc.n_facts}** | {outcome} "
            f"| {sum(x.loops for x in r.records):,} | {sum(x.grad_steps for x in r.records):,} "
            f"| {sum(x.compute for x in r.records) / 1e6:.2f} M "
            f"| {pct(lv.test_mean) if lv else 'n/a'} · {pct(lv.test_min) if lv else 'n/a'} · "
            f"{pct(lv.test_share_ge_80) if lv else 'n/a'} |"
        )

    stops = [f"- Seed {r.seed}: {r.stop_reason}." for r in results if r.stopped] or ["- No seed stopped."]

    # Cost per fact, by position band (accepted facts only)
    bands = [(1, 10), (11, 25), (26, 50), (51, 75), (76, 100)]
    band_rows = ["| Facts | Mean loops per fact | Mean training updates per fact | Mean compute per fact |",
                 "|---|---|---|---|"]
    for lo, hi in bands:
        recs = [x for r in results for x in r.records if lo <= x.position <= hi and x.accepted]
        if not recs:
            band_rows.append(f"| {lo}–{hi} | (not reached) | | |")
            continue
        band_rows.append(f"| {lo}–{hi} | {st.mean(x.loops for x in recs):.2f} "
                         f"| {st.mean(x.grad_steps for x in recs):.0f} "
                         f"| {st.mean(x.compute for x in recs) / 1e3:.1f} k |")

    accepted_counts = [r.facts_accepted for r in results]
    md = [
        "# Strict RGLL: results",
        "",
        "**Hypothesis (Jonathan Harriss):** present one fact at a time. The network may advance only after it has",
        "demonstrated retention of the new fact **and every previously accepted fact**. If it cannot, it stops.",
        "",
        "**Rules applied:** no deferral, no skipping, no recovery pass. After every learning loop, every fact is",
        f"checked: new fact ≥ {sc.tau_new:.0%} and every accepted fact ≥ {sc.tau_old:.0%} on probe samples it never",
        f"trained on. Budget: {sc.max_loops_per_fact} loops ({sc.max_loops_per_fact * sc.k:,} training updates) per fact;",
        "reaching it means STOP. Retention is also verified on a third, separate test set at every acceptance.",
        "",
        f"{n_seeds} seeds · stream of {wc.n_facts} facts · main-setting world and network (unchanged from `results/`).",
        "",
        "## Outcome",
        "",
        *rows,
        "",
        f"Facts accepted: {', '.join(map(str, accepted_counts))} (min {min(accepted_counts)}, "
        f"max {max(accepted_counts)}).",
        "",
        "## Stopping points",
        "",
        *stops,
        "",
        "A stopping point is the ceiling **under this architecture, training method, thresholds and budget**.",
        "It is not an absolute capacity limit of the network.",
        "",
        "## Cost per fact as the accepted set grows (all seeds)",
        "",
        *band_rows,
        "",
        "Per-fact records (loops, updates, compute, gate results, verified retention): `facts_seed<N>.csv`.",
        "",
        "## Config (frozen in config.json before training)",
        "```",
        f"world:  {asdict(wc)}",
        f"train:  {asdict(tc)}",
        f"strict: {asdict(sc)}",
        "```",
    ]
    text = "\n".join(md)
    (outdir / "summary.md").write_text(text, encoding="utf-8")
    (outdir / "summary.json").write_text(json.dumps(
        [{**{k: v for k, v in asdict(r).items() if k != "records"},
          "records": [asdict(x) for x in r.records]} for r in results], indent=1), encoding="utf-8")
    return text


def plot(results, wc, sc, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Strict RGLL: advance only when every fact so far is demonstrably retained", fontsize=13)
    cmap = plt.get_cmap("viridis")
    for i, r in enumerate(results):
        c = cmap(i / max(1, len(results) - 1))
        pos = [x.position for x in r.records]
        lab = f"seed {r.seed}: {r.facts_accepted} accepted" + (" (stopped)" if r.stopped else "")
        ax[0, 0].plot(pos, [x.loops for x in r.records], color=c, lw=1.2, label=lab)
        ax[0, 1].plot(pos, [sum(y.compute for y in r.records[:j + 1]) / 1e6 for j in range(len(pos))],
                      color=c, lw=1.5, label=lab)
        acc = [x for x in r.records if x.accepted]
        ax[1, 0].plot([x.position for x in acc], [x.test_mean * 100 for x in acc], color=c, lw=1.5)
        ax[1, 0].plot([x.position for x in acc], [x.test_min * 100 for x in acc], color=c, lw=1, ls="--")
        ax[1, 1].plot([x.position for x in acc], [x.test_share_ge_80 * 100 for x in acc], color=c, lw=1.5)
        if r.stopped:
            last = r.records[-1]
            ax[0, 0].scatter([last.position], [last.loops], color="red", marker="X", s=90, zorder=5)
    ax[0, 0].axhline(sc.max_loops_per_fact, color="red", ls=":", lw=1)
    ax[0, 0].set(title=f"Learning loops needed per fact (red line = budget, X = stop)", xlabel="Fact position",
                 ylabel="Loops (log scale)", yscale="log")
    ax[0, 0].legend(fontsize=8)
    ax[0, 1].set(title="Cumulative training cost", xlabel="Fact position",
                 ylabel="Compute (millions of forward-pass equivalents)")
    ax[1, 0].set(title="Independently verified retention at each acceptance (solid = mean, dashed = worst fact)",
                 xlabel="Fact position", ylabel="Test accuracy over accepted facts (%)", ylim=(0, 102))
    ax[1, 0].title.set_fontsize(9)
    ax[1, 1].set(title="Share of accepted facts verified ≥80% on unseen test samples", xlabel="Fact position",
                 ylabel="%", ylim=(0, 102))
    for a in ax.flat:
        a.set_xlim(0, wc.n_facts + 1)
    fig.tight_layout()
    fig.savefig(outdir / "strict_results.png", dpi=130)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--facts", type=int, default=100)
    p.add_argument("--hidden", type=int, default=TrainConfig.hidden, help="hidden units in the MLP")
    p.add_argument("--budget", type=int, default=200, help="max learning loops per fact; reaching it means STOP")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default="results_strict")
    args = p.parse_args()
    if args.quick:
        args.seeds, args.facts = 1, 15

    torch.set_num_threads(2)  # keep CPU use light; other workloads share this machine
    wc, tc, sc = WorldConfig(n_facts=args.facts), TrainConfig(hidden=args.hidden), StrictConfig(max_loops_per_fact=args.budget)
    outdir = Path(args.out)
    outdir.mkdir(exist_ok=True)

    # Freeze the setup before any result exists.
    (outdir / "config.json").write_text(json.dumps({
        "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seeds": list(range(args.seeds)),
        "world": asdict(wc), "train": asdict(tc), "strict": asdict(sc),
        "code_sha256": {f: sha(f) for f in CODE},
    }, indent=1), encoding="utf-8")

    results = []
    for s in range(args.seeds):
        r = run_strict(FactWorld(wc, s), tc, sc, s, log=lambda m: print(m, flush=True))
        results.append(r)
        with open(outdir / f"facts_seed{s}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[fl.name for fl in fields(FactRecord)])
            w.writeheader()
            for x in r.records:
                w.writerow(asdict(x))
        print(f"== seed {s}: {r.facts_accepted}/{wc.n_facts} accepted"
              + (f", STOPPED: {r.stop_reason}" if r.stopped else ", completed") + f" ({r.seconds:.0f}s)", flush=True)

    print()
    print(summarise(results, wc, tc, sc, outdir))
    plot(results, wc, sc, outdir)
    print(f"\nWrote {outdir}/")


if __name__ == "__main__":
    main()
