"""Run the RGLL prototype experiment.

    .venv/Scripts/python run.py              # main run (5 seeds, 100 facts)
    .venv/Scripts/python run.py --hard       # stress test: denser facts, smaller network
    .venv/Scripts/python run.py --quick      # smoke test

Writes <out>/summary.md, <out>/summary.json and <out>/rgll_results.png.
"""

import argparse
import json
import statistics as st
from dataclasses import asdict
from pathlib import Path

import torch

from rgll.data import FactWorld, WorldConfig
from rgll.methods import RETAINED, RGLLConfig, TrainConfig, run_fixed, run_joint, run_replay_revisit, run_rgll

ORDER = ["naive", "replay", "replay_steps", "replay_budget", "replay_revisit_steps", "replay_revisit_budget",
         "rgll_acq_only", "rgll", "joint"]
LABELS = {
    "naive": "Naive fine-tuning",
    "replay": "Replay",
    "replay_steps": "Replay, same gradient steps as RGLL",
    "replay_budget": "Replay, same total compute as RGLL",
    "replay_revisit_steps": "Replay + end-of-stream revisit, same steps as RGLL",
    "replay_revisit_budget": "Replay + end-of-stream revisit, same compute as RGLL",
    "rgll_acq_only": "RGLL, acquisition gate only",
    "rgll": "RGLL (both gates)",
    "joint": "Joint shuffled (upper bound)",
}
SHORT = {
    "naive": "Naive", "replay": "Replay", "replay_steps": "Replay\n(=steps)",
    "replay_budget": "Replay\n(=compute)", "replay_revisit_steps": "Revisit\n(=steps)",
    "replay_revisit_budget": "Revisit\n(=compute)", "rgll_acq_only": "RGLL\n(acq. only)",
    "rgll": "RGLL", "joint": "Joint\n(upper bd)",
}
COLORS = {
    "naive": "#c0392b", "replay": "#e67e22", "replay_steps": "#95a5a6", "replay_budget": "#566573",
    "replay_revisit_steps": "#c39bd3", "replay_revisit_budget": "#7d3c98",
    "rgll_acq_only": "#5dade2", "rgll": "#1f618d", "joint": "#27ae60",
}


def run_seed(seed, wc, tc, rc):
    world = FactWorld(wc, seed)
    out = {}
    out["naive"] = run_fixed(world, tc, seed, tc.k, replay=False, name="naive")
    out["replay"] = run_fixed(world, tc, seed, tc.k, replay=True, name="replay")
    out["rgll_acq_only"] = run_rgll(world, tc, RGLLConfig(**{**asdict(rc), "audit": False}), seed, "rgll_acq_only")
    out["rgll"] = run_rgll(world, tc, rc, seed)
    rg = out["rgll"]
    # Fair comparisons: give plain replay the same budget RGLL used, either the same number
    # of gradient steps, or the same total compute (training + the cost of RGLL's checking).
    k_steps = max(1, round(rg.grad_steps / wc.n_facts))
    k_budget = max(1, round(rg.compute / (3 * (wc.n_train + tc.replay_bs) * wc.n_facts)))
    out["replay_steps"] = run_fixed(world, tc, seed, k_steps, replay=True, name="replay_steps")
    out["replay_budget"] = run_fixed(world, tc, seed, k_budget, replay=True, name="replay_budget")
    # Ablation: replay plus one end-of-stream audit and a revisit pass over weak facts, without
    # RGLL's checking during the stream. The budget is split across the expected number of
    # visits (facts + RGLL's own deferrals); actual cost is reported.
    visits = wc.n_facts + rg.deferred
    kd_steps = max(1, round(rg.grad_steps / visits))
    kd_budget = max(1, round(rg.compute / (3 * (wc.n_train + tc.replay_bs) * visits)))
    out["replay_revisit_steps"] = run_replay_revisit(world, tc, seed, kd_steps, rc.tau_acq,
                                                 "replay_revisit_steps")
    out["replay_revisit_budget"] = run_replay_revisit(world, tc, seed, kd_budget, rc.tau_acq,
                                                  "replay_revisit_budget")
    out["joint"] = run_joint(world, tc, seed, rg.grad_steps)
    return out


def mean(xs):
    return st.mean(xs)


def sd(xs):
    return st.stdev(xs) if len(xs) > 1 else 0.0


def pct(x):
    return f"{x * 100:.1f}%"


def summarise(runs, wc, tc, rc, outdir):
    n = len(runs)
    get = lambda m, f: [getattr(r[m], f) for r in runs]

    # 1. Headline table
    head = [
        "| Method | Advances when… | Retention accuracy | Facts retained (≥80%) | Forgetting "
        "| Learning loops | Gradient steps | Total compute | Seed consistency (sd · worst seed) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m in ORDER:
        acc = get(m, "final_acc")
        if m == "joint":
            when, loops, fg = "n/a (not sequential)", "n/a", "n/a"
        elif m.startswith("replay_revisit"):
            k = round(mean(get(m, "grad_steps")) / mean(get(m, "total_loops")))
            when = f"{k} steps done; weak facts revisited once at end"
            loops = f"{mean(get(m, 'total_loops')):.0f} ({mean(get(m, 'deferred')):.0f} revisits, 1 audit at end)"
            fg = pct(mean(get(m, "forgetting")))
        elif m.startswith("rgll"):
            when = "probe check passes" if m == "rgll_acq_only" else "**it has shown it remembers**"
            loops = f"{mean(get(m, 'total_loops')):.0f} ({mean(get(m, 'total_loops')) / wc.n_facts:.2f}/fact)"
            fg = pct(mean(get(m, "forgetting")))
        else:
            k = runs[0][m].grad_steps // wc.n_facts
            when, loops, fg = f"{k} steps done", f"{wc.n_facts} (1/fact, unchecked)", pct(mean(get(m, "forgetting")))
        head.append(
            f"| {LABELS[m]} | {when} | {pct(mean(acc))} ± {sd(acc) * 100:.1f} "
            f"| {pct(mean(get(m, 'facts_retained')))} | {fg} | {loops} "
            f"| {mean(get(m, 'grad_steps')):,.0f} | {mean(get(m, 'compute')) / 1e6:.2f} M "
            f"| {sd(acc) * 100:.1f} pts · {pct(min(acc))} |"
        )

    # 2. Per-seed consistency
    seeds = ["| Seed | " + " | ".join(SHORT[m].replace("\n", " ") for m in ORDER) + " |",
             "|---|" + "---|" * len(ORDER)]
    for i, r in enumerate(runs):
        seeds.append(f"| {i} | " + " | ".join(pct(r[m].final_acc) for m in ORDER) + " |")
    diffs = [r["rgll"].final_acc - r["replay_budget"].final_acc for r in runs]
    verdict = (f"RGLL beat replay given the same total compute on **{sum(d > 0 for d in diffs)} of {n} seeds** "
               f"(mean difference {mean(diffs) * 100:+.1f} pts, per-seed: "
               + ", ".join(f"{d * 100:+.1f}" for d in diffs) + ").")
    for ab in ("replay_revisit_steps", "replay_revisit_budget"):
        da = [r["rgll"].final_acc - r[ab].final_acc for r in runs]
        verdict += (f"\n\nRGLL vs {LABELS[ab].lower()}: RGLL ahead on **{sum(d > 0 for d in da)} of {n} seeds** "
                    f"(mean {mean(da) * 100:+.1f} pts; per-seed " + ", ".join(f"{d * 100:+.1f}" for d in da))
        # Seed 0 was used to calibrate the stress setting, so also report without it.
        verdict += f"; seeds 1–{n - 1} only {mean(da[1:]) * 100:+.1f} pts)." if n > 1 else ")."

    # 3. RGLL detail per seed
    det = ["| Seed | Retention acc. | Facts retained | Forgetting | Loops (total) | Loops/fact: mean · max "
           "| First-try passes | Old facts repaired | Deferred / flagged | Gradient steps | Compute | of which checking |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(runs):
        g = r["rgll"]
        lp = g.loops_per_fact
        det.append(
            f"| {i} | {pct(g.final_acc)} | {pct(g.facts_retained)} | {pct(g.forgetting)} | {g.total_loops} "
            f"| {mean(lp):.2f} · {max(lp)} | {pct(sum(x == 1 for x in lp) / len(lp))} | {g.retention_failures} "
            f"| {g.deferred} / {g.flagged} | {g.grad_steps:,} | {g.compute / 1e6:.2f} M "
            f"| {pct(g.probe_samples / g.compute)} |"
        )
    half = wc.n_facts // 2
    early = mean([mean(r["rgll"].loops_per_fact[:half]) for r in runs])
    late = mean([mean(r["rgll"].loops_per_fact[half:]) for r in runs])

    md = [
        "# RGLL prototype results",
        "",
        "The fixed-step methods move to the next fact because their step counter ran out. "
        "RGLL moves on only when the network has **demonstrated** that it knows the new fact "
        "(acquisition gate) and still knows the earlier ones (retention gate).",
        "",
        f"{n} seeds · {wc.n_facts} facts · {wc.n_values} possible values · {wc.n_groups} key groups. "
        "All accuracies are on held-out test samples that neither training nor the gates ever saw.",
        "",
        "- **Retention accuracy**: mean held-out accuracy over all facts at the end of the stream (± = sd across seeds).",
        f"- **Facts retained**: share of facts still recalled at ≥{RETAINED:.0%} held-out accuracy at the end.",
        "- **Forgetting**: average drop from a fact's accuracy just after it was learned to its accuracy at the end.",
        "- **Learning loops**: learn→check cycles. Fixed-step methods do one unchecked cycle per fact.",
        "- **Total compute**: forward-pass equivalents (training sample = 3, verification probe sample = 1), "
        "so RGLL pays for its checking.",
        "",
        "## Results (mean over seeds)",
        "",
        *head,
        "",
        "## Consistency across seeds (retention accuracy)",
        "",
        *seeds,
        "",
        verdict,
        "",
        "## RGLL per seed",
        "",
        *det,
        "",
        f"Average loops per fact: **{early:.2f}** for the first {half} facts vs **{late:.2f}** for the last "
        f"{wc.n_facts - half}. "
        + ("New facts took more loops as memory filled (interference grows)." if late > early * 1.1 else
           "New facts took fewer loops later on (learning got easier)." if late < early * 0.9 else
           "Loops per fact stayed roughly flat."),
        "",
        "## Configs",
        "```",
        f"world: {asdict(wc)}",
        f"train: {asdict(tc)}",
        f"rgll:  {asdict(rc)}",
        "```",
    ]
    text = "\n".join(md)
    (outdir / "summary.md").write_text(text, encoding="utf-8")
    raw = {m: [asdict(r[m]) for r in runs] for m in ORDER}
    (outdir / "summary.json").write_text(json.dumps(raw, indent=1), encoding="utf-8")
    return text


def plot(runs, wc, outdir, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=13)

    a = ax[0, 0]
    for m in ORDER[:-1]:
        curves = [r[m].curve for r in runs]
        n = min(len(c) for c in curves)
        avg = torch.tensor([c[:n] for c in curves]).mean(0)
        a.plot(avg * 100, label=LABELS[m], color=COLORS[m], lw=2.2 if m == "rgll" else 1.2)
    a.set(title="Retention accuracy on all facts seen so far", xlabel="Facts presented",
          ylabel="Held-out accuracy (%)", ylim=(0, 102))
    a.legend(fontsize=8)

    a = ax[0, 1]
    for i, m in enumerate(ORDER):
        accs = [r[m].final_acc * 100 for r in runs]
        a.scatter([i] * len(accs), accs, color=COLORS[m], s=40, zorder=3)
        a.hlines(mean(accs), i - 0.3, i + 0.3, color=COLORS[m], lw=2)
    a.set_xticks(range(len(ORDER)), [SHORT[m] for m in ORDER], fontsize=8)
    lo = min(r[m].final_acc for r in runs for m in ORDER if m != "naive") * 100
    a.set(title="Final retention accuracy, one dot per seed (bar = mean)", ylabel="%", ylim=(max(0, lo - 5), 101))
    a.annotate("naive off-scale below", (0, max(0, lo - 4)), ha="center", fontsize=7, color=COLORS["naive"])

    a = ax[1, 0]
    for m in ORDER[1:-1]:
        a.scatter([r[m].compute / 1e6 for r in runs], [r[m].final_acc * 100 for r in runs],
                  color=COLORS[m], label=LABELS[m], s=35)
    a.set(title="Retention accuracy vs total training cost (one dot per seed)",
          xlabel="Total compute (millions of forward-pass equivalents)", ylabel="Final accuracy (%)")
    a.legend(fontsize=8)

    a = ax[1, 1]
    w = 10
    for r in runs:
        lp = r["rgll"].loops_per_fact
        smooth = [mean(lp[max(0, i - w + 1):i + 1]) for i in range(len(lp))]
        a.plot(smooth, color=COLORS["rgll"], alpha=0.55)
    a.set(title="RGLL learning loops per fact (10-fact moving average, one line per seed)",
          xlabel="Facts committed", ylabel="Loops until it demonstrated retention")

    fig.tight_layout()
    fig.savefig(outdir / "rgll_results.png", dpi=130)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--facts", type=int, default=100)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--hard", action="store_true",
                   help="stress test: denser facts, smaller network, less replay memory, calibrated gates")
    p.add_argument("--out", default="results")
    args = p.parse_args()
    if args.quick:
        args.seeds, args.facts = 2, 30

    torch.set_num_threads(2)  # keep CPU use light; other workloads may share this machine
    wc, tc, rc = WorldConfig(n_facts=args.facts), TrainConfig(), RGLLConfig()
    title = "RGLL prototype: main setting"
    if args.hard:
        # Calibrated so the task is learnable (joint training ~95%) but hard to learn sequentially.
        wc = WorldConfig(n_facts=args.facts, n_groups=5, group_spread=0.5, noise=0.35)
        tc = TrainConfig(hidden=64, mem_per_item=2)
        # Gates set just under the measured ceiling: a jointly trained net reaches >=80% probe
        # accuracy on ~95% of facts. A 90% gate here is unreachable and the loop never ends.
        rc = RGLLConfig(tau_acq=0.8, tau_ret=0.7, max_loops=15)
        title = "RGLL prototype: stress setting (denser facts, smaller network, less memory)"
    outdir = Path(args.out)
    outdir.mkdir(exist_ok=True)

    runs = []
    for s in range(args.seeds):
        out = run_seed(s, wc, tc, rc)
        runs.append(out)
        print(f"seed {s}: " + "  ".join(f"{m}={out[m].final_acc * 100:.1f}%" for m in ORDER), flush=True)

    print()
    print(summarise(runs, wc, tc, rc, outdir))
    plot(runs, wc, outdir, title)
    print(f"\nWrote {outdir}/summary.md, summary.json, rgll_results.png")


if __name__ == "__main__":
    main()
