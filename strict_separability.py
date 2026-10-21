"""Is the data itself still separable at a strict run's stopping point? (Trains nothing.)

    .venv/Scripts/python strict_separability.py results_strict_1000

For each seed, rebuilds the world and scores the Bayes-optimal classifier (true keys and noise level known,
equal priors) on the probe and test samples of every fact presented up to the stop, among those facts only.
100% means the facts presented so far do not overlap in the data, so a stop comes from the learner, not the world.
"""

import csv
import json
import sys
from pathlib import Path

import torch

from rgll.data import FactWorld, WorldConfig


def main():
    torch.set_num_threads(2)
    out = Path(sys.argv[1])
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    wc = WorldConfig(**cfg["world"])
    for s in cfg["seeds"]:
        p = out / f"facts_seed{s}.csv"
        if not p.exists():
            continue
        n = sum(1 for _ in csv.DictReader(open(p, encoding="utf-8")))
        w = FactWorld(wc, s)
        ids = torch.tensor(w.order[:n])
        keys, vals = w.keys[ids], w.values[ids]

        def acc(split):
            x = getattr(w, f"{split}_x")[ids].reshape(-1, wc.dim)
            post = torch.softmax(-torch.cdist(x, keys) ** 2 / (2 * wc.noise ** 2), dim=1)
            by_value = torch.zeros(len(x), wc.n_values).index_add_(1, vals, post)
            return (by_value.argmax(1).reshape(n, -1) == vals[:, None]).float().mean(1)

        pa, ta = acc("probe"), acc("test")
        print(f"seed {s}: {n} facts presented; Bayes-optimal probe acc min {pa.min():.0%}, "
              f"test acc mean {ta.mean():.1%} min {ta.min():.0%}; facts below 80% on probe {int((pa < 0.8).sum())}")


if __name__ == "__main__":
    main()
