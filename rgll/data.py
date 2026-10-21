"""Synthetic "fact world": each fact maps a key (a point in input space) to a value.

A fact is one "piece of information". The network never sees the key itself, only
noisy samples around it, so learning a fact means generalising to unseen samples.

Each fact has three disjoint sample sets:
  train  - what the learner trains on (part of it is kept in replay memory)
  probe  - what the RGLL gates check against ("has it really learned it?")
  test   - held out; used only for the final, honest evaluation of every method

Keys are drawn around a few group centres, so facts in the same group are close
together but often have different values. That is what causes interference.
"""

from dataclasses import dataclass

import torch


@dataclass
class WorldConfig:
    n_facts: int = 100
    n_values: int = 10
    dim: int = 32
    n_groups: int = 10
    group_spread: float = 0.6
    noise: float = 0.35
    n_train: int = 8
    n_probe: int = 20
    n_test: int = 50


class FactWorld:
    def __init__(self, cfg: WorldConfig, seed: int):
        self.cfg = cfg
        g = torch.Generator().manual_seed(seed)
        centres = torch.randn(cfg.n_groups, cfg.dim, generator=g)
        self.group = torch.randint(cfg.n_groups, (cfg.n_facts,), generator=g)
        self.keys = centres[self.group] + cfg.group_spread * torch.randn(cfg.n_facts, cfg.dim, generator=g)
        self.values = torch.randint(cfg.n_values, (cfg.n_facts,), generator=g)

        def draw(n):
            return self.keys[:, None, :] + cfg.noise * torch.randn(cfg.n_facts, n, cfg.dim, generator=g)

        self.train_x = draw(cfg.n_train)
        self.probe_x = draw(cfg.n_probe)
        self.test_x = draw(cfg.n_test)
        # The order facts arrive in; identical for every method within a seed.
        self.order = torch.randperm(cfg.n_facts, generator=g).tolist()

    def xy(self, fid: int, split: str, n: int | None = None):
        x = getattr(self, f"{split}_x")[fid]
        if n is not None:
            x = x[:n]
        return x, self.values[fid].repeat(len(x))
