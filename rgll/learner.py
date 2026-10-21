"""The learner network, its training step, and the replay memory shared by methods."""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import FactWorld


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int, out: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, out)

    def forward(self, x):
        return self.head(self.body(x))


class Learner:
    """Wraps the model and counts every gradient step and every probe evaluation,
    so methods can be compared at equal compute."""

    def __init__(self, world: FactWorld, hidden: int, lr: float, seed: int):
        torch.manual_seed(seed)
        self.world = world
        self.model = MLP(world.cfg.dim, hidden, world.cfg.n_values)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.grad_steps = 0
        self.train_samples = 0      # samples pushed through forward + backward
        self.probe_samples = 0      # samples pushed through forward only (verification)

    def step(self, x, y):
        self.model.train()
        loss = F.cross_entropy(self.model(x), y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.grad_steps += 1
        self.train_samples += len(x)

    @property
    def compute(self) -> int:
        """Total cost in forward-pass equivalents: a training sample costs ~3
        (forward + backward), a verification probe sample costs 1."""
        return 3 * self.train_samples + self.probe_samples

    @torch.no_grad()
    def probe(self, fid: int):
        """Check one fact on its probe set. Returns (accuracy, margin, embedding)."""
        self.model.eval()
        x, y = self.world.xy(fid, "probe")
        self.probe_samples += len(x)
        h = self.model.body(x)
        p = self.model.head(h).softmax(-1)
        acc = (p.argmax(-1) == y).float().mean().item()
        p_true = p.gather(1, y[:, None]).squeeze(1)
        p_other = p.scatter(1, y[:, None], -1.0).max(1).values
        margin = (p_true - p_other).mean().item()
        return acc, margin, h.mean(0)

    @torch.no_grad()
    def test_acc(self, fids: list[int]) -> dict[int, float]:
        """Held-out accuracy per fact. Evaluation only; not counted as compute."""
        if not fids:
            return {}
        self.model.eval()
        x = self.world.test_x[fids]                     # (n, n_test, dim)
        y = self.world.values[fids][:, None]            # (n, 1)
        acc = (self.model(x).argmax(-1) == y).float().mean(1)
        return dict(zip(fids, acc.tolist()))


class ReplayMemory:
    """Stores the first `per_item` training samples of each committed fact."""

    def __init__(self, world: FactWorld, per_item: int, rng: random.Random):
        self.world, self.per_item, self.rng = world, per_item, rng
        self.xs, self.ys = [], []
        self._cache = None

    def add(self, fid: int):
        x, y = self.world.xy(fid, "train", self.per_item)
        self.xs.append(x)
        self.ys.append(y)
        self._cache = None

    def sample(self, n: int):
        if not self.xs:
            return None
        if self._cache is None:
            self._cache = (torch.cat(self.xs), torch.cat(self.ys))
        X, Y = self._cache
        idx = torch.tensor([self.rng.randrange(len(X)) for _ in range(n)])
        return X[idx], Y[idx]


def make_batch(*parts):
    parts = [p for p in parts if p is not None]
    return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])
