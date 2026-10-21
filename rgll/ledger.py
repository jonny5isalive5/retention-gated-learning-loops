"""The memory ledger: every committed fact, how well it is known, and when it was last checked."""

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Entry:
    fid: int
    committed_at: int
    last_verified: int
    acc: float
    margin: float
    embed: torch.Tensor
    bar: float              # retention bar: must not drop below what it demonstrated
    flagged: bool = False


class Ledger:
    def __init__(self, rng: random.Random):
        self.entries: dict[int, Entry] = {}
        self.clock = 0          # increases by one per audit, used for staleness
        self.rng = rng

    def __len__(self):
        return len(self.entries)

    def commit(self, fid, acc, margin, embed, tau_ret, slack, flagged=False):
        # A fact is held to the standard it actually demonstrated. Normal commits passed the
        # acquisition gate, so their bar is tau_ret. A flagged fact never got that far; holding
        # it to tau_ret would fail every future audit and stall the whole stream.
        bar = min(tau_ret, acc - slack)
        self.entries[fid] = Entry(fid, self.clock, self.clock, acc, margin, embed, bar, flagged)

    def record(self, fid, acc, margin, embed):
        e = self.entries[fid]
        e.acc, e.margin, e.embed, e.last_verified = acc, margin, embed, self.clock

    def audit_sample(self, new_embed, n_similar, n_fragile, n_stale, n_random, full=False):
        """Pick the committed facts most at risk of having been forgotten."""
        self.clock += 1
        ids = list(self.entries)
        if full or len(ids) <= n_similar + n_fragile + n_stale + n_random:
            return set(ids)
        es = [self.entries[i] for i in ids]
        chosen = set()

        # Similar: facts the network represents like the new one interfere most.
        sims = F.cosine_similarity(torch.stack([e.embed for e in es]), new_embed[None], dim=1)
        chosen |= {ids[i] for i in sims.topk(n_similar).indices.tolist()}
        # Fragile: passed with the smallest margin last time.
        chosen |= {e.fid for e in sorted(es, key=lambda e: e.margin)[:n_fragile]}
        # Stale: not checked for the longest time.
        chosen |= {e.fid for e in sorted(es, key=lambda e: e.last_verified)[:n_stale]}
        # Random: catches failures the heuristics miss.
        rest = [i for i in ids if i not in chosen]
        chosen |= set(self.rng.sample(rest, min(n_random, len(rest))))
        return chosen
