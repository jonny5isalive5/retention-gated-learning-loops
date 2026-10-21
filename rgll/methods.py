"""The learning methods being compared.

  naive    - train k steps on each new fact, nothing else (shows raw forgetting)
  replay   - train k steps on each new fact mixed with replay from memory
  rgll     - Retention-Gated Learning Loop: loop until the fact passes the
             acquisition gate AND an audit of earlier facts passes the retention gate

The fixed-step methods advance because a step counter ran out. RGLL advances only
once the network has demonstrated that it knows the new fact and still knows the old ones.
  joint    - shuffled training on all facts at once (an upper bound, not continual)
"""

import random
import time
from collections import deque
from dataclasses import dataclass, field

import torch

from .data import FactWorld
from .learner import Learner, ReplayMemory, make_batch
from .ledger import Ledger


@dataclass
class TrainConfig:
    hidden: int = 128
    lr: float = 1e-3
    replay_bs: int = 32
    mem_per_item: int = 4
    k: int = 20                 # steps per fact for the fixed-step baselines


@dataclass
class RGLLConfig:
    k: int = 10                 # gradient steps per loop iteration
    consolidate: int = 5        # replay-only steps before the second acquisition check
    tau_acq: float = 0.9        # acquisition gate threshold (probe accuracy)
    tau_ret: float = 0.8        # retention gate threshold for earlier facts
    flag_slack: float = 0.1     # a flagged fact may not drop more than this below where it got to
    max_loops: int = 40
    max_defers: int = 1
    audit_similar: int = 5
    audit_fragile: int = 5
    audit_stale: int = 5
    audit_random: int = 5
    full_sweep_every: int = 10  # every N commits, audit everything
    audit: bool = True          # False = acquisition gate only (ablation)


RETAINED = 0.8   # a fact counts as retained if held-out accuracy on it is >= this


@dataclass
class Result:
    method: str
    seed: int
    final_acc: float = 0.0              # retention accuracy: mean held-out accuracy over all facts
    facts_retained: float = 0.0         # share of facts with held-out accuracy >= RETAINED
    worst_fact_acc: float = 0.0
    forgetting: float = 0.0
    grad_steps: int = 0
    train_samples: int = 0
    probe_samples: int = 0
    compute: int = 0                    # forward-pass equivalents, training + verification
    total_loops: int = 0                # learn->check cycles (fixed methods: one per fact, never checked)
    seconds: float = 0.0
    curve: list = field(default_factory=list)          # mean held-out acc on seen facts, per stream step
    loops_per_fact: list = field(default_factory=list)
    deferred: int = 0
    flagged: int = 0
    retention_failures: int = 0


class Tracker:
    """Records held-out accuracy as facts arrive, for curves and forgetting."""

    def __init__(self, learner: Learner):
        self.learner = learner
        self.seen: list[int] = []
        self.acc_when_learned: dict[int, float] = {}
        self.curve = []

    def after(self, fid: int):
        if fid not in self.seen:
            self.seen.append(fid)
        accs = self.learner.test_acc(self.seen)
        self.acc_when_learned[fid] = accs[fid]
        self.curve.append(sum(accs.values()) / len(accs))

    def finish(self, res: Result, all_fids):
        final = self.learner.test_acc(list(all_fids))
        fill_final(res, final)
        res.forgetting = sum(self.acc_when_learned[f] - final[f] for f in self.acc_when_learned) / len(final)
        res.curve = self.curve
        fill_cost(res, self.learner)


def fill_final(res: Result, final: dict[int, float]):
    res.final_acc = sum(final.values()) / len(final)
    res.facts_retained = sum(a >= RETAINED for a in final.values()) / len(final)
    res.worst_fact_acc = min(final.values())


def fill_cost(res: Result, learner: Learner):
    res.grad_steps = learner.grad_steps
    res.train_samples = learner.train_samples
    res.probe_samples = learner.probe_samples
    res.compute = learner.compute


def run_fixed(world: FactWorld, tc: TrainConfig, seed: int, k: int, replay: bool, name: str) -> Result:
    t0 = time.time()
    rng = random.Random(seed)
    learner = Learner(world, tc.hidden, tc.lr, seed)
    memory = ReplayMemory(world, tc.mem_per_item, rng)
    tracker = Tracker(learner)
    for fid in world.order:
        new = world.xy(fid, "train")
        for _ in range(k):
            learner.step(*make_batch(new, memory.sample(tc.replay_bs) if replay else None))
        memory.add(fid)
        tracker.after(fid)
    res = Result(name, seed, total_loops=len(world.order))
    tracker.finish(res, world.order)
    res.seconds = time.time() - t0
    return res


def run_rgll(world: FactWorld, tc: TrainConfig, rc: RGLLConfig, seed: int, name="rgll") -> Result:
    t0 = time.time()
    rng = random.Random(seed)
    learner = Learner(world, tc.hidden, tc.lr, seed)
    memory = ReplayMemory(world, tc.mem_per_item, rng)
    ledger = Ledger(rng)
    tracker = Tracker(learner)
    res = Result(name, seed)
    queue = deque(world.order)
    defers: dict[int, int] = {}
    commits = 0

    def train(fid, repair, steps, include_new=True):
        new = world.xy(fid, "train") if include_new else None
        rep = make_batch(*[world.xy(r, "train", tc.mem_per_item) for r in repair]) if repair else None
        for _ in range(steps):
            replay = memory.sample(tc.replay_bs)
            if new is None and rep is None and replay is None:
                return
            learner.step(*make_batch(new, rep, replay))

    while queue:
        fid = queue.popleft()
        repair: set[int] = set()
        committed = False
        for loop in range(1, rc.max_loops + 1):
            res.total_loops += 1
            # 1-2. Learn: the new fact + any facts being repaired + replay.
            train(fid, repair, rc.k)

            # 3. Acquisition gate, checked twice: right away, then after some
            #    replay-only training, so "retained" means it survives interference.
            acc, margin, emb = learner.probe(fid)
            if acc < rc.tau_acq:
                continue
            if len(memory.xs) > 0 and rc.consolidate > 0:
                train(fid, repair, rc.consolidate, include_new=False)
                acc, margin, emb = learner.probe(fid)
                if acc < rc.tau_acq:
                    continue

            # 5. Retention audit of earlier facts; failures go into the repair set.
            if rc.audit and len(ledger):
                full = commits > 0 and commits % rc.full_sweep_every == 0
                to_check = ledger.audit_sample(emb, rc.audit_similar, rc.audit_fragile,
                                               rc.audit_stale, rc.audit_random, full) | repair
                failed = set()
                for old in to_check:
                    a, m, e = learner.probe(old)
                    ledger.record(old, a, m, e)
                    if a < ledger.entries[old].bar:
                        failed.add(old)
                if failed:
                    res.retention_failures += len(failed)
                    repair = failed
                    continue

            # 6. Both gates passed: commit and move on.
            ledger.commit(fid, acc, margin, emb, rc.tau_ret, rc.flag_slack)
            committed = True
            break

        if not committed:
            # Escalate: defer once (it may be easier later), otherwise accept and flag it.
            if defers.get(fid, 0) < rc.max_defers:
                defers[fid] = defers.get(fid, 0) + 1
                res.deferred += 1
                queue.append(fid)
                tracker.after(fid)
                continue
            acc, margin, emb = learner.probe(fid)
            ledger.commit(fid, acc, margin, emb, rc.tau_ret, rc.flag_slack, flagged=True)
            res.flagged += 1

        memory.add(fid)
        commits += 1
        res.loops_per_fact.append(loop)
        tracker.after(fid)

    tracker.finish(res, world.order)
    res.seconds = time.time() - t0
    return res


def run_replay_revisit(world: FactWorld, tc: TrainConfig, seed: int, k: int, tau: float, name: str) -> Result:
    """Ablation: fixed-step replay plus ONE retention check at the end and a revisit pass.

    The stream is learned with plain replay (k steps per fact, nothing checked). At the end,
    every fact is probed once; each fact below `tau` gets one more round of k steps (fact +
    replay), in stream order. This mirrors what RGLL does with its deferred facts, without
    RGLL's checking during the stream. It separates "check as you go" from "check once,
    then go back over what is weak".

    (A literal "defer if the new fact fails its own probe" never triggers here: with this many
    steps per fact the new fact always passes. RGLL's deferrals come from failed retention audits.)
    """
    t0 = time.time()
    rng = random.Random(seed)
    learner = Learner(world, tc.hidden, tc.lr, seed)
    memory = ReplayMemory(world, tc.mem_per_item, rng)
    tracker = Tracker(learner)
    res = Result(name, seed)

    def visit(fid):
        new = world.xy(fid, "train")
        for _ in range(k):
            learner.step(*make_batch(new, memory.sample(tc.replay_bs)))
        res.total_loops += 1
        res.loops_per_fact.append(1)

    for fid in world.order:
        visit(fid)
        memory.add(fid)
        tracker.after(fid)

    weak = [fid for fid in world.order if learner.probe(fid)[0] < tau]    # one end-of-stream audit (counted)
    res.deferred = len(weak)
    for fid in weak:
        visit(fid)
        tracker.after(fid)

    tracker.finish(res, world.order)
    res.seconds = time.time() - t0
    return res


def run_joint(world: FactWorld, tc: TrainConfig, seed: int, steps: int) -> Result:
    """Upper bound: all facts available at once, shuffled, same number of steps."""
    t0 = time.time()
    learner = Learner(world, tc.hidden, tc.lr, seed)
    X = world.train_x.reshape(-1, world.cfg.dim)
    Y = world.values.repeat_interleave(world.cfg.n_train)
    g = torch.Generator().manual_seed(seed)
    bs = world.cfg.n_train + tc.replay_bs
    for _ in range(steps):
        idx = torch.randint(len(X), (bs,), generator=g)
        learner.step(X[idx], Y[idx])
    res = Result("joint", seed)
    fill_final(res, learner.test_acc(world.order))
    fill_cost(res, learner)
    res.seconds = time.time() - t0
    return res
