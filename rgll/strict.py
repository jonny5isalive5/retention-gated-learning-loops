"""Strict RGLL: Jonathan Harriss's original hypothesis, implemented literally.

    Present one fact. Train until it AND every previously accepted fact are demonstrably
    retained. Only then advance. If that cannot be achieved, stop.

Rules:
  - One new fact at a time, in a fixed order.
  - Each learning loop = k training updates on the new fact, the stored samples of any
    accepted fact that failed the last check, and replay from accepted facts.
  - After every loop, EVERY fact is checked on probe samples it was never trained on:
    the new fact against tau_new, every accepted fact against tau_old.
  - Advance only when every check passes. No deferral, no skipping, no recovery pass.
  - A per-fact budget prevents infinite execution. Reaching it means STOP, not advance.

Retention is also verified independently: at each acceptance, every accepted fact is tested
on a third sample set (test) that is used for neither training nor gating.
"""

import random
import time
from dataclasses import dataclass, field

from .data import FactWorld
from .learner import Learner, ReplayMemory, make_batch
from .methods import TrainConfig


@dataclass
class StrictConfig:
    k: int = 10                       # training updates per learning loop
    tau_new: float = 0.9              # the new fact must reach this on its probe samples
    tau_old: float = 0.8              # every previously accepted fact must stay at or above this
    max_loops_per_fact: int = 200     # budget; reaching it means STOP


@dataclass
class FactRecord:
    position: int                     # 1-based position in the stream
    fid: int
    accepted: bool
    loops: int
    grad_steps: int
    train_samples: int
    probe_samples: int
    compute: int                      # forward-pass equivalents (train sample = 3, probe sample = 1)
    seconds: float
    # Last check of this fact's final loop (gate, probe samples)
    new_fact_probe_acc: float
    old_facts_failing: int            # accepted facts below tau_old at the last check
    min_old_probe_acc: float
    # Independent verification on test samples, over all accepted facts (including this one if accepted)
    test_mean: float = float("nan")
    test_min: float = float("nan")
    test_share_ge_80: float = float("nan")


@dataclass
class StrictResult:
    seed: int
    facts_presented: int
    facts_accepted: int
    stopped: bool
    stop_reason: str = ""
    records: list = field(default_factory=list)
    seconds: float = 0.0


def run_strict(world: FactWorld, tc: TrainConfig, sc: StrictConfig, seed: int, log=None) -> StrictResult:
    t0 = time.time()
    rng = random.Random(seed)
    learner = Learner(world, tc.hidden, tc.lr, seed)
    memory = ReplayMemory(world, tc.mem_per_item, rng)
    accepted: list[int] = []
    res = StrictResult(seed, 0, 0, False)

    for pos, fid in enumerate(world.order, start=1):
        res.facts_presented = pos
        s0, ts0, ps0, c0, tt = (learner.grad_steps, learner.train_samples, learner.probe_samples,
                                learner.compute, time.time())
        new = world.xy(fid, "train")
        failing: list[int] = []
        passed = False
        for loop in range(1, sc.max_loops_per_fact + 1):
            # Learn: the new fact + accepted facts that failed the last check + replay.
            repair = make_batch(*[world.xy(f, "train", tc.mem_per_item) for f in failing]) if failing else None
            for _ in range(sc.k):
                learner.step(*make_batch(new, repair, memory.sample(tc.replay_bs)))

            # Check: the new fact and EVERY previously accepted fact.
            new_acc = learner.probe(fid)[0]
            old_accs = {f: learner.probe(f)[0] for f in accepted}
            failing = [f for f, a in old_accs.items() if a < sc.tau_old]
            if new_acc >= sc.tau_new and not failing:
                passed = True
                break

        rec = FactRecord(
            position=pos, fid=fid, accepted=passed, loops=loop,
            grad_steps=learner.grad_steps - s0, train_samples=learner.train_samples - ts0,
            probe_samples=learner.probe_samples - ps0, compute=learner.compute - c0,
            seconds=time.time() - tt, new_fact_probe_acc=new_acc, old_facts_failing=len(failing),
            min_old_probe_acc=min(old_accs.values()) if old_accs else float("nan"),
        )
        if passed:
            accepted.append(fid)
            memory.add(fid)
        verify = learner.test_acc(accepted)
        if verify:
            v = list(verify.values())
            rec.test_mean, rec.test_min = sum(v) / len(v), min(v)
            rec.test_share_ge_80 = sum(a >= 0.8 for a in v) / len(v)
        res.records.append(rec)
        res.facts_accepted = len(accepted)

        if log and (pos % 10 == 0 or not passed):
            log(f"seed {seed} fact {pos}: {'accepted' if passed else 'NOT accepted'} after {loop} loops; "
                f"{len(accepted)} accepted so far")

        if not passed:
            parts = []
            if new_acc < sc.tau_new:
                parts.append(f"new fact at {new_acc:.0%} (needs {sc.tau_new:.0%})")
            if failing:
                parts.append(f"{len(failing)} accepted fact(s) below {sc.tau_old:.0%}")
            res.stopped = True
            res.stop_reason = (f"budget of {sc.max_loops_per_fact} loops reached at fact {pos}: "
                               + " and ".join(parts))
            break

    res.seconds = time.time() - t0
    return res
