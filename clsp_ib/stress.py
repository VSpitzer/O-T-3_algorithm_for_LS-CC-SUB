"""Randomised stress test: T3 vs T4 vs the exact reference.

    python -m clsp_ib.stress --max-T 14 --per-T 12
"""
from __future__ import annotations

import argparse
import random
import sys
import time

from .generator import random_instance
from .instance import Instance, InfeasibleInstance, StorageCapacityError
from .reference import exact_optimum, plan_cost
from .reformulate import check_storage_capacity, to_equivalent_model
from .t3 import solve_T3
from .t4 import solve_T4


def spiky_instance(T: int, C: int, seed: int) -> Instance:
    """Demand may exceed C, so Section 2.2.2 has real work to do."""
    rng = random.Random(seed)
    d = [rng.choice([0, rng.randint(0, C), rng.randint(C + 1, 2 * C)]) for _ in range(T)]
    S = [rng.randint(4 * C, 7 * C) for _ in range(T)]
    s = [rng.randint(0, 30) for _ in range(T)]
    c = [rng.randint(1, 8) for _ in range(T)]
    h = [rng.randint(0, 2) for _ in range(T)]
    return Instance(T=T, C=C, d=d, S=S, s=s, c=c, h=h, I0=3 * C)


def run(max_T: int, per_T: int, C: int, verbose: bool) -> int:
    checked = mismatch = rejected = infeasible = capped = 0
    t_start = time.time()
    for T in range(1, max_T + 1):
        for trial in range(per_T):
            for kind, inst in (
                ("plain", random_instance(T, C=C, seed=7919 * T + trial)),
                ("spiky", spiky_instance(T, C, seed=104729 * T + trial)),
            ):
                try:
                    eq = to_equivalent_model(inst)
                except InfeasibleInstance:
                    infeasible += 1
                    continue
                try:
                    check_storage_capacity(eq)
                except StorageCapacityError:
                    rejected += 1
                    continue
                if any(a != b for a, b in zip(inst.d, eq.d)):
                    capped += 1
                s3 = solve_T3(eq, initialisation="incremental")
                s3e = solve_T3(eq, initialisation="exact")
                s4 = solve_T4(eq)
                s3.check_feasible(eq)
                s3e.check_feasible(eq)
                s4.check_feasible(eq)
                ref = exact_optimum(inst)
                checked += 1
                scale = max(1.0, abs(ref))
                bad = []
                if abs(s4.cost - ref) > 1e-7 * scale:
                    bad.append(f"T4 {s4.cost} vs exact {ref}")
                if abs(s3.cost - ref) > 1e-7 * scale:
                    bad.append(f"T3(incremental) {s3.cost} vs exact {ref}")
                if abs(s3e.cost - ref) > 1e-7 * scale:
                    bad.append(f"T3(exact init) {s3e.cost} vs exact {ref}")
                if abs(plan_cost(inst, s3.x) - s3.cost) > 1e-7 * scale:
                    bad.append(f"T3 plan cost {plan_cost(inst, s3.x)} vs reported {s3.cost}")
                if bad:
                    mismatch += 1
                    print(f"  MISMATCH T={T} {kind} trial={trial}: " + "; ".join(bad))
                elif verbose:
                    print(f"  ok T={T} {kind} trial={trial}: {ref}")
        print(f"T={T:3d} cumulative: {checked} checked, {mismatch} mismatched, "
              f"{rejected} out of scope, {infeasible} infeasible")
    print(f"\n{checked} instances compared in {time.time() - t_start:.1f}s: "
          f"{mismatch} mismatches")
    print(f"({capped} of them needed demand capping, {rejected} rejected by the 2C check, "
          f"{infeasible} were infeasible)")
    return 1 if mismatch else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="clsp_ib.stress", description=__doc__)
    ap.add_argument("--max-T", type=int, default=12)
    ap.add_argument("--per-T", type=int, default=10)
    ap.add_argument("--C", type=int, default=6)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    return run(a.max_T, a.per_T, a.C, a.verbose)


if __name__ == "__main__":
    sys.exit(main())
