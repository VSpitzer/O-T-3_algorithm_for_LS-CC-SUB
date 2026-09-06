"""Compare the algorithms against a MILP solver on random instances.

    python -m clsp_ib.validate_milp --repeat 20 --T 12

For each instance it reformulates into the equivalent model, checks the
Psi >= 2C hypothesis, then reports the optimal cost obtained by

    T3    the O(T^3) algorithm
    T4    the O(T^4) algorithm
    MILP  model (1a)-(1f) handed to pulp/CBC or pyomo, at zero MIP gap
    DP    an exact integer-inventory dynamic program (integral instances only)

and fails if any two disagree.  If no MILP backend is installed the MILP column
is skipped and the DP still provides an independent exact answer.
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import List, Optional

from .generator import random_instance
from .instance import InfeasibleInstance, StorageCapacityError
from .milp import NoSolverAvailable, available_backends, solve_milp
from .reference import exact_optimum, plan_cost
from .reformulate import check_storage_capacity, to_equivalent_model
from .t3 import solve_T3
from .t4 import solve_T4


def _close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def run(repeat: int, T: int, C: int, seed: int, tol: float,
        use_milp: bool, verbose: bool) -> int:
    backends = available_backends()
    if use_milp and not backends:
        print("no MILP backend found (pip install pulp) -- comparing against the "
              "exact DP only\n")
        use_milp = False
    elif use_milp:
        print(f"MILP backends available: {', '.join(backends)}\n")

    header = f"{'#':>4} {'T':>4} {'T3':>14} {'T4':>14}"
    if use_milp:
        header += f" {'MILP':>14}"
    header += f" {'DP':>14}  verdict"
    print(header)
    print("-" * len(header))

    failures = rejected = 0
    t_start = time.time()
    for i in range(repeat):
        inst = random_instance(T, C=C, seed=seed + i)
        try:
            eq = to_equivalent_model(inst)
            check_storage_capacity(eq)
        except StorageCapacityError:
            print(f"{i:>4} {T:>4}   rejected: potential for storage below 2C")
            rejected += 1
            continue
        except InfeasibleInstance as exc:
            print(f"{i:>4} {T:>4}   infeasible: {exc}")
            rejected += 1
            continue

        s3, s4 = solve_T3(eq), solve_T4(eq)
        s3.check_feasible(eq)
        s4.check_feasible(eq)
        values = [("T3", s3.cost), ("T4", s4.cost)]

        if use_milp:
            try:
                milp = solve_milp(inst, gap=0.0)
                values.append(("MILP", milp.cost))
            except NoSolverAvailable as exc:
                print(f"  MILP unavailable: {exc}")
                use_milp = False

        try:
            values.append(("DP", exact_optimum(inst)))
        except ValueError:
            pass  # non-integral data, DP not applicable

        ref = values[0][1]
        bad = [name for name, v in values if not _close(v, ref, tol)]
        # the reported cost must also match the plan the algorithm returns
        if not _close(plan_cost(inst, s3.x), s3.cost, tol):
            bad.append("T3-plan")

        row = f"{i:>4} {T:>4}" + "".join(f" {v:>14.6g}" for _, v in values)
        if bad:
            failures += 1
            print(row + f"  MISMATCH ({', '.join(sorted(set(bad)))})")
        else:
            print(row + "  ok")
        if verbose:
            print(f"       plan: {[round(v, 3) for v in s3.x]}")

    print(f"\n{repeat - failures - rejected}/{repeat} instances matched "
          f"({rejected} rejected, {failures} mismatched) in {time.time() - t_start:.1f}s")
    return 1 if failures else 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="clsp_ib.validate_milp", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeat", type=int, default=10, help="number of instances")
    ap.add_argument("--T", type=int, default=12, help="planning horizon")
    ap.add_argument("--C", type=int, default=8, help="production capacity")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-6, help="relative tolerance")
    ap.add_argument("--no-milp", action="store_true", help="skip the MILP column")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    return run(a.repeat, a.T, a.C, a.seed, a.tol, not a.no_milp, a.verbose)


if __name__ == "__main__":
    sys.exit(main())
