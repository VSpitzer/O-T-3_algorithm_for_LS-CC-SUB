"""An exact reference solver, used only for testing.

For a fixed setup pattern the remaining problem is a network flow with integral
data, so its linear relaxation has an integral optimum.  Enumerating integer
production quantities and integer inventory levels therefore yields the true
optimum, with no optimality gap and no external solver.  Requires integer
``C``, ``d``, ``S`` and ``I0``; the cost coefficients may be arbitrary floats.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from .instance import Instance

INF = float("inf")


def exact_optimum(inst: Instance, return_plan: bool = False):
    """Exact optimal cost of model (1a)-(1f) by dynamic programming."""
    for name, seq in (("C", [inst.C]), ("d", inst.d), ("S", inst.S), ("I0", [inst.I0])):
        for v in seq:
            if abs(v - round(v)) > 1e-9:
                raise ValueError(f"exact_optimum requires integral {name}")
    T = inst.T
    C = int(round(inst.C))
    I0 = int(round(inst.I0))
    d = [int(round(v)) for v in inst.d]
    S = [int(round(v)) for v in inst.S]
    Smax = max(S) if S else 0

    if I0 > Smax:
        Smax = I0
    cur: List[float] = [INF] * (Smax + 1)
    cur[I0] = 0.0
    choice: List[List[int]] = []

    for t in range(1, T + 1):
        nxt = [INF] * (Smax + 1)
        pick = [-1] * (Smax + 1)
        st, ct, ht = inst.s[t - 1], inst.c[t - 1], inst.h[t - 1]
        for prev in range(Smax + 1):
            base = cur[prev]
            if base == INF:
                continue
            for x in range(0, C + 1):
                inv = prev + x - d[t - 1]
                if inv < 0 or inv > S[t - 1]:
                    continue
                val = base + (st if x > 0 else 0.0) + ct * x + ht * inv
                if val < nxt[inv]:
                    nxt[inv] = val
                    pick[inv] = x
        cur = nxt
        choice.append(pick)

    best = min(cur)
    if best == INF:
        raise AssertionError("reference solver found no feasible plan")
    if not return_plan:
        return best

    inv = min(range(Smax + 1), key=lambda i: cur[i])
    plan = [0] * T
    for t in range(T, 0, -1):
        x = choice[t - 1][inv]
        plan[t - 1] = x
        inv = inv - x + d[t - 1]
    return best, plan


def plan_cost(inst: Instance, x: List[float]) -> float:
    """Objective (1a) of an explicit plan, for cross-checking."""
    inv = inst.I0
    total = 0.0
    for t in range(1, inst.T + 1):
        q = x[t - 1]
        inv += q - inst.d[t - 1]
        total += (inst.s[t - 1] if q > 1e-12 else 0.0) + inst.c[t - 1] * q + inst.h[t - 1] * inv
    return total
