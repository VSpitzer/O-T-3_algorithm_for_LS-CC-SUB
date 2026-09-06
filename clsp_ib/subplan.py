"""Subplans (Definition 2) and the T2 per-subplan solver shared by T3 and T4.

Following Definition 2 a subplan is a tuple ``(t1, t2, I^1, I^2)``; here it is
stored with the entry and exit inventory levels already resolved, since for the
first subplan the entry level is the given ``I_0`` rather than ``I^1 * S_0``.

The total production over a subplan is fixed:

    d_{t1,t2} + I_out - I_in = K * C + f,     K in N,  f in [0, C)

so any feasible solution consists of exactly ``K`` full production periods plus,
when ``f > 0``, one fractional period producing ``f`` (Love, 1973).

Two problems are distinguished, as in Definitions 3 and 4:

``P(t0)``   the fractional production takes place in period ``t0``: that period
            produces exactly ``f`` and cannot also carry a full production.
``P'(t0)``  the ``f`` units become available in ``t0`` for free -- no cost and no
            capacity -- so ``t0`` remains available for a full production.

``solve_subplan`` covers both, and with ``frac_period="any"`` it minimises over
all admissible fractional periods, which is exactly the O(T^2) per-subplan
computation used by the T4 algorithm.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

from .instance import EquivalentInstance

INF = float("inf")
TOL = 1e-9


@dataclass(frozen=True)
class Subplan:
    t1: int
    t2: int
    I_in: float
    I_out: float

    @property
    def length(self) -> int:
        return self.t2 - self.t1 + 1

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return f"({self.t1},{self.t2},I_in={self.I_in:g},I_out={self.I_out:g})"


@dataclass
class SubplanSolution:
    """A production plan for one subplan."""

    cost: float
    x: Dict[int, float] = field(default_factory=dict)  # period -> quantity
    full: Set[int] = field(default_factory=set)        # periods with a full production
    frac_period: Optional[int] = None

    def quantity(self, t: int) -> float:
        return self.x.get(t, 0.0)


def totals(eq: EquivalentInstance, sp: Subplan, tol: float = TOL) -> Optional[Tuple[int, float]]:
    """Return ``(K, f)`` from equation (8), or ``None`` if total production < 0."""
    total = eq.demand(sp.t1, sp.t2) + sp.I_out - sp.I_in
    if total < -tol:
        return None
    K = int(math.floor(total / eq.C + tol))
    f = total - K * eq.C
    if f <= tol:
        f = 0.0
    elif eq.C - f <= tol:  # numerically a full batch
        K += 1
        f = 0.0
    return K, f


def _cum_demand(eq: EquivalentInstance, t1: int, t2: int) -> Dict[int, float]:
    acc = 0.0
    out: Dict[int, float] = {}
    for t in range(t1, t2 + 1):
        acc += eq.d[t - 1]
        out[t] = acc
    return out


def solve_subplan(
    eq: EquivalentInstance,
    sp: Subplan,
    K: int,
    f: float,
    frac_period: Union[int, str, None] = "any",
    free_fractional: bool = False,
    tol: float = TOL,
) -> Optional[SubplanSolution]:
    """Exact solver for one subplan by dynamic programming.

    The state after period ``t`` is ``(k, phi)`` -- the number of full
    production periods used so far and whether the fractional quantity has been
    placed -- because the inventory level is then determined:

        I_t = I_in + k*C + phi*f - d_{t1,t}

    which is the structural property that makes the subplan decomposition
    useful.  Complexity is O((t2-t1+1) * K), i.e. O(T^2) per subplan.

    ``frac_period``: ``"any"`` to optimise over all admissible periods, an
    ``int`` to pin the fractional period, ``None`` when ``f == 0``.
    ``free_fractional``: solve ``P'`` instead of ``P``.
    """
    t1, t2 = sp.t1, sp.t2
    need_frac = f > tol
    if not need_frac:
        frac_period = None
    dcum = _cum_demand(eq, t1, t2)

    # states[i] maps (k, phi) -> (cost, previous_state, quantity_produced_at_t)
    states: List[Dict[Tuple[int, int], Tuple[float, Tuple[int, int], float]]] = []
    cur: Dict[Tuple[int, int], Tuple[float, Tuple[int, int], float]] = {(0, 0): (0.0, (0, 0), 0.0)}

    for t in range(t1, t2 + 1):
        nxt: Dict[Tuple[int, int], Tuple[float, Tuple[int, int], float]] = {}
        S_t = eq.bound(t)
        for (k, phi), (cost, _prev, _q) in cur.items():
            moves: List[Tuple[int, int, float, float]] = [(k, phi, 0.0, 0.0)]
            if k + 1 <= K:
                moves.append((k + 1, phi, eq.cf(t), eq.C))
            if need_frac and phi == 0 and (frac_period == "any" or t == frac_period):
                if free_fractional:
                    # the f units arrive for free; t stays available for a full batch
                    moves.append((k, 1, 0.0, 0.0))
                    if k + 1 <= K:
                        moves.append((k + 1, 1, eq.cf(t), eq.C))
                else:
                    moves.append((k, 1, eq.frac_cost(t, f), f))
            for k2, phi2, add, qty in moves:
                inv = sp.I_in + k2 * eq.C + phi2 * f - dcum[t]
                if inv < -tol or inv > S_t + tol:
                    continue
                key = (k2, phi2)
                val = cost + add
                old = nxt.get(key)
                if old is None or val < old[0] - tol:
                    nxt[key] = (val, (k, phi), qty)
        cur = nxt
        states.append(cur)
        if not cur:
            return None

    goal = (K, 1 if need_frac else 0)
    if goal not in cur:
        return None

    # walk the backpointers
    x: Dict[int, float] = {}
    full: Set[int] = set()
    frac_at: Optional[int] = None
    key = goal
    for i in range(len(states) - 1, -1, -1):
        t = t1 + i
        cost_i, prev, qty = states[i][key]
        if qty > tol:
            x[t] = x.get(t, 0.0) + qty
            if abs(qty - eq.C) <= tol:
                full.add(t)
        if need_frac and key[1] == 1 and prev[1] == 0:
            frac_at = t
            if free_fractional:
                x.setdefault(t, 0.0)  # f is free: it produces nothing
            else:
                pass  # qty == f was already recorded above
        key = prev

    total_cost = states[-1][goal][0]
    return SubplanSolution(cost=total_cost, x=x, full=full, frac_period=frac_at)


def _ceil_units(value: float, C: float, tol: float) -> int:
    return int(math.ceil((value - tol) / C))


def _floor_units(value: float, C: float, tol: float) -> int:
    return int(math.floor((value + tol) / C))


def feasible_fractional_periods(
    eq: EquivalentInstance,
    sp: Subplan,
    K: int,
    f: float,
    allow_full_at_frac: bool,
    tol: float = TOL,
) -> List[int]:
    """Periods ``t0`` for which ``P(t0)`` (or ``P'(t0)``) admits a feasible plan.

    For a fixed ``t0`` the inventory level after period ``t`` is determined by
    the number ``k_t`` of full productions used so far, and ``k_t`` moves by 0
    or 1 per period, so the set of reachable ``k_t`` is an interval.  Feasibility
    is therefore decided by propagating that interval forward in O(t2-t1+1).

    Section 3 of the article proves this set is contiguous and non-empty for
    subplans whose potential for storage is at least twice the capacity; the
    scan below does not rely on that, it simply reports what is feasible.
    """
    t1, t2 = sp.t1, sp.t2
    dcum = _cum_demand(eq, t1, t2)
    need_frac = f > tol
    candidates = range(t1, t2 + 1) if need_frac else [t1]
    out: List[int] = []

    for t0 in candidates:
        lo, hi = 0, 0
        ok = True
        for t in range(t1, t2 + 1):
            step_max = 1
            if need_frac and t == t0 and not allow_full_at_frac:
                step_max = 0  # in P(t0) period t0 produces exactly f
            lo_n, hi_n = lo, min(hi + step_max, K)
            phi = 1.0 if (need_frac and t >= t0) else 0.0
            base = sp.I_in + phi * f - dcum[t]
            lo_n = max(lo_n, _ceil_units(-base, eq.C, tol), 0)
            hi_n = min(hi_n, _floor_units(eq.bound(t) - base, eq.C, tol), K)
            if lo_n > hi_n:
                ok = False
                break
            lo, hi = lo_n, hi_n
        if ok and lo <= K <= hi:
            out.append(t0)
    return out if need_frac else (list(range(t1, t2 + 1)) if out else [])
