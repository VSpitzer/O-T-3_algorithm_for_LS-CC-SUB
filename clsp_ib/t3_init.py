"""Section 4: the dynamic initialisation of the P'(t'_max) solutions.

For a fixed first period ``t1`` and a fixed pair of inventory parameters, the
optimal solution to ``P'(t'_max)`` is needed for every ending period ``t2``.
Rather than solving each one, the largest subplan ``(t1, T)`` is solved once
with the O(T^2) routine and ``t2`` is then decreased one period at a time, each
step costing O(T).  Over all ``t1`` that is O(T^3).

This is a direct port of ``init_solve_T3`` in ``solve_CLSP_IB.cpp``, translated
from the C++ 0-indexed local indices to the article's 1-indexed absolute
periods.  Two representations are simplified without changing behaviour:

* the C++ maintains ``J`` through the auxiliary demand ``d_p`` and the backward
  recursion of its lines 379-383.  That recursion is algebraically equivalent to
  ``J[t] = S_{t-1} - I_{t-1}``, which is what ``dyna_solve_T3`` itself uses, so
  ``J`` is derived from ``I`` here.  The C++ never reads ``J`` at the first
  index, which is the only place the two could differ.
* the C++ interleaves the backward walk of ``t0`` with incremental updates of
  ``I`` and ``J``.  Because ``I[t] = I_in + C*|full and [t1,t]| + f*[t >= t0] -
  d_{t1,t}``, walking ``t0`` down and adding ``f`` at each new ``t0`` gives
  exactly the array obtained by recomputing it at the final ``t0``, so it is
  recomputed once the walk has finished.

Correctness of this procedure is not established in the material available to
this package, so ``validate_initialisation`` checks it against the exact
per-subplan solver.

No ``J`` array appears here.  The C++ uses the name ``J`` for two different
quantities -- ``S_{t-1} - I_{t-1}`` in ``dyna_solve_T3``, and that same quantity
shifted by ``-C`` in ``init_solve_T3``, because its recursion is anchored on the
post-insertion end inventory while ``I`` is one batch short.  Reading the second
as the first is what makes the C++ test ``J[t_J+1] >= 0`` look wrong when it is
not.  To remove the ambiguity this module works with one explicit notion,

    slack(t) = S_t - I_t

and states each requirement over the interval on which it actually has to hold:

* inserting a full batch in ``p`` raises the inventory by ``C`` over ``[p, t2]``,
  so ``p`` is admissible iff ``min slack(t) over [p, t2] >= C``;
* removing a full batch from ``p`` lowers it by ``C`` over the same interval, so
  ``p`` is admissible iff ``min I_t over [p, t2] >= C``.

Both are evaluated from suffix minima, in O(t2 - t1).  The C++ instead uses each
condition as the terminating test of a backward scan started at ``t0``, which
leaves the periods in ``(t0, t2]`` unchecked; the suffix minima close that gap.
When no admissible period exists the walk stops and the exact solver covers the
remaining subplans, so nothing is lost.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

from .instance import EquivalentInstance
from .subplan import TOL, Subplan, solve_subplan, totals

InitEntry = Tuple[FrozenSet[int], int]  # (full production periods, fractional period t0)


def _round(v: float) -> float:
    return round(v, 9)


def latest_fractional_period(eq: EquivalentInstance, t1: int, t2: int,
                             I_out: float, f: float, tol: float = TOL) -> int:
    """The C++ choice of t'_max: the latest p with ``I_out + d_{p,t2} >= f``.

    If the fractional quantity is produced in ``p`` then, looking at the tail of
    the subplan, ``I_{p-1} = I_out + d_{p,t2} - f - C * (full batches after p)``,
    so ``I_out + d_{p,t2} >= f`` is necessary for feasibility.  Note that it is
    only necessary: it ignores the inventory upper bounds, so this period can be
    later than the true latest feasible fractional period.
    """
    best = t1
    for p in range(t1 + 1, t2 + 1):
        if I_out + eq.demand(p, t2) >= f - tol:
            best = p
    return best



def _cheapest(eq, full, t1: int, t0: int, min_slack, need: float, tol: float,
              available: bool) -> Optional[int]:
    """Cheapest period in [t1, t0] that can take a full batch.

    ``min_slack[p]`` is ``min S_t - I_t`` over ``[p, t2]``; a batch inserted in
    ``p`` raises the inventory by ``C`` over exactly that interval.
    """
    best: Optional[int] = None
    for p in range(t0, t1 - 1, -1):
        if (p in full) == available:
            continue
        if min_slack[p] < need - tol:
            continue
        if best is None or eq.cf(p) < eq.cf(best):
            best = p
    return best


def _most_expensive(eq, full, t1: int, t0: int, min_inventory, need: float,
                    tol: float) -> Optional[int]:
    """Most expensive full production period in [t1, t0] that can be removed.

    ``min_inventory[p]`` is ``min I_t`` over ``[p, t2]``; removing a batch from
    ``p`` lowers the inventory by ``C`` over exactly that interval.
    """
    best: Optional[int] = None
    for p in range(t0, t1 - 1, -1):
        if p not in full:
            continue
        if min_inventory[p] < need - tol:
            continue
        if best is None or eq.cf(p) > eq.cf(best):
            best = p
    return best

@dataclass
class InitDiagnostics:
    stopped_at: Optional[int] = None      # t2 at which the walk gave up, if any
    reason: str = ""
    exact_fallbacks: int = 0


def incremental_initialisation(
    eq: EquivalentInstance,
    t1: int,
    I_in: float,
    e_out: int,
    tol: float = TOL,
) -> Tuple[Dict[int, InitEntry], InitDiagnostics]:
    """Initial P' solutions for every subplan ``(t1, t2)`` with this parameter pair."""
    T = eq.T
    table: Dict[int, InitEntry] = {}
    diag = InitDiagnostics()

    # ---- the largest subplan, solved exactly (Wolsey's O(T^2) algorithm)
    t2 = T
    I_out = e_out * eq.bound(t2)
    kf = totals(eq, Subplan(t1, t2, I_in, I_out), tol)
    if kf is None:
        diag.reason = "total production negative for t2 = T"
        return table, diag
    K, f = kf
    if K + (1 if f > tol else 0) > t2 - t1 + 1:
        diag.reason = "not enough periods for t2 = T"
        return table, diag

    t0 = latest_fractional_period(eq, t1, t2, I_out, f, tol)
    seed = solve_subplan(eq, Subplan(t1, t2, I_in, I_out), K, f,
                         frac_period=t0, free_fractional=True, tol=tol)
    if seed is None:
        diag.reason = f"P'({t0}) infeasible for t2 = T"
        return table, diag
    full = set(seed.full)
    table[t2] = (frozenset(full), t0)

    # ---- decrease t2 one period at a time
    for t2 in range(T - 1, t1 - 1, -1):
        I_out = e_out * eq.bound(t2)
        kf = totals(eq, Subplan(t1, t2, I_in, I_out), tol)
        if kf is None:
            diag.stopped_at, diag.reason = t2, "total production negative"
            break
        K_new, f_new = kf

        # walk t0 backwards while it is no longer an admissible fractional period
        change, t_change = False, None
        while t0 > t2 or I_out + eq.demand(t0, t2) < f_new - tol:
            if t0 in full:
                change, t_change = True, t0
            t0 -= 1
            if t0 < t1:
                break
        if t0 < t1:
            diag.stopped_at, diag.reason = t2, "fractional period walked past t1"
            break

        # inventory of the current plan, and the two suffix minima it needs
        I: Dict[int, float] = {}
        inv = I_in
        for t in range(t1, t2 + 1):
            if t in full:
                inv += eq.C
            if t == t0 and f_new > tol:
                inv += f_new
            inv -= eq.d[t - 1]
            I[t] = inv
        min_slack: Dict[int, float] = {}
        min_inventory: Dict[int, float] = {}
        acc_slack = acc_inv = float("inf")
        for t in range(t2, t1 - 1, -1):
            acc_slack = min(acc_slack, eq.bound(t) - I[t])
            acc_inv = min(acc_inv, I[t])
            min_slack[t] = acc_slack
            min_inventory[t] = acc_inv

        if change and K_new == K:
            # the batch the fractional period ran into left the window and has to
            # be put back inside it: cheapest period with a full batch of slack
            best = _cheapest(eq, full, t1, t0, min_slack, eq.C, tol, available=True)
            if best is None:
                diag.stopped_at, diag.reason = t2, "no available period to relocate a batch"
                break
            full.add(best)
            full.discard(t_change)
        elif change and K_new < K:
            full.discard(t_change)
        elif (not change) and K_new < K:
            # one batch fewer is needed: drop the most expensive removable one
            best = _most_expensive(eq, full, t1, t0, min_inventory, eq.C, tol)
            if best is None:
                diag.stopped_at, diag.reason = t2, "no batch can be removed"
                break
            full.discard(best)

        table[t2] = (frozenset(full), t0)
        K, f = K_new, f_new

    return table, diag


def entry_levels(eq: EquivalentInstance, t1: int) -> List[float]:
    if t1 == 1:
        return [eq.I0]
    levels = [0.0, eq.bound(t1 - 1)]
    return [levels[0]] if abs(levels[0] - levels[1]) <= TOL else levels


def build_initialisation_cache(eq: EquivalentInstance, tol: float = TOL
                               ) -> Dict[Tuple[int, float, int, float], InitEntry]:
    """Run the Section 4 procedure for every (t1, entry level, exit parameter)."""
    cache: Dict[Tuple[int, float, int, float], InitEntry] = {}
    for t1 in range(1, eq.T + 1):
        for I_in in entry_levels(eq, t1):
            for e_out in (0, 1):
                table, _diag = incremental_initialisation(eq, t1, I_in, e_out, tol)
                for t2, entry in table.items():
                    key = (t1, _round(I_in), t2, _round(e_out * eq.bound(t2)))
                    cache.setdefault(key, entry)
    return cache


def lookup(cache, sp: Subplan) -> Optional[InitEntry]:
    return cache.get((sp.t1, _round(sp.I_in), sp.t2, _round(sp.I_out)))


@dataclass
class ValidationReport:
    subplans: int = 0
    covered: int = 0
    t0_differs: int = 0
    infeasible_plan: int = 0
    suboptimal: int = 0
    examples: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.infeasible_plan == 0 and self.suboptimal == 0

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"{self.covered}/{self.subplans} subplans covered by the Section 4 "
            f"procedure; t'_max differs from the exact value in {self.t0_differs}; "
            f"{self.infeasible_plan} infeasible plans; {self.suboptimal} suboptimal"
        )


def validate_initialisation(eq: EquivalentInstance, tol: float = TOL) -> ValidationReport:
    """Check the ported procedure against the exact per-subplan solver.

    For every subplan it verifies that the plan returned is (a) feasible for
    ``P'(t0)`` and (b) of the same cost as the exact optimum of ``P'(t0)``.  It
    also reports how often the C++ choice of ``t'_max`` differs from the latest
    genuinely feasible fractional period.
    """
    from .subplan import feasible_fractional_periods

    cache = build_initialisation_cache(eq, tol)
    rep = ValidationReport()
    for t1 in range(1, eq.T + 1):
        for I_in in entry_levels(eq, t1):
            for t2 in range(t1, eq.T + 1):
                for e_out in (0, 1):
                    I_out = e_out * eq.bound(t2)
                    sp = Subplan(t1, t2, I_in, I_out)
                    kf = totals(eq, sp, tol)
                    if kf is None:
                        continue
                    K, f = kf
                    rep.subplans += 1
                    entry = lookup(cache, sp)
                    if entry is None:
                        continue
                    rep.covered += 1
                    full, t0 = entry
                    exact = solve_subplan(eq, sp, K, f, frac_period=t0,
                                          free_fractional=True, tol=tol)
                    if exact is None:
                        rep.infeasible_plan += 1
                        if len(rep.examples) < 6:
                            rep.examples.append(f"{sp}: P'({t0}) is infeasible")
                        continue
                    # feasibility of the returned plan
                    inv = I_in
                    bad = None
                    for t in range(t1, t2 + 1):
                        if t in full:
                            inv += eq.C
                        if t == t0 and f > tol:
                            inv += f
                        inv -= eq.d[t - 1]
                        if inv < -1e-6 or inv > eq.bound(t) + 1e-6:
                            bad = t
                            break
                    if bad is not None or len(full) != K:
                        rep.infeasible_plan += 1
                        if len(rep.examples) < 6:
                            rep.examples.append(
                                f"{sp}: plan invalid (period {bad}, |full|={len(full)} vs K={K})")
                        continue
                    cost = sum(eq.cf(t) for t in full)
                    if cost > exact.cost + 1e-6 * max(1.0, abs(exact.cost)):
                        rep.suboptimal += 1
                        if len(rep.examples) < 6:
                            rep.examples.append(
                                f"{sp}: cost {cost} vs exact P'({t0}) {exact.cost}")
                    feas = feasible_fractional_periods(eq, sp, K, f,
                                                       allow_full_at_frac=True, tol=tol)
                    if feas and t0 != feas[-1]:
                        rep.t0_differs += 1
    return rep
