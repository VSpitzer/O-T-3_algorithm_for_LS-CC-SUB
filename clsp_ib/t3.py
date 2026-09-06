"""The O(T^3) algorithm of the article: Algorithms 1 to 5 of Section 6.

Naming follows Section 6 as closely as Python allows:

    slack(t) = S_t - I_t    inventory headroom; the package uses this one
                            quantity everywhere, never a shifted variant
    u, s, delta, gamma      the four critical periods of Section 6.1
    q, eta                  the two additional parameters of Section 6.2
    M, gamma_M              the profitability flag of Algorithm 2 / 5
    Cx                      the running cost C^x of the current P' solution
    cf(t)                   s_t + C * c_t

Deviations from the printed algorithms, all deliberate and all commented at the
point of use:

1. Two initialisations are available, selected by ``solve_T3(eq,
   initialisation=...)``.  ``"incremental"`` (the default) is the Section 4
   procedure ported in ``t3_init.py``; ``"exact"`` solves P'(t'_max) with the
   shared T2 solver for every subplan, which is slower but independent of
   Section 4.  The incremental procedure does not cover every subplan -- its
   descending walk stops when the fractional period would fall before t1 -- and
   the exact solver is used for the remainder, so no subplan arc is lost.
   The bounds t_min, t_max and t'_min are still obtained by the exact O(T^2)
   scan of ``feasible_fractional_periods`` in both modes.
2. Algorithm 5 registers a candidate for P(t0) only inside the while loop, i.e.
   from t'_max - 1 downwards, yet line 24 minimises over t0 in [t_min, t_max]
   and t_max may equal t'_max.  A candidate is therefore also registered for
   t0 = t'_max, before the loop.
3. Line 13 updates the inventory over [gamma_t0, t0].  That is the correct range
   whenever delta > t0, and the entries above t0 are never read again, which is
   what makes the truncation sound.  When delta = t0 (permitted by line 15) the
   range [gamma_t0, t0] would decrement I_{t0} even though the batch is added
   there, so the exact range [gamma_t0, delta - 1] capped at t0 is used instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .dag import Solution, run_subplan_dag
from .instance import EquivalentInstance
from .subplan import (
    INF,
    TOL,
    Subplan,
    SubplanSolution,
    feasible_fractional_periods,
    solve_subplan,
)


class _DynamicSubplan:
    """One run of Algorithm 5 over a single subplan."""

    def __init__(self, eq: EquivalentInstance, sp: Subplan, K: int, f: float,
                 t0: int, full: Set[int], tol: float = TOL) -> None:
        self.eq, self.sp, self.K, self.f, self.tol = eq, sp, K, f, tol
        self.t0 = t0
        self.full: Set[int] = set(full)
        self.full_initial: Set[int] = set(full)
        self.moves: List[Optional[Tuple[int, int]]] = []
        self.Cx = sum(eq.cf(t) for t in self.full)          # Algorithm 5, line 3

        # inventory of the current P' solution
        self.I: Dict[int, float] = {}
        inv = sp.I_in
        for t in range(sp.t1, sp.t2 + 1):
            if t in self.full:
                inv += eq.C
            if t == t0:
                inv += f
            inv -= eq.d[t - 1]
            self.I[t] = inv

        self.M = True                                        # Algorithm 5, line 2
        self.gamma_M = -1
        self.u = sp.t2
        self.s = sp.t1
        self.delta: Optional[int] = None
        self.gamma: Dict[int, Optional[int]] = {}
        self.q = sp.t1
        self.eta: Dict[int, Optional[int]] = {}
        self._initialise_parameters()                        # Algorithm 5, line 4

    # ------------------------------------------------------------------ helpers

    def _initialise_parameters(self) -> None:
        eq, sp, t0 = self.eq, self.sp, self.t0
        # u^{t0} = min({t in [t0,t2] : I_t < C} u {t2})
        self.u = sp.t2
        for t in range(t0, sp.t2 + 1):
            if self.I[t] < eq.C - self.tol:
                self.u = t
                break
        # delta^{t0} = cheapest available period in [t0+1, u]
        self.delta = None
        for t in range(t0 + 1, self.u + 1):
            if t not in self.full and (self.delta is None or eq.cf(t) < eq.cf(self.delta)):
                self.delta = t
        self._recompute_s_gamma()
        self._recompute_q_eta()

    def _recompute_s_gamma(self) -> None:
        """s^{t0} and gamma_r^{t0} for r in [s, t0] (Algorithm 1, lines 7-8)."""
        eq, sp, t0 = self.eq, self.sp, self.t0
        s = sp.t1
        for t in range(t0, sp.t1, -1):          # latest t in [t1+1,t0] with I_{t-1} < C
            if self.I[t - 1] < eq.C - self.tol:
                s = t
                break
        self.s = s
        self.gamma = {}
        best: Optional[int] = None
        for r in range(s, t0 + 1):
            if r in self.full and (best is None or eq.cf(r) > eq.cf(best)):
                best = r
            self.gamma[r] = best

    def _slack(self, t: int) -> float:
        """S_t - I_t, the room left under the inventory bound at the end of t.

        The article's condition ``I_{t-1} > S_{t-1} - C`` in the definition of
        ``q`` is ``slack(t-1) < C``.  Note that ``init_solve_T3`` in the C++ uses
        the name ``J`` for this quantity shifted by ``-C``; nothing in this
        package carries such a shift.
        """
        return self.eq.bound(t) - self.I[t]

    def _recompute_q_eta(self) -> None:
        """q^{t0} and then eta (Algorithm 3, lines 2-3)."""
        eq, sp, t0 = self.eq, self.sp, self.t0
        q = sp.t1
        for t in range(t0, sp.t1, -1):          # latest t with slack(t-1) < C
            if self._slack(t - 1) < eq.C - self.tol:
                q = t
                break
        self.q = q
        self._fill_eta()

    def _fill_eta(self) -> None:
        """eta_{q,v} = cheapest available period in [q, v], for v in [q, t0]."""
        eq = self.eq
        self.eta = {}
        best: Optional[int] = None
        for v in range(self.q, self.t0 + 1):
            if v not in self.full and (best is None or eq.cf(v) < eq.cf(best)):
                best = v
            self.eta[v] = best

    # --------------------------------------------------------------- algorithms

    def algorithm_1(self) -> None:
        """Parameter update after the fractional period shifts from t0+1 to t0."""
        eq, sp, t0 = self.eq, self.sp, self.t0
        self.I[t0] += self.f                                        # line 1
        if self.I[t0] < eq.C - self.tol:                            # line 2
            self.u = t0
            self.delta = None                                       # line 3
        else:                                                       # line 4
            nxt = t0 + 1
            if nxt <= sp.t2 and nxt not in self.full:
                if self.delta is None or eq.cf(nxt) < eq.cf(self.delta):
                    self.delta = nxt                                # line 5
        if t0 == self.s - 1:                                        # line 6
            self._recompute_s_gamma()                               # lines 7-8

    def algorithm_3(self) -> None:
        """(q, eta) update after the fractional period shifts."""
        if self.q == self.t0 + 1:                                   # line 1
            self._recompute_q_eta()                                 # lines 2-3

    def algorithm_4(self, gamma_t0: int) -> None:
        """(q, eta) update after a full production reallocation."""
        if gamma_t0 < self.q:                                       # line 1
            self.q = gamma_t0
        self._fill_eta()                                            # lines 2-3

    # -------------------------------------------------------------------- driver

    def reallocate(self, gamma_t0: int, target: int) -> None:
        """Algorithm 5, lines 12-16."""
        eq, t0 = self.eq, self.t0
        self.full.discard(gamma_t0)                                 # line 12
        self.full.add(target)
        # line 13, exact range (see deviation 3 in the module docstring)
        for t in range(gamma_t0, min(target, t0 + 1)):
            self.I[t] -= eq.C
        self.Cx += eq.cf(target) - eq.cf(gamma_t0)                  # line 14
        self.u = t0                                                 # line 15
        self.delta = t0 if t0 not in self.full else None
        self.algorithm_4(gamma_t0)                                  # line 16

    def candidate_for_P(self) -> Optional[Tuple[float, Optional[int]]]:
        """Algorithm 5, lines 18-23: cost of the best solution to P(t0).

        Returns ``(cost, zeta)`` where ``zeta`` is the period receiving the full
        production displaced from ``t0``, or ``None`` if P(t0) has no solution.
        """
        eq, t0 = self.eq, self.t0
        cost = self.Cx + eq.frac_cost(t0, self.f)                   # line 19
        if t0 not in self.full:
            return cost, None
        # line 21: zeta = argmin over {eta_{q,t0-1}, delta} of cf
        candidates: List[int] = []
        before = self.eta.get(t0 - 1) if t0 - 1 >= self.q else None
        if before is not None:
            candidates.append(before)
        if self.delta is not None:
            candidates.append(self.delta)
        if not candidates:
            return None
        zeta = min(candidates, key=eq.cf)
        cost += eq.cf(zeta) - eq.cf(t0)                             # line 23
        return cost, zeta

    def run(self, t_min: int, t_max: int, tp_min: int) -> Optional[SubplanSolution]:
        eq, sp, f = self.eq, self.sp, self.f
        best: Optional[Tuple[float, int, int, Optional[int]]] = None  # cost, t0, n_moves, zeta

        def register() -> None:
            nonlocal best
            if not (t_min <= self.t0 <= t_max):
                return
            got = self.candidate_for_P()
            if got is None:
                return
            cost, zeta = got
            if best is None or cost < best[0] - self.tol:
                best = (cost, self.t0, len(self.moves), zeta)

        register()                                    # deviation 2: t0 = t'_max
        while self.t0 > t_min:                        # line 5
            self.t0 -= 1                              # line 6
            self.algorithm_1()                        # line 7
            self.algorithm_3()                        # line 8
            if not self.M and self.t0 < self.gamma_M:  # line 9
                self.M = True
            g = self.gamma.get(self.t0)
            profitable = (
                g is not None and self.delta is not None
                and eq.cf(g) > eq.cf(self.delta) + self.tol
            )
            infeasible = self._slack(self.t0) < -self.tol    # I_{t0} > S_{t0}
            if (profitable and self.M) or infeasible:  # line 10
                if profitable:                         # line 11
                    self.M = False
                    self.gamma_M = g
                if g is None or self.delta is None:
                    # Section 3 rules this out for in-scope subplans; if it does
                    # happen the P' recursion cannot continue, so stop and keep
                    # the candidates gathered so far.
                    self.moves.append(None)
                    break
                target = self.delta
                self.moves.append((g, target))
                self.reallocate(g, target)
            else:
                self.moves.append(None)
            register()                                 # lines 17-23

        if best is None:
            return None
        cost, t0_star, n_moves, zeta = best
        full = set(self.full_initial)
        for mv in self.moves[:n_moves]:
            if mv is not None:
                g, target = mv
                full.discard(g)
                full.add(target)
        if zeta is not None:
            full.discard(t0_star)
            full.add(zeta)
        x = {t: eq.C for t in full}
        x[t0_star] = x.get(t0_star, 0.0) + f
        return SubplanSolution(cost=cost, x=x, full=full, frac_period=t0_star)


def solve_subplan_T3(eq: EquivalentInstance, sp: Subplan, K: int, f: float,
                     tol: float = TOL,
                     init: Optional[Tuple[frozenset, int]] = None
                     ) -> Optional[SubplanSolution]:
    """Optimal solution of one subplan by the dynamic procedure of Section 6.

    ``init`` optionally supplies the ``(full periods, t'_max)`` pair produced by
    the Section 4 procedure; when omitted, or when it is not usable, P'(t'_max)
    is solved exactly instead.
    """
    if K == 0 and f <= tol:
        return SubplanSolution(cost=0.0, x={}, full=set(), frac_period=None)

    if f <= tol:
        # "If the fractional production f is of zero value, then the optimal
        # solution to P'(t'_max) is also an optimal solution to the subplan."
        return solve_subplan(eq, sp, K, 0.0, frac_period=None, free_fractional=False, tol=tol)

    tp = feasible_fractional_periods(eq, sp, K, f, allow_full_at_frac=True, tol=tol)
    tt = feasible_fractional_periods(eq, sp, K, f, allow_full_at_frac=False, tol=tol)
    if not tp or not tt:
        return None
    tp_min, tp_max = tp[0], tp[-1]
    t_min, t_max = tt[0], tt[-1]

    # Section 4 initialisation: use the supplied seed when it is consistent with
    # the feasible range, otherwise solve P'(t'_max) exactly (see deviation 1).
    seed_full, seed_t0 = None, None
    if init is not None and init[1] in tp:
        seed_full, seed_t0 = set(init[0]), init[1]
        if len(seed_full) != K:
            seed_full, seed_t0 = None, None
    if seed_full is None:
        exact = solve_subplan(eq, sp, K, f, frac_period=tp_max,
                              free_fractional=True, tol=tol)
        if exact is None:
            return None
        seed_full, seed_t0 = set(exact.full), tp_max

    dyn = _DynamicSubplan(eq, sp, K, f, seed_t0, seed_full, tol=tol)
    return dyn.run(t_min=t_min, t_max=t_max, tp_min=tp_min)


def solve_T3(eq: EquivalentInstance, initialisation: str = "incremental") -> Solution:
    """Solve the whole horizon with the O(T^3) algorithm.

    ``initialisation="incremental"`` uses the Section 4 procedure of
    ``t3_init.py``; ``"exact"`` solves every P'(t'_max) with the T2 solver.
    """
    if initialisation == "exact":
        return run_subplan_dag(eq, solve_subplan_T3, "T3")
    if initialisation != "incremental":
        raise ValueError("initialisation must be 'incremental' or 'exact'")

    from .t3_init import build_initialisation_cache, lookup

    cache = build_initialisation_cache(eq, TOL)

    def solver(eq_: EquivalentInstance, sp: Subplan, K: int, f: float):
        return solve_subplan_T3(eq_, sp, K, f, tol=TOL, init=lookup(cache, sp))

    return run_subplan_dag(eq, solver, "T3")
