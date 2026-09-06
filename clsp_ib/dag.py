"""Combining subplan solutions: the shortest-path phase shared by T3 and T4.

Love (1973) shows that an optimal solution is a concatenation of optimal
subplan solutions whose entry and exit inventory levels are at a bound, so the
overall problem is a shortest path in an acyclic graph whose arcs are subplans
(Section 2.3, and Phase 2 of van Hoesel and Wagelmans).

Node ``(t, e)`` means "period t closes a subplan with ending inventory
``e * S_t``"; node ``(0, 0)`` is the source and carries the given ``I_0``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .instance import EquivalentInstance
from .subplan import INF, TOL, Subplan, SubplanSolution, totals

Node = Tuple[int, int]

# a solver takes (eq, subplan, K, f) and returns a SubplanSolution or None
SubplanSolver = Callable[[EquivalentInstance, Subplan, int, float], Optional[SubplanSolution]]


@dataclass
class Solution:
    """A solution to the whole horizon."""

    algorithm: str
    cost: float                       # objective of the ORIGINAL model (1a)
    equivalent_cost: float            # objective of the equivalent model
    x: List[float] = field(default_factory=list)
    I: List[float] = field(default_factory=list)
    subplans: List[Subplan] = field(default_factory=list)
    n_subplans_solved: int = 0

    def check_feasible(self, eq: EquivalentInstance, tol: float = 1e-6) -> None:
        inv = eq.I0
        for t in range(1, eq.T + 1):
            q = self.x[t - 1]
            if q < -tol or q > eq.C + tol:
                raise AssertionError(f"x_{t} = {q} outside [0, C]")
            inv += q - eq.d[t - 1]
            if inv < -tol or inv > eq.bound(t) + tol:
                raise AssertionError(f"I_{t} = {inv} outside [0, S_{t}={eq.bound(t)}]")


def entry_inventory(eq: EquivalentInstance, t1: int, e_in: int) -> float:
    """Inventory entering a subplan that starts in period ``t1``.

    The first subplan enters with the given ``I_0``, as in inequality (4);
    later subplans enter at one of the two bounds of Definition 2.
    """
    if t1 == 1:
        return eq.I0
    return e_in * eq.bound(t1 - 1)


def exit_levels(eq: EquivalentInstance, t2: int, tol: float = TOL):
    """Admissible ending inventory levels for a subplan closing in ``t2``.

    Definition 2 restricts these to ``{0, S_{t2}}``.  One extra level is needed
    at the end of the horizon: when the initial inventory exceeds the total
    demand, the cheapest plan produces nothing and leaves
    ``I_T = I_0 - d_{1,T} > 0``, which is interior.  Love's decomposition is
    stated for instances that start empty, so that residual level has to be
    offered explicitly, otherwise no concatenation of subplans can represent the
    optimum.  It coincides with 0 whenever ``d_{1,T} >= I_0``, i.e. always
    except in that degenerate case.
    """
    levels = [(0, 0.0), (1, eq.bound(t2))]
    if t2 == eq.T:
        residual = max(0.0, eq.I0 - eq.demand(1, eq.T))
        if all(abs(residual - lv) > tol for _, lv in levels):
            levels.append((2, residual))
    return levels


def run_subplan_dag(eq: EquivalentInstance, solver: SubplanSolver, name: str,
                    tol: float = TOL) -> Solution:
    """Solve every subplan with ``solver`` and return the best concatenation."""
    T = eq.T
    dist: Dict[Node, float] = {(0, 0): 0.0}
    prev: Dict[Node, Tuple[Node, Subplan, SubplanSolution]] = {}
    solved = 0

    for p in range(0, T):
        e_range = (0,) if p == 0 else (0, 1)
        for e_in in e_range:
            here = dist.get((p, e_in))
            if here is None or here == INF:
                continue
            t1 = p + 1
            I_in = entry_inventory(eq, t1, e_in)
            for t2 in range(t1, T + 1):
                for e_out, I_out in exit_levels(eq, t2, tol):
                    sp = Subplan(t1, t2, I_in, I_out)
                    kf = totals(eq, sp, tol)
                    if kf is None:
                        continue
                    K, f = kf
                    sol = solver(eq, sp, K, f)
                    solved += 1
                    if sol is None:
                        continue
                    cand = here + sol.cost
                    node = (t2, e_out)
                    if cand < dist.get(node, INF) - tol:
                        dist[node] = cand
                        prev[node] = ((p, e_in), sp, sol)

    best_node, best = None, INF
    for e, _lv in exit_levels(eq, T, tol):
        v = dist.get((T, e), INF)
        if v < best - tol:
            best, best_node = v, (T, e)
    if best_node is None:
        raise AssertionError("no feasible concatenation of subplans was found")

    x = [0.0] * T
    chain: List[Subplan] = []
    node = best_node
    while node != (0, 0):
        pnode, sp, sol = prev[node]
        for t, q in sol.x.items():
            x[t - 1] += q
        chain.append(sp)
        node = pnode
    chain.reverse()

    inv: List[float] = []
    cur = eq.I0
    for t in range(1, T + 1):
        cur += x[t - 1] - eq.d[t - 1]
        inv.append(cur)

    return Solution(
        algorithm=name,
        cost=best + eq.cost_offset,
        equivalent_cost=best,
        x=x,
        I=inv,
        subplans=chain,
        n_subplans_solved=solved,
    )
