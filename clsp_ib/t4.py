"""The state-of-the-art O(T^4) algorithm (Wolsey, 2005).

Every subplan is solved independently and optimally, then the cheapest
concatenation is found by a shortest path.  Each subplan is solved by the same
T2 routine used by the T3 algorithm -- with ``frac_period="any"`` it minimises
over all admissible fractional periods in a single O(T^2) dynamic program --
so the two algorithms differ only in how the per-subplan optimum is obtained.

There are O(T^2) subplans and each costs O(T^2), hence O(T^4).
"""
from __future__ import annotations

from typing import Optional

from .dag import Solution, run_subplan_dag
from .instance import EquivalentInstance
from .subplan import Subplan, SubplanSolution, solve_subplan


def solve_subplan_T4(eq: EquivalentInstance, sp: Subplan, K: int, f: float
                     ) -> Optional[SubplanSolution]:
    """Optimal solution of one subplan, minimising over all fractional periods."""
    return solve_subplan(eq, sp, K, f, frac_period="any", free_fractional=False)


def solve_T4(eq: EquivalentInstance) -> Solution:
    return run_subplan_dag(eq, solve_subplan_T4, "T4")
