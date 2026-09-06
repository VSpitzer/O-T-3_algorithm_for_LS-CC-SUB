"""Top level: reformulate, check the hypothesis, run both algorithms, compare."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .dag import Solution
from .instance import EquivalentInstance, Instance
from .reformulate import check_storage_capacity, to_equivalent_model
from .t3 import solve_T3
from .t4 import solve_T4


class CostMismatch(AssertionError):
    """The two algorithms disagreed on the optimal cost."""


@dataclass
class ComparisonResult:
    equivalent: EquivalentInstance
    t3: Solution
    t4: Solution
    gap: float
    agree: bool

    def __str__(self) -> str:  # pragma: no cover - reporting aid
        verdict = "MATCH" if self.agree else "MISMATCH"
        return (
            f"{verdict}: T3 = {self.t3.cost:.10g}, T4 = {self.t4.cost:.10g}, "
            f"|gap| = {self.gap:.3g}  "
            f"(subplan solves: T3 {self.t3.n_subplans_solved}, T4 {self.t4.n_subplans_solved})"
        )


def compare(
    inst: Instance,
    tol: float = 1e-6,
    check_capacity: bool = True,
    strict: bool = True,
) -> ComparisonResult:
    """Run the full pipeline on an instance of the original model.

    1. reformulate into the equivalent model of Section 2.2;
    2. verify that the potential for storage is at least twice the production
       capacity, raising ``StorageCapacityError`` otherwise;
    3. solve with the O(T^4) algorithm and with the O(T^3) algorithm;
    4. check that both report the same optimal cost.

    With ``strict=True`` a disagreement raises ``CostMismatch``; otherwise the
    result is returned with ``agree=False``.
    """
    eq = to_equivalent_model(inst)
    if check_capacity:
        check_storage_capacity(eq)

    sol4 = solve_T4(eq)
    sol3 = solve_T3(eq)
    sol4.check_feasible(eq)
    sol3.check_feasible(eq)

    gap = abs(sol3.cost - sol4.cost)
    agree = gap <= tol * max(1.0, abs(sol4.cost))
    if strict and not agree:
        raise CostMismatch(
            f"T3 cost {sol3.cost!r} != T4 cost {sol4.cost!r} (gap {gap!r})"
        )
    return ComparisonResult(equivalent=eq, t3=sol3, t4=sol4, gap=gap, agree=agree)
