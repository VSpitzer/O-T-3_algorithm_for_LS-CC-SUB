"""LS-CC-SUB / CLSP-IB: reference implementations of the O(T^4) and O(T^3) algorithms.

Public entry points
-------------------
Instance            original model (1a)-(1f)
to_equivalent_model reformulation of Section 2.2
solve_T4            state-of-the-art O(T^4) algorithm (Wolsey, 2005)
solve_T3            the O(T^3) algorithm of the article (Algorithms 1-5)
compare             reformulate, check Psi >= 2C, run both, assert equal cost
"""
from .instance import Instance, EquivalentInstance, InfeasibleInstance, StorageCapacityError
from .reformulate import to_equivalent_model, check_storage_capacity
from .subplan import Subplan
from .t4 import solve_T4
from .t3 import solve_T3
from .solve import Solution, compare, ComparisonResult

__all__ = [
    "Instance", "EquivalentInstance", "InfeasibleInstance", "StorageCapacityError",
    "to_equivalent_model", "check_storage_capacity", "Subplan",
    "solve_T4", "solve_T3", "Solution", "compare", "ComparisonResult",
]
