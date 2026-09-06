"""Compare T3 and T4 against a MILP solve of model (1a)-(1f).

The MILP part is skipped when neither pulp nor pyomo is installed; the exact
dynamic program is always used, so the test remains meaningful either way.

    pip install pulp        # gives a bundled CBC, no system solver needed
    python -m unittest tests.test_vs_milp -v
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clsp_ib.generator import random_instance
from clsp_ib.milp import NoSolverAvailable, available_backends, solve_milp
from clsp_ib.reference import exact_optimum, plan_cost
from clsp_ib.reformulate import check_storage_capacity, to_equivalent_model
from clsp_ib.solve import compare

TOL = 1e-6
HAVE_MILP = bool(available_backends())


def _close(a: float, b: float, tol: float = TOL) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


class TestAgainstMilp(unittest.TestCase):
    @unittest.skipUnless(HAVE_MILP, "no MILP backend installed (pip install pulp)")
    def test_algorithms_match_milp(self):
        for T in (1, 4, 9, 14):
            for trial in range(4):
                inst = random_instance(T, C=7, seed=9000 + 41 * T + trial)
                eq = to_equivalent_model(inst)
                check_storage_capacity(eq)
                res = compare(inst)
                milp = solve_milp(inst, gap=0.0)
                self.assertTrue(
                    _close(res.t4.cost, milp.cost),
                    f"T={T} trial={trial}: T4 {res.t4.cost} vs MILP {milp.cost} "
                    f"({milp.backend}, {milp.status})",
                )
                self.assertTrue(
                    _close(res.t3.cost, milp.cost),
                    f"T={T} trial={trial}: T3 {res.t3.cost} vs MILP {milp.cost}",
                )

    @unittest.skipUnless(HAVE_MILP, "no MILP backend installed (pip install pulp)")
    def test_milp_matches_exact_dp(self):
        """Guards the MILP wiring itself: it must reproduce the DP optimum."""
        for trial in range(6):
            inst = random_instance(10, C=6, seed=5500 + trial)
            self.assertTrue(_close(solve_milp(inst, gap=0.0).cost, exact_optimum(inst)))

    def test_algorithms_match_exact_dp(self):
        """Always runs: no MILP solver required."""
        for T in (1, 4, 9, 14):
            for trial in range(4):
                inst = random_instance(T, C=7, seed=9000 + 41 * T + trial)
                eq = to_equivalent_model(inst)
                check_storage_capacity(eq)
                res = compare(inst)
                ref = exact_optimum(inst)
                self.assertTrue(_close(res.t3.cost, ref), f"T={T} trial={trial}")
                self.assertTrue(_close(res.t4.cost, ref), f"T={T} trial={trial}")
                self.assertTrue(_close(plan_cost(inst, res.t3.x), res.t3.cost))

    def test_reports_missing_backend_clearly(self):
        if HAVE_MILP:
            self.skipTest("a backend is installed")
        with self.assertRaises(NoSolverAvailable) as ctx:
            solve_milp(random_instance(5, seed=1))
        self.assertIn("pip install", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
