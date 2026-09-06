"""Tests: reformulation correctness, the 2C check, and T3 == T4 == exact optimum."""
from __future__ import annotations

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clsp_ib.generator import out_of_scope_instance, random_instance
from clsp_ib.instance import Instance, StorageCapacityError
from clsp_ib.reference import exact_optimum, plan_cost
from clsp_ib.reformulate import check_storage_capacity, to_equivalent_model
from clsp_ib.solve import compare
from clsp_ib.t3 import solve_T3
from clsp_ib.t3_init import validate_initialisation
from clsp_ib.t4 import solve_T4

TOL = 1e-7


def _rel_close(a: float, b: float, tol: float = TOL) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


class TestReformulation(unittest.TestCase):
    def test_holding_cost_folding(self):
        """modified objective + offset must equal the original objective."""
        rng = random.Random(7)
        for trial in range(40):
            inst = random_instance(6, C=10, seed=100 + trial)
            eq = to_equivalent_model(inst)
            # a feasible plan: produce exactly the demand each period
            x = list(inst.d)
            original = plan_cost(inst, x)
            # the same plan in the equivalent model (demand is unchanged here
            # because d_t <= C already, so x carries over)
            inv = eq.I0
            modified = 0.0
            for t in range(1, eq.T + 1):
                q = x[t - 1]
                inv += q - eq.d[t - 1]
                modified += (eq.s[t - 1] if q > 1e-12 else 0.0) + eq.c[t - 1] * q
            self.assertTrue(
                _rel_close(original, modified + eq.cost_offset),
                f"trial {trial}: original {original} vs {modified + eq.cost_offset}",
            )

    def test_reformulation_preserves_optimum(self):
        for trial in range(15):
            inst = random_instance(9, C=6, seed=500 + trial)
            eq = to_equivalent_model(inst)
            equiv_as_instance = Instance(
                T=eq.T, C=eq.C, d=eq.d, S=eq.S, s=eq.s, c=eq.c,
                h=[0.0] * eq.T, I0=eq.I0,
            )
            self.assertTrue(
                _rel_close(exact_optimum(inst), exact_optimum(equiv_as_instance) + eq.cost_offset),
                f"trial {trial}",
            )

    def test_demand_capped_and_bounds_tightened(self):
        inst = Instance(T=4, C=10, d=[0, 25, 3, 4], S=[100, 100, 100, 100],
                        s=[1, 1, 1, 1], c=[1, 1, 1, 1], h=[0, 0, 0, 0], I0=30)
        eq = to_equivalent_model(inst)
        self.assertTrue(all(v <= eq.C + 1e-9 for v in eq.d))
        self.assertAlmostEqual(sum(eq.d), sum(inst.d) - (inst.I0 - eq.I0))
        self.assertLessEqual(eq.S[0], eq.I0 + eq.C - eq.d[0] + 1e-9)
        for t in range(2, eq.T + 1):
            self.assertLessEqual(eq.S[t - 1], eq.S[t - 2] + eq.C - eq.d[t - 1] + 1e-9)
        for t in range(1, eq.T):
            self.assertLessEqual(eq.S[t - 1], eq.S[t] + eq.d[t] + 1e-9)


class TestCapacityCheck(unittest.TestCase):
    def test_zero_initial_inventory_is_rejected(self):
        for T in (5, 12):
            inst = out_of_scope_instance(T, seed=3)
            with self.assertRaises(StorageCapacityError):
                check_storage_capacity(to_equivalent_model(inst))
            with self.assertRaises(StorageCapacityError):
                compare(inst)

    def test_in_scope_instances_pass(self):
        for trial in range(20):
            eq = to_equivalent_model(random_instance(12, seed=trial))
            check_storage_capacity(eq)  # must not raise
            self.assertTrue(all(v >= 2 * eq.C - 1e-9 for v in eq.S))


class TestAlgorithms(unittest.TestCase):
    def test_T3_equals_T4_equals_exact(self):
        for T in (1, 2, 3, 5, 8, 12):
            for trial in range(8):
                inst = random_instance(T, C=6, seed=1000 * T + trial)
                eq = to_equivalent_model(inst)
                check_storage_capacity(eq)
                s3, s4 = solve_T3(eq), solve_T4(eq)
                s3.check_feasible(eq)
                s4.check_feasible(eq)
                ref = exact_optimum(inst)
                self.assertTrue(_rel_close(s4.cost, ref),
                                f"T={T} trial={trial}: T4 {s4.cost} vs exact {ref}")
                self.assertTrue(_rel_close(s3.cost, ref),
                                f"T={T} trial={trial}: T3 {s3.cost} vs exact {ref}")
                self.assertTrue(_rel_close(plan_cost(inst, s3.x), s3.cost),
                                f"T={T} trial={trial}: T3 plan cost inconsistent")

    def test_no_holding_costs(self):
        for trial in range(10):
            inst = random_instance(10, C=8, seed=77 + trial, holding_hi=0)
            res = compare(inst)
            self.assertTrue(res.agree)
            self.assertTrue(_rel_close(res.t4.cost, exact_optimum(inst)))

    def test_zero_setup_costs(self):
        for trial in range(6):
            inst = random_instance(9, C=7, seed=200 + trial, setup_hi=0)
            res = compare(inst)
            self.assertTrue(res.agree)
            self.assertTrue(_rel_close(res.t4.cost, exact_optimum(inst)))

    def test_large_setup_costs(self):
        for trial in range(6):
            inst = random_instance(9, C=7, seed=300 + trial, setup_hi=500)
            res = compare(inst)
            self.assertTrue(res.agree)
            self.assertTrue(_rel_close(res.t4.cost, exact_optimum(inst)))

    def test_zero_demand(self):
        inst = Instance(T=6, C=10, d=[0] * 6, S=[30] * 6, s=[5] * 6,
                        c=[1] * 6, h=[0] * 6, I0=20)
        res = compare(inst)
        self.assertTrue(res.agree)
        self.assertTrue(_rel_close(res.t4.cost, exact_optimum(inst)))

    def test_both_initialisations_agree(self):
        for T in (1, 3, 6, 10, 14):
            for trial in range(5):
                inst = random_instance(T, C=6, seed=6100 + 31 * T + trial)
                eq = to_equivalent_model(inst)
                check_storage_capacity(eq)
                inc = solve_T3(eq, initialisation="incremental")
                exa = solve_T3(eq, initialisation="exact")
                inc.check_feasible(eq)
                self.assertTrue(_rel_close(inc.cost, exa.cost),
                                f"T={T} trial={trial}: {inc.cost} vs {exa.cost}")
                self.assertTrue(_rel_close(inc.cost, exact_optimum(inst)))

    def test_section4_initialisation_is_optimal(self):
        """Every seed the Section 4 procedure returns is feasible and optimal for P'(t0)."""
        for T in (4, 8, 11):
            for trial in range(5):
                eq = to_equivalent_model(random_instance(T, C=6, seed=7700 + 31 * T + trial))
                check_storage_capacity(eq)
                rep = validate_initialisation(eq)
                self.assertEqual(rep.infeasible_plan, 0, f"T={T} trial={trial}: {rep}")
                self.assertEqual(rep.suboptimal, 0, f"T={T} trial={trial}: {rep}")
                self.assertEqual(rep.t0_differs, 0, f"T={T} trial={trial}: {rep}")
                self.assertGreater(rep.covered, 0)

    def test_fractional_costs(self):
        for trial in range(6):
            inst = random_instance(8, C=6, seed=900 + trial, integral_costs=False)
            res = compare(inst)
            self.assertTrue(res.agree)
            self.assertTrue(_rel_close(res.t4.cost, exact_optimum(inst), 1e-6))


if __name__ == "__main__":
    unittest.main(verbosity=2)
