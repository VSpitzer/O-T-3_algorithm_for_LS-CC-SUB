# clsp_ib

Reference implementations of the two exact algorithms for the single-item
capacitated lot-sizing problem with constant capacity and inventory upper
bounds (LS-CC-SUB / CLSP-IB):

* **T4** -- the state-of-the-art O(T^4) algorithm (Wolsey, 2005): every subplan
  is solved independently, then the cheapest concatenation is found by a
  shortest path.
* **T3** -- the algorithm of the article: Algorithms 1-5 of Section 6, which
  solve each subplan by a dynamic procedure that is linear in the subplan
  length.

Both use the **same** per-subplan T2 routine (`subplan.solve_subplan`), so a
disagreement can only come from the dynamic procedure, not from a difference in
how subplans are evaluated.

## Pipeline

```python
from clsp_ib import Instance, to_equivalent_model, check_storage_capacity, compare

inst = Instance(T=5, C=10,
                d=[4, 9, 2, 7, 5],
                S=[35, 40, 38, 36, 33],
                s=[12, 8, 15, 9, 11],
                c=[3, 2, 4, 2, 3],
                h=[1, 1, 1, 1, 1],
                I0=20)

result = compare(inst)      # reformulate -> check Psi >= 2C -> T4 -> T3 -> compare
print(result)
```

`compare` performs, in order:

1. **Reformulation** into the equivalent model of Section 2.2 -- holding costs
   folded into the production costs (2.2.1), demand capped at `C` (2.2.2),
   inventory bounds tightened (2.2.3).
2. **Verification** that the potential for storage is at least twice the
   production capacity. By Section 2.2.4 the equivalent model has `Psi_t = S_t`,
   so this is `S_t >= 2C` for every period; otherwise
   `StorageCapacityError` is raised.
3. **Both algorithms**, followed by a feasibility check of each returned plan.
4. **Cost comparison**; `CostMismatch` is raised on disagreement unless
   `strict=False`.

Costs are reported in the objective of the *original* model (1a): the constant
produced by folding the holding costs is added back via `cost_offset`.

## The `Psi_t >= 2C` condition and the initial inventory

Tightening always gives `S_1 <= I_0 + C - d_1`, so `I_0 >= C + d_1` is
*necessary* for the hypothesis to hold at `t = 1`. In particular every instance
with `I_0 = 0` is out of scope, and `compare` will reject it. `I_0` is therefore
kept as an explicit parameter of the equivalent model rather than being absorbed
into the demand of the first periods.

`generator.random_instance` produces in-scope instances (`I_0 = 2C`, raw bounds
in `[3C, 5C]`, `d_t <= C`); `generator.out_of_scope_instance` produces one that
is correctly rejected.

## Layout

| file | contents |
|---|---|
| `instance.py` | `Instance` (model (1a)-(1f)), `EquivalentInstance`, `cf(t) = s_t + C c_t` |
| `reformulate.py` | Section 2.2 transformations and the `Psi >= 2C` check |
| `subplan.py` | `Subplan` (Definition 2), equation (8), and the shared T2 solver for `P` and `P'` |
| `dag.py` | the shortest-path phase over subplans, shared by both algorithms |
| `t4.py` | the O(T^4) algorithm |
| `t3.py` | Algorithms 1-5 of Section 6 |
| `t3_init.py` | Section 4's dynamic initialisation, ported from `init_solve_T3` |
| `solve.py` | `compare` |
| `reference.py` | exact integer-inventory DP, used by the tests as ground truth |
| `generator.py` | random in-scope instances |

## Known deviations from the printed algorithms

Documented at the point of use in `t3.py`:

1. **Two initialisations.** `solve_T3(eq, initialisation="incremental")` (the
   default) uses the Section 4 procedure ported in `t3_init.py`;
   `initialisation="exact"` solves every `P'(t'_max)` with the shared T2 solver.
   The incremental walk does not cover every subplan -- it stops when the
   fractional period would fall before `t1` -- so the exact solver fills the
   remainder and no subplan arc is lost. `t_min`, `t_max` and `t'_min` are still
   obtained by the exact O(T^2) scan in both modes, which is the remaining
   obstacle to an asymptotically O(T^3) implementation.
2. **Algorithm 5 never registers a candidate for `t0 = t'_max`.** Lines 17-23
   sit inside the `while` loop, which decrements `t0` first, yet line 24
   minimises over `[t_min, t_max]` and `t_max` can equal `t'_max`. A candidate
   is registered for `t'_max` before the loop.
3. **Line 13's inventory range.** `[gamma_t0, t0]` is correct whenever
   `delta > t0`, and entries above `t0` are never read again. When `delta = t0`
   (allowed by line 15) that range would wrongly decrement `I_{t0}`, so the
   exact range `[gamma_t0, delta - 1]` capped at `t0` is used.

## One notation for inventory headroom

The C++ uses the name `J` for two different quantities: `S_{t-1} - I_{t-1}` in
`dyna_solve_T3`, and that same quantity shifted by `-C` in `init_solve_T3`, whose
recursion is anchored on the post-insertion end inventory while `I` is one batch
short. Reading the second as the first makes the C++ test `J[t_J+1] >= 0` look
like it is missing a `- C` when it is not; tightening it to `>= C` applies the
shift twice and is what made 28 of 36 validation instances deviate.

This package avoids the trap by using a single quantity throughout,

```
slack(t) = S_t - I_t
```

and by stating each requirement over the interval on which it must actually hold
rather than folding it into the terminating test of a backward scan:

| operation | admissible periods `p` |
|---|---|
| insert a full batch in `p` | `min slack(t) over [p, t2] >= C` |
| remove a full batch from `p` | `min I_t over [p, t2] >= C` |

Both come from suffix minima computed in O(t2 - t1), so the cost is unchanged.
This also closes a gap in the C++ scans, which start at `t0` and therefore never
examine the periods in `(t0, t2]`; on the instances tested `t0 = t2` whenever the
relocation branch fires, so the gap is latent there. When no admissible period
exists the incremental walk stops and the exact solver covers the remaining
subplans, so no subplan arc is lost.

`t3_init.validate_initialisation(eq)` checks this independently: for every
subplan it verifies the seed is feasible and of the same cost as the exact
optimum of `P'(t0)`. Over 30 instances and 3712 covered subplans: 0 infeasible,
0 suboptimal, and the C++ formula for `t'_max` matched the latest genuinely
feasible fractional period every time.

## Tests

```
python -m unittest discover -s clsp_ib/tests -t . -v     # 11 unit tests
python -m clsp_ib --T 20 --seed 0 --repeat 5             # quick T3 vs T4 check
python -m clsp_ib.stress --max-T 22 --per-T 4 --C 8      # T3 vs T4 vs exact optimum
```

`stress.py` also generates instances whose demand exceeds `C`, so that the
demand-capping step of Section 2.2.2 is exercised rather than being a no-op.

Verified at the time of writing: 13 unit tests, plus 232 instances across
`T = 1..16` in which T3 (both initialisations), T4 and the exact reference all
agree to machine precision -- 87 of them required demand capping and 22 were
correctly rejected by the `Psi >= 2C` check. An earlier run without the
incremental initialisation covered `T = 1..40`.
