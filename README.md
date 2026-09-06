# clsp-ib

Exact algorithms for the **single-item capacitated lot-sizing problem with
constant capacity and inventory upper bounds** (LS-CC-SUB, also written
CLSP-IB):

```
min  sum_t ( s_t z_t + c_t x_t + h_t I_t )
s.t. x_t <= C z_t                      capacity, with setup
     I_t = I_{t-1} + x_t - d_t         inventory balance, I_0 given
     I_t <= S_t                        inventory upper bound
     x_t, I_t >= 0,  z_t in {0,1}
```

Two exact methods are implemented and can be run against each other:

| | complexity | source |
|---|---|---|
| **T4** | O(T^4) | the state of the art (Wolsey, 2005): solve every subplan, then shortest path |
| **T3** | O(T^3) | the algorithm under study: subplans solved by a dynamic procedure that is linear in the subplan length |

Both call the **same** per-subplan routine, so a disagreement can only come from
the dynamic procedure and not from a difference in how subplans are evaluated.
A MILP formulation and an exact dynamic program are included as independent
references.

---

## Install

```bash
git clone <your-repo-url> && cd clsp-ib
pip install -e .                # the algorithms, no dependencies
pip install -e ".[milp]"        # also pulp, which bundles a CBC solver
```

Python 3.8+, and a pip new enough to read PEP 621 metadata (setuptools >= 64,
i.e. any pip from 2022 onwards). The core package has **no dependencies**.

## Quickstart

```python
from clsp_ib import Instance, compare

inst = Instance(
    T=5, C=10,
    d=[4, 9, 2, 7, 5],        # demand
    S=[35, 40, 38, 36, 33],   # inventory upper bounds
    s=[12, 8, 15, 9, 11],     # setup costs
    c=[3, 2, 4, 2, 3],        # unit production costs
    h=[1, 1, 1, 1, 1],        # unit holding costs
    I0=20,                    # initial inventory
)

result = compare(inst)
print(result)          # MATCH: T3 = 56, T4 = 56, |gap| = 0
print(result.t3.x)     # the production plan
print(result.t3.I)     # the resulting inventory levels
```

`compare` runs the whole pipeline:

1. **reformulate** into the equivalent model of Section 2.2 -- holding costs
   folded into the production costs, demand capped at `C`, inventory bounds
   tightened;
2. **verify** that the potential for storage is at least twice the production
   capacity, raising `StorageCapacityError` if not;
3. **solve** with T4 and with T3, feasibility-checking both plans;
4. **compare** the two costs, raising `CostMismatch` on disagreement
   (`strict=False` returns the result instead).

Costs are reported in the objective of the original model: the constant produced
by folding the holding costs is added back automatically.

To run a single algorithm:

```python
from clsp_ib import to_equivalent_model, check_storage_capacity, solve_T3, solve_T4

eq = to_equivalent_model(inst)
check_storage_capacity(eq)
sol = solve_T3(eq)                              # or solve_T4(eq)
sol = solve_T3(eq, initialisation="exact")      # bypass the Section 4 procedure
```

## Validate it yourself

Three independent checks, in increasing order of thoroughness.

**1. Unit tests** -- reformulation algebra, the 2C check, T3 = T4 = exact DP,
both initialisations, and the Section 4 seeds. 17 tests, of which 2 are skipped
unless a MILP backend is installed:

```bash
python -m unittest discover -s tests -t . -v
```

**2. Against a MILP solver** -- the point of comparison most readers will want.
Needs `pip install pulp` (a CBC binary ships with the wheel); the MILP is solved
at a **zero** MIP gap so the numbers are directly comparable.

```bash
python -m clsp_ib.validate_milp --repeat 20 --T 12
```

```
   #    T             T3             T4           MILP             DP  verdict
------------------------------------------------------------------------------
   0   12            517            517            517            517  ok
   1   12            438            438            438            438  ok
...
20/20 instances matched (0 rejected, 0 mismatched) in 3.4s
```

Without a MILP backend the MILP column is dropped and the exact DP still gives
an independent answer. `python -m unittest tests.test_vs_milp -v` does the same
as a test, skipping the MILP assertions if no solver is installed.

**3. Randomised stress test** -- T3 (both initialisations), T4 and the exact DP
over many horizons, including instances whose demand exceeds `C` so the
demand-capping reformulation is genuinely exercised:

```bash
python -m clsp_ib.stress --max-T 16 --per-T 8 --C 6
```

Why a dynamic program is a valid reference: for a fixed setup pattern the
remaining problem is a network flow with integral data, so its linear relaxation
has an integral optimum. Enumerating integer production quantities and integer
inventory levels therefore gives the true optimum, with no gap and no solver.
It requires integral `C`, `d`, `S` and `I0`; costs may be arbitrary floats.

## The `Psi_t >= 2C` condition, and why `I_0` matters

The T3 algorithm is proved correct only for instances whose **potential for
storage** -- the spread between the highest and lowest feasible inventory level
in a period -- is at least twice the production capacity everywhere. After
reformulation the equivalent model has `Psi_t = S_t`, so the check is
`S_t >= 2C` for all `t`, and `check_storage_capacity` raises otherwise.

The initial inventory is what makes this satisfiable. Tightening always gives

```
S_1 <= I_0 + C - d_1
```

so **`I_0 >= C + d_1` is necessary** for the condition to hold at `t = 1`. Every
instance with `I_0 = 0` is out of scope and will be rejected. `I_0` is therefore
kept as an explicit parameter of the equivalent model rather than being absorbed
into the demand of the first periods.

```python
from clsp_ib.generator import random_instance, out_of_scope_instance

random_instance(20, C=10, seed=0)      # in scope: I_0 = 2C, bounds in [3C, 5C]
out_of_scope_instance(20, seed=0)      # I_0 = 0, correctly rejected
```

## The C++ implementation

`cpp/` holds the original C++ implementation of T3 and its validation harness.
See [`cpp/README.md`](cpp/README.md). Quick version:

```bash
cd cpp
make BOOST=shim          # or plain `make` if Boost.Graph is installed
./validate_t3 20 2 4 999
```

## Layout

```
clsp_ib/                 the Python package
  instance.py            Instance (model (1a)-(1f)), EquivalentInstance, cf(t)
  reformulate.py         Section 2.2 transformations, the Psi >= 2C check
  subplan.py             subplans, equation (8), the shared T2 solver for P and P'
  dag.py                 the shortest-path phase over subplans
  t4.py                  the O(T^4) algorithm
  t3.py                  Algorithms 1-5 of Section 6
  t3_init.py             Section 4's dynamic initialisation
  solve.py               compare()
  milp.py                MILP formulation (pulp or pyomo)
  reference.py           exact integer-inventory dynamic program
  generator.py           random in-scope instances
  stress.py              randomised T3 vs T4 vs DP sweep
  validate_milp.py       randomised T3 vs T4 vs MILP vs DP sweep
cpp/                     the C++ implementation, Makefile, minimal Boost stand-in
tests/                   unit tests and the MILP comparison test
docs/algorithm_notes.md  where this implementation departs from the printed algorithms, and why
```

## Notes for readers of the article

`docs/algorithm_notes.md` records every place this implementation departs from
the algorithms as printed, with the reasoning. The three that matter:

* Algorithm 5 registers a candidate for `P(t0)` only inside its `while` loop,
  which decrements `t0` first, yet line 24 minimises over `[t_min, t_max]` and
  `t_max` can equal `t'_max`. A candidate is registered for `t'_max` as well.
* Equation (2) prints the folded production cost as a prefix sum of the holding
  costs; reproducing `sum_t h_t I_t` needs the suffix sum plus a constant
  `I_0 * sum_t h_t`. `test_holding_cost_folding` pins this down.
* Definition 2 writes the entry inventory of a subplan as `I^1 * S_{t1-1}`,
  which has no meaning for the first subplan, and Love's decomposition assumes
  the horizon starts empty. With `I_0 > d_{1,T}` the cheapest plan produces
  nothing and ends at an interior inventory level, so `dag.exit_levels` offers
  that residual level at `t2 = T`.

## Status

Verified at the time of writing:

* 17 unit tests pass.
* T3 (both initialisations), T4 and the exact DP agree to machine precision on
  several hundred random instances over `T = 1..40`, including instances that
  require demand capping and instances correctly rejected by the 2C check.
* The C++ harness reports 0 deviations from the exact DP over 660 instances at
  `T` in {110, 120, 150, 200}, across seven seed/bound configurations, a
  tie-heavy cost range, and `x_on` in {0, 2, 5, 8}; it is clean under
  AddressSanitizer and UBSan.
* The MILP comparison path (`pulp`/`pyomo`) has been exercised against a stub
  solver but not against a real installed solver, so run
  `python -m clsp_ib.validate_milp` once yourself after installing `pulp`.

## Licence

No licence file is included yet -- add one before making the repository public.
