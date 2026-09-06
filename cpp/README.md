# C++ implementation

`solve_CLSP_IB.cpp` is the original implementation of the O(T^3) algorithm.
`validate_T3.cpp` is a harness that generates random instances, runs the
algorithm, and compares it against an exact integer-inventory dynamic program.

## Requirements

* a C++17 compiler
* OpenMP
* Boost.Graph -- **or** the bundled `boost_shim/`

`boost_shim/` implements only the four Boost.Graph facilities this file uses
(`adjacency_list`, `compressed_sparse_row_graph`, `bellman_ford_shortest_paths`,
`add_edge`/`num_vertices`). It exists so the code can be built and validated
without installing Boost; it is not a general replacement.

## Build and run

```bash
make                     # system Boost
make BOOST=shim          # bundled stand-in

./solve_clsp_ib          # the file's own main(): times a built-in instance set
./validate_t3 <instances_per_T> <S_lo> <S_hi> <seed> [cost_max] [x_on]
```

The harness prints, per horizon, how many instances deviate from the exact
optimum. Zero is the expected outcome:

```bash
./validate_t3 20 2 4 999
T=110  -> 0/20 deviate (all positive), total excess=0
T=120  -> 0/20 deviate (all positive), total excess=0
T=150  -> 0/20 deviate (all positive), total excess=0
```

Arguments: `S_lo`/`S_hi` are the raw storage bounds as multiples of `C`,
`cost_max` the largest integer unit cost (small values force frequent cost ties,
which is a useful stress), and `x_on` the setup consumption.

### Memory checking

```bash
make asan
ASAN_OPTIONS=detect_leaks=0 ./validate_t3_asan 20 2 4 999
```

## Cost model

This implementation specialises the objective: the setup consumes `x_on` units
at the period's unit cost, so `s_t = x_on * c_t` and

```
cf(t) = s_t + C * c_t = (C + x_on) * c_t
```

which is why `(C + x_on) * cf[t]` appears throughout, where `cf[]` holds the
unit costs `c_t`. The Python package uses the general `s_t`, `c_t`, `h_t` of
model (1a)-(1f); to compare the two, build an `Instance` with
`s_t = x_on * c_t` and `h_t = 0`.

## Two pitfalls worth knowing

**`NUM_THREADS`.** Keep it at 1 while validating. The `tab_t1` work-sharing
permutation at the top of `CLSP_T3` is only a permutation when `NUM_THREADS`
divides `T`; otherwise it produces out-of-range `t1` values. For `T = 10` and
`NUM_THREADS = 8`, `t = 6` yields `t1 = 12`.

**`J` means two different things.** In `dyna_solve_T3`,
`J[t] = S[t-1] - I[t-1]`. In `init_solve_T3` the same name denotes that quantity
shifted by `-C`, because its recursion is anchored on the post-insertion end
inventory while `I` is one batch short:

```
J_init[t] = S[t-1] - I[t-1] - C
```

So the scan test `J[t_J+1] >= 0` in `init_solve_T3` **already** requires a full
batch of inventory headroom. Tightening it to `>= C` applies the shift twice,
makes the scan terminate early, and silently drops subplans -- roughly three
quarters of validation instances then return suboptimal plans. The Python port
in `clsp_ib/t3_init.py` avoids the ambiguity by using one quantity,
`slack(t) = S_t - I_t`, and stating each requirement over the interval on which
it must hold.
