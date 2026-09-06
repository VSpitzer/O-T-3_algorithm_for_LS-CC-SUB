"""Section 2.2: reformulation into the equivalent model, and the Psi >= 2C check."""
from __future__ import annotations

from typing import List, Tuple

from .instance import EquivalentInstance, Instance, InfeasibleInstance, StorageCapacityError


def fold_holding_costs(inst: Instance) -> Tuple[List[float], float]:
    """Section 2.2.1 -- integrate holding costs into the production costs.

    sum_t h_t I_t  with  I_t = I_0 + sum_{u<=t} x_u - d_{1,t}  gives

        sum_t h_t I_t = I_0 * sum_t h_t
                        + sum_u x_u * ( sum_{t>=u} h_t )
                        - sum_t h_t d_{1,t}

    so the coefficient picked up by x_u is the SUFFIX sum sum_{t>=u} h_t, and

        c'_u = c_u + sum_{t=u}^{T} h_t

    with the remaining terms a constant.  Returns ``(c_prime, offset)``.

    Note: equation (2) of the article prints this as
    ``c'_t = c_t + sum_{t'=1}^{t} h_{t'}`` (a prefix sum) and omits the
    ``I_0 * sum_t h_t`` term.  The suffix sum is the one that reproduces
    ``sum_t h_t I_t``; see ``tests/test_clsp_ib.py::test_holding_cost_folding``.
    """
    T = inst.T
    suffix = [0.0] * (T + 1)
    for t in range(T, 0, -1):
        suffix[t - 1] = suffix[t] + inst.h[t - 1]
    c_prime = [inst.c[t - 1] + suffix[t - 1] for t in range(1, T + 1)]

    offset = inst.I0 * sum(inst.h)
    cum = 0.0
    for t in range(1, T + 1):
        cum += inst.d[t - 1]
        offset -= inst.h[t - 1] * cum
    return c_prime, offset


def cap_demand(inst: Instance) -> Tuple[List[float], List[float], float]:
    """Section 2.2.2 -- make the demand not exceed C in every period.

    ``I_bar_t`` is the minimum inventory that must be carried out of period t
    for the remaining demand to be producible at capacity:

        I_bar_T = 0,   I_bar_t = max(0, I_bar_{t+1} + d_{t+1} - C)

    Substituting I'_t = I_t - I_bar_t gives an equivalent instance with

        d'_t = d_t + I_bar_t - I_bar_{t-1},   S'_t = S_t - I_bar_t,
        I'_0 = I_0 - I_bar_0

    which preserves total demand and satisfies d'_t <= C.
    """
    T = inst.T
    I_bar = [0.0] * (T + 1)  # I_bar[t] for t = 0..T
    for t in range(T - 1, -1, -1):
        I_bar[t] = max(0.0, I_bar[t + 1] + inst.d[t] - inst.C)

    d_new = [inst.d[t - 1] + I_bar[t] - I_bar[t - 1] for t in range(1, T + 1)]
    S_new = [inst.S[t - 1] - I_bar[t] for t in range(1, T + 1)]
    I0_new = inst.I0 - I_bar[0]

    if I0_new < -1e-9:
        raise InfeasibleInstance(
            "initial inventory too small: demand cannot be met at capacity "
            f"(need I_0 >= {I_bar[0]}, got {inst.I0})"
        )
    for t in range(1, T + 1):
        if S_new[t - 1] < -1e-9:
            raise InfeasibleInstance(
                f"storage bound too small in period {t}: anticipated production "
                f"of {I_bar[t]} units cannot be stored"
            )
    return d_new, S_new, max(0.0, I0_new)


def tighten_bounds(T: int, C: float, d: List[float], S: List[float], I0: float) -> List[float]:
    """Section 2.2.3 -- tighten the inventory upper bounds, inequalities (4)-(6).

        S_1 <- min(S_1, I_0 + C - d_1)
        S_t <- min(S_t, S_{t-1} + C - d_t)      t = 2..T
        S_t <- min(S_t, S_{t+1} + d_{t+1})      t = T-1..1
    """
    S = list(S)
    S[0] = min(S[0], I0 + C - d[0])
    for t in range(2, T + 1):
        S[t - 1] = min(S[t - 1], S[t - 2] + C - d[t - 1])
    for t in range(T - 1, 0, -1):
        S[t - 1] = min(S[t - 1], S[t] + d[t])
    for t in range(1, T + 1):
        if S[t - 1] < -1e-9:
            raise InfeasibleInstance(f"tightened storage bound is negative in period {t}")
    return S


def to_equivalent_model(inst: Instance) -> EquivalentInstance:
    """Apply Sections 2.2.1, 2.2.2 and 2.2.3 in that order.

    The initial inventory I_0 is kept as an explicit parameter, as in
    inequality (4) of the article.  It is deliberately NOT eliminated by
    absorbing it into the demand: doing so forces S_1 <= C - d_1 and makes the
    Psi_t >= 2C hypothesis unsatisfiable at t = 1 for every instance.
    """
    c_prime, offset = fold_holding_costs(inst)
    d_new, S_new, I0_new = cap_demand(inst)
    S_new = tighten_bounds(inst.T, inst.C, d_new, S_new, I0_new)
    return EquivalentInstance(
        T=inst.T, C=inst.C, d=d_new, S=S_new,
        s=list(inst.s), c=c_prime, I0=I0_new, cost_offset=offset,
    )


def check_storage_capacity(eq: EquivalentInstance, tol: float = 1e-9) -> None:
    """Verify Psi_t >= 2*C for every period, and raise otherwise.

    By Section 2.2.4 the equivalent model has Psi_t = S_t, so the condition
    under which the O(T^3) algorithm is valid reduces to S_t >= 2*C.
    """
    bad = [(t, eq.S[t - 1]) for t in range(1, eq.T + 1) if eq.S[t - 1] < 2 * eq.C - tol]
    if bad:
        shown = ", ".join(f"t={t}: S_t={v:g}" for t, v in bad[:8])
        more = "" if len(bad) <= 8 else f" (and {len(bad) - 8} more)"
        raise StorageCapacityError(
            f"the potential for storage is below 2*C = {2 * eq.C:g} in "
            f"{len(bad)} of {eq.T} periods after reformulation: {shown}{more}. "
            "The O(T^3) algorithm is not applicable to this instance. Note that "
            "S_1 <= I_0 + C - d_1 always holds after tightening, so I_0 >= C + d_1 "
            "is necessary for the condition to be satisfiable at t = 1."
        )
