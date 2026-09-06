"""Random instances that satisfy the assumptions of Section 2.2.

To make ``Psi_t >= 2C`` satisfiable the initial inventory matters: tightening
gives ``S_1 <= I_0 + C - d_1``, so ``I_0 >= C + d_1`` is necessary.  The default
``I0 = 2*C`` together with ``d_t <= C`` and raw bounds at least ``3*C`` keeps
every tightened bound at or above ``2*C``.
"""
from __future__ import annotations

import random
from typing import Optional

from .instance import Instance


def random_instance(
    T: int,
    C: int = 10,
    seed: Optional[int] = None,
    S_lo_mult: float = 3.0,
    S_hi_mult: float = 5.0,
    setup_hi: int = 40,
    unit_hi: int = 10,
    holding_hi: int = 2,
    I0_mult: float = 2.0,
    integral_costs: bool = True,
) -> Instance:
    rng = random.Random(seed)
    d = [rng.randint(0, C) for _ in range(T)]
    S = [rng.randint(int(S_lo_mult * C), int(S_hi_mult * C)) for _ in range(T)]
    if integral_costs:
        s = [rng.randint(0, setup_hi) for _ in range(T)]
        c = [rng.randint(1, unit_hi) for _ in range(T)]
        h = [rng.randint(0, holding_hi) for _ in range(T)]
    else:
        s = [rng.uniform(0, setup_hi) for _ in range(T)]
        c = [rng.uniform(0.5, unit_hi) for _ in range(T)]
        h = [rng.uniform(0, holding_hi) for _ in range(T)]
    return Instance(T=T, C=C, d=d, S=S, s=s, c=c, h=h, I0=int(I0_mult * C))


def out_of_scope_instance(T: int, C: int = 10, seed: Optional[int] = None) -> Instance:
    """An instance with I_0 = 0, which always violates Psi_1 >= 2C."""
    inst = random_instance(T, C=C, seed=seed)
    return Instance(T=inst.T, C=inst.C, d=inst.d, S=inst.S, s=inst.s,
                    c=inst.c, h=inst.h, I0=0)
