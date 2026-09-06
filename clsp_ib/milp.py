"""MILP formulation of model (1a)-(1f), for independent validation.

The combinatorial algorithms in this package exploit a lot of structure, so it
is worth checking them against a formulation that assumes none of it.  This
module builds

    min  sum_t ( s_t z_t + c_t x_t + h_t I_t )
    s.t. x_t <= C z_t                       for all t
         I_t = I_{t-1} + x_t - d_t          for all t,  I_0 given
         I_t <= S_t                         for all t
         x_t, I_t >= 0,  z_t in {0,1}

and hands it to whichever solver is installed.  Two backends are supported:

``pulp``   ``pip install pulp`` -- ships a CBC binary, so nothing else is needed.
``pyomo``  ``pip install pyomo`` plus a solver on PATH (cbc, glpk, highs,
           cplex, gurobi).

Neither is a hard dependency: without one, ``clsp_ib.reference.exact_optimum``
still provides an exact answer for integral instances.

Note on tolerances: ask for a zero relative MIP gap when validating.  A default
gap of 1e-4 lets the solver stop above the optimum, which then shows up as a
spurious disagreement with the algorithms.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .instance import Instance


class NoSolverAvailable(RuntimeError):
    """Neither pulp nor pyomo (with a working solver) could be used."""


class MilpPlanInvalid(RuntimeError):
    """The solver returned a plan that violates the model."""


def _cost_of(inst: Instance, x: List[float], tol: float = 1e-6) -> float:
    """Objective (1a) recomputed from a plan, and a feasibility check.

    The cost is taken from the returned variable values rather than from the
    solver's reported objective: it removes a dependency on backend API details
    and it catches a solver that stops early or reports a stale bound.
    """
    inv = inst.I0
    total = 0.0
    for t in range(1, inst.T + 1):
        q = x[t - 1]
        if q < -tol or q > inst.C + tol:
            raise MilpPlanInvalid(f"x_{t} = {q} outside [0, C={inst.C}]")
        inv += q - inst.d[t - 1]
        if inv < -tol or inv > inst.S[t - 1] + tol:
            raise MilpPlanInvalid(f"I_{t} = {inv} outside [0, S_{t}={inst.S[t - 1]}]")
        total += ((inst.s[t - 1] if q > tol else 0.0)
                  + inst.c[t - 1] * q + inst.h[t - 1] * inv)
    return total


@dataclass
class MilpResult:
    cost: float
    x: List[float] = field(default_factory=list)
    I: List[float] = field(default_factory=list)
    backend: str = ""
    status: str = ""


def available_backends() -> List[str]:
    """Which MILP backends can be used right now."""
    out: List[str] = []
    try:
        import pulp  # noqa: F401
        out.append("pulp")
    except Exception:
        pass
    try:
        import pyomo.environ  # noqa: F401
        out.append("pyomo")
    except Exception:
        pass
    return out


def _solve_pulp(inst: Instance, gap: float, time_limit: Optional[float],
                solver_name: Optional[str]) -> MilpResult:
    import pulp

    T = inst.T
    prob = pulp.LpProblem("clsp_ib", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{t}", lowBound=0) for t in range(1, T + 1)]
    inv = [pulp.LpVariable(f"I_{t}", lowBound=0, upBound=inst.S[t - 1])
           for t in range(1, T + 1)]
    z = [pulp.LpVariable(f"z_{t}", cat="Binary") for t in range(1, T + 1)]

    prob += pulp.lpSum(
        inst.s[t - 1] * z[t - 1] + inst.c[t - 1] * x[t - 1] + inst.h[t - 1] * inv[t - 1]
        for t in range(1, T + 1)
    )
    for t in range(1, T + 1):
        prob += x[t - 1] <= inst.C * z[t - 1], f"cap_{t}"
        prev = inst.I0 if t == 1 else inv[t - 2]
        prob += inv[t - 1] == prev + x[t - 1] - inst.d[t - 1], f"bal_{t}"

    kwargs = {"msg": 0, "gapRel": gap}
    if time_limit is not None:
        kwargs["timeLimit"] = time_limit
    solver = pulp.getSolver(solver_name, **kwargs) if solver_name else pulp.PULP_CBC_CMD(**kwargs)
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        raise NoSolverAvailable(f"pulp returned status {status!r}")
    plan = [float(v.value()) for v in x]
    return MilpResult(
        cost=_cost_of(inst, plan),
        x=plan,
        I=[float(v.value()) for v in inv],
        backend="pulp",
        status=status,
    )


def _solve_pyomo(inst: Instance, gap: float, time_limit: Optional[float],
                 solver_name: Optional[str]) -> MilpResult:
    import pyomo.environ as pyo

    T = inst.T
    m = pyo.ConcreteModel()
    m.time = pyo.RangeSet(1, T)
    m.x = pyo.Var(m.time, domain=pyo.NonNegativeReals)
    m.I = pyo.Var(m.time, domain=pyo.NonNegativeReals)
    m.z = pyo.Var(m.time, domain=pyo.Binary)

    m.obj = pyo.Objective(
        expr=sum(inst.s[t - 1] * m.z[t] + inst.c[t - 1] * m.x[t] + inst.h[t - 1] * m.I[t]
                 for t in m.time),
        sense=pyo.minimize,
    )

    @m.Constraint(m.time)
    def capacity(mm, t):
        return mm.x[t] <= inst.C * mm.z[t]

    @m.Constraint(m.time)
    def balance(mm, t):
        prev = inst.I0 if t == 1 else mm.I[t - 1]
        return mm.I[t] == prev + mm.x[t] - inst.d[t - 1]

    @m.Constraint(m.time)
    def storage(mm, t):
        return mm.I[t] <= inst.S[t - 1]

    last_error = None
    for name in ([solver_name] if solver_name else ["cbc", "glpk", "highs", "cplex", "gurobi"]):
        try:
            opt = pyo.SolverFactory(name)
            if opt is None or not opt.available(exception_flag=False):
                continue
            for key, value in (("mipgap", gap), ("ratioGap", gap)):
                try:
                    opt.options[key] = value
                except Exception:
                    pass
            if time_limit is not None:
                for key in ("timelimit", "seconds", "TimeLimit"):
                    try:
                        opt.options[key] = time_limit
                    except Exception:
                        pass
            res = opt.solve(m, tee=False)
            plan = [float(pyo.value(m.x[t])) for t in m.time]
            return MilpResult(
                cost=_cost_of(inst, plan),
                x=plan,
                I=[float(pyo.value(m.I[t])) for t in m.time],
                backend=f"pyomo/{name}",
                status=str(res.solver.termination_condition),
            )
        except Exception as exc:  # try the next solver
            last_error = exc
    raise NoSolverAvailable(f"no pyomo solver worked (last error: {last_error})")


def solve_milp(
    inst: Instance,
    backend: str = "auto",
    gap: float = 0.0,
    time_limit: Optional[float] = None,
    solver_name: Optional[str] = None,
) -> MilpResult:
    """Solve model (1a)-(1f) exactly with a MILP solver.

    ``backend`` is ``"auto"``, ``"pulp"`` or ``"pyomo"``.  ``gap`` is the
    relative MIP gap and defaults to 0 so the result is comparable with the
    exact algorithms.
    """
    order = [backend] if backend != "auto" else ["pulp", "pyomo"]
    errors = []
    for name in order:
        if name == "pulp":
            try:
                return _solve_pulp(inst, gap, time_limit, solver_name)
            except ImportError as exc:
                errors.append(f"pulp: {exc}")
            except NoSolverAvailable as exc:
                errors.append(f"pulp: {exc}")
        elif name == "pyomo":
            try:
                return _solve_pyomo(inst, gap, time_limit, solver_name)
            except ImportError as exc:
                errors.append(f"pyomo: {exc}")
            except NoSolverAvailable as exc:
                errors.append(f"pyomo: {exc}")
        else:
            raise ValueError(f"unknown backend {backend!r}")
    raise NoSolverAvailable(
        "no MILP backend available. Install one with `pip install clsp-ib[milp]` "
        "(pulp, which bundles CBC) or `pip install pyomo` plus a solver. "
        "Details: " + "; ".join(errors)
    )
