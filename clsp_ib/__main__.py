"""Command line entry point.

    python -m clsp_ib --T 20 --seed 1 --repeat 5
"""
from __future__ import annotations

import argparse
import sys

from .generator import random_instance
from .instance import StorageCapacityError
from .solve import compare


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="clsp_ib", description=__doc__)
    ap.add_argument("--T", type=int, default=15)
    ap.add_argument("--C", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--no-holding", action="store_true")
    args = ap.parse_args(argv)

    failures = 0
    for i in range(args.repeat):
        inst = random_instance(args.T, C=args.C, seed=args.seed + i,
                               holding_hi=0 if args.no_holding else 2)
        try:
            res = compare(inst, strict=False)
        except StorageCapacityError as exc:
            print(f"[{i}] rejected: {exc}")
            failures += 1
            continue
        print(f"[{i}] T={args.T} {res}")
        if not res.agree:
            failures += 1
    print(f"\n{args.repeat - failures}/{args.repeat} instances matched")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
