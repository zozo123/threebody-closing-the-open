#!/usr/bin/env python3
"""Run one sharded ±1 rank-event audit slice for CI fan-out."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import audit_minus_one_rank_loss_corridor as minus
import audit_plus_one_rank_jump as plus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("minus_one", "plus_one"))
    parser.add_argument("input")
    parser.add_argument("m1", type=float)
    parser.add_argument("output")
    args = parser.parse_args()

    payload = minus.load(Path(args.input)) if args.kind == "minus_one" else plus.load(Path(args.input))
    if args.kind == "minus_one":
        rows = minus.corridor_rows(payload)
        matches = [row for row in rows if abs(float(row["masses"][0]) - args.m1) <= 5e-7]
        if len(matches) != 1:
            raise SystemExit(f"expected one minus-one corridor row at m1={args.m1}, got {len(matches)}")
        result = minus.audit_row(matches[0])
        passed = bool(result["rank_loss_passed"])
    else:
        rows = plus.plus_targets(payload)
        matches = [row for row in rows if abs(float(row["m1"]) - args.m1) <= 5e-7]
        if len(matches) != 1:
            raise SystemExit(f"expected one plus-one target at m1={args.m1}, got {len(matches)}")
        result = plus.audit(matches[0])
        passed = bool(result["rank_jump_passed"])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "schema": "atlas.v1.rank-event-slice/1",
                "kind": args.kind,
                "m1": args.m1,
                "passed": passed,
                "result": result,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"kind": args.kind, "m1": args.m1, "passed": passed}, indent=2))
    raise SystemExit(0 if passed else 3)


if __name__ == "__main__":
    main()
