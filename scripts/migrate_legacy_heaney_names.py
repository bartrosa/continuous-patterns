#!/usr/bin/env python3
"""One-shot rename of legacy SK/D2B comparison artifacts to Heaney naming.

Run once on an existing master_comparison timestamp directory, e.g.:

  uv run python scripts/migrate_legacy_heaney_names.py results/master_comparison/20260609T152334Z
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

_RUN_DIR_RENAMES = {
    "E1_SK_open": "E1_Heaney_open",
    "E2_SK_control": "E2_Heaney_control",
    "E3_SK_closed": "E3_Heaney_closed",
}

_METRICS_KEY_RENAMES = {
    **_RUN_DIR_RENAMES,
    "E9_D2B_best": "E9_Heaney_movingfront",
}


def migrate_comparison_dir(out_dir: Path) -> None:
    out_dir = out_dir.resolve()
    runs = out_dir / "runs"
    if runs.is_dir():
        for old, new in _RUN_DIR_RENAMES.items():
            src = runs / old
            dst = runs / new
            if src.is_dir() and not dst.exists():
                src.rename(dst)
                print(f"renamed runs/{old} -> runs/{new}")

    for name in ("comparison_progress.json", "metrics.json"):
        path = out_dir / name
        if not path.is_file():
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        changed = False
        for block_key in ("completed", "failed", "experiments"):
            block = doc.get(block_key)
            if not isinstance(block, dict):
                continue
            for old, new in _METRICS_KEY_RENAMES.items():
                if old in block:
                    block[new] = block.pop(old)
                    changed = True
        if changed:
            path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
            print(f"updated keys in {path.name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Migrate legacy SK/D2B names on disk")
    parser.add_argument("out_dir", type=Path, help="master_comparison timestamp directory")
    args = parser.parse_args(argv)
    out_dir = args.out_dir if args.out_dir.is_absolute() else _REPO / args.out_dir
    if not out_dir.is_dir():
        raise SystemExit(f"not a directory: {out_dir}")
    migrate_comparison_dir(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
