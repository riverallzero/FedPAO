from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.orchestrators import list_orchestrators


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--third-party-dir", default="third_party")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    statuses = list_orchestrators(args.third_party_dir)
    if args.json:
        print(json.dumps(statuses, indent=2))
        return

    for status in statuses:
        marker = "OK" if status["available"] and not status["missing_paths"] else "MISSING"
        print(f"{status['name']:10s} {marker}")
        print(f"  repo: {status['repo_path']}")
        if status["missing_paths"]:
            for path in status["missing_paths"]:
                print(f"  missing: {path}")


if __name__ == "__main__":
    main()
