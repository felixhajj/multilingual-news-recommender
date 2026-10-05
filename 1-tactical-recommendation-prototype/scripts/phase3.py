"""Explicit Phase 3 validation and final-test operations."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.finish_phase2 import verify_completion
from src.news_selection import lock_selection, score_extraction, verify_selection_lock


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["validation", "test", "lock", "check-lock"])
    parser.add_argument("--runs", nargs="+", default=["base", "extraction-only-v1", "domain-extraction-v1"])
    parser.add_argument("--base", default="base-v3")
    args = parser.parse_args()
    verify_completion()
    if args.operation in {"validation", "test"}:
        output = {run: score_extraction(run, "test" if args.operation == "test" else "validation") for run in args.runs}
    elif args.operation == "lock":
        output = lock_selection(args.base, args.runs)
    else:
        output = verify_selection_lock()
    print(json.dumps(output, ensure_ascii=False, indent=2))
