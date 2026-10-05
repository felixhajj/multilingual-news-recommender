import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.review_pipeline import promote_reviewed_examples  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Append analyst-approved article extractions to the LoRA training split."
    )
    parser.add_argument(
        "queue",
        type=Path,
        help="JSONL review queue with corrected labels and review metadata.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data" / "extraction_examples.jsonl",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    promoted, skipped = promote_reviewed_examples(args.queue, args.dataset)
    print(
        json.dumps(
            {
                "promoted": len(promoted),
                "skipped": skipped,
                "dataset": str(args.dataset),
                "next_step": "Rerun the extraction QLoRA notebook for the next versioned adapter.",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
