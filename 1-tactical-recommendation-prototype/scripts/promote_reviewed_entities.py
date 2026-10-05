import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.entity_catalogue_pipeline import promote_reviewed_entities  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge reviewed Wikidata entity candidates and aliases into the catalogue."
    )
    parser.add_argument("candidates", type=Path)
    parser.add_argument(
        "--catalogue",
        type=Path,
        default=ROOT / "data" / "entities.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    promoted, skipped = promote_reviewed_entities(args.candidates, args.catalogue)
    print(
        json.dumps(
            {
                "promoted": len(promoted),
                "skipped": skipped,
                "catalogue": str(args.catalogue),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
