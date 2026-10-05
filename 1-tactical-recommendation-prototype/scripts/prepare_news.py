"""Download and reconstruct the attributed English/Arabic release corpus."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.news_corpus import build_corpus

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=5000)
    args = parser.parse_args()
    if not 1 <= args.target <= 15000:
        parser.error("target must be between 1 and 15000")
    build_corpus(args.target)
