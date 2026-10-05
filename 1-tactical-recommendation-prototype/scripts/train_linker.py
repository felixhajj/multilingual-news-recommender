import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.portfolio_config import ARTIFACTS, read_jsonl
from src.learned_linker import train_linker

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate contextual entity linking on real published links")
    parser.add_argument("--train-mentions", type=int, default=2000)
    parser.add_argument("--eval-mentions", type=int, default=300)
    args = parser.parse_args()
    train_linker(read_jsonl(ARTIFACTS / "corpus.jsonl"), args.train_mentions, args.eval_mentions)
