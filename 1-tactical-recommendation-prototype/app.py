"""Local entrypoint for the shared, live multilingual news pipeline."""
import sys

from portfolio_app import launch


if __name__ == "__main__":
    launch(int(sys.argv[1]) if len(sys.argv) > 1 else None)
