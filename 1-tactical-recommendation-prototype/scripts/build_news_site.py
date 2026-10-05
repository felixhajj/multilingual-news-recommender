"""Build a model-free static visitor site from genuine exported artifacts."""
import sys
import shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.portfolio_config import ROOT


def main():
    from markdown_it import MarkdownIt
    site = ROOT.parent / "docs"
    site.mkdir(exist_ok=True)
    base = "https://github.com/felixhajj/multilingual-news-recommender/blob/main/1-tactical-recommendation-prototype/"
    for source, output in (("START_HERE.md", "guide.html"), ("DATA_AND_MODEL_CARD.md", "model-card.html"), ("docs/PORTFOLIO_RESULTS.md", "results.html")):
        rendered = MarkdownIt("commonmark").enable("table").render((ROOT / source).read_text(encoding="utf-8"))
        from bs4 import BeautifulSoup
        tree = BeautifulSoup(rendered, "html.parser")
        for link in tree.find_all("a", href=True):
            if link["href"] in {"walkthrough.pdf", "guide.html", "results.html", "model-card.html", "evidence.json"}:
                continue
            if not link["href"].startswith(("http:", "https:", "#", "mailto:")):
                link["href"] = base + link["href"]
        page = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>News Recommender / Guide</title><link rel="stylesheet" href="site.css"></head><body>'
                '<nav aria-label="Main navigation"><a class="brand" href="index.html">BETWEEN LANGUAGES</a><div><a href="results.html">Results</a><a href="walkthrough.pdf">Walkthrough PDF</a><a href="https://huggingface.co/spaces/felixhajj/multilingual-news-recommender">Try the demo</a></div></nav>'
                f'<main class="document">{tree}</main></body></html>')
        (site / output).write_text(page, encoding="utf-8")
    for name in ("example_ranking.json", "example_analyses.json", "evidence.json", "deployment.json"):
        source = ROOT / "data" / "release" / name
        if source.exists():
            shutil.copyfile(source, site / name)
    print(site)


if __name__ == "__main__":
    main()
