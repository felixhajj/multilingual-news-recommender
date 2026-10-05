"""Small, attributed news corpus reconstructed from the published Mewsli descriptors.

This is a project split and reconstruction, not an official Mewsli benchmark run.
Only uniquely aligned hyperlink mentions are accepted as linking supervision.
"""
import bz2
import csv
import hashlib
import io
import re
import time
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import requests

from src.portfolio_config import ARTIFACTS, digest, file_digest, write_json, write_jsonl

DESCRIPTORS = "https://storage.googleapis.com/gresearch/mewsli/mewsli-9.zip"
NEWS_TERMS = re.compile(
    r"saudi|arabia|lebanon|israel|palestin|iran|iraq|syria|egypt|qatar|yemen|"
    r"jordan|libya|tunisia|morocco|algeria|defen[sc]e|military|missile|drone|"
    r"diplomac|election|sanction|parliament|president|minister|oil|energy|"
    r"\u0627\u0644\u0633\u0639\u0648\u062f|\u0644\u0628\u0646\u0627\u0646|"
    r"\u0633\u0648\u0631\u064a\u0627|\u0641\u0644\u0633\u0637\u064a\u0646|"
    r"\u0625\u064a\u0631\u0627\u0646|\u0645\u0635\u0631|\u0627\u0644\u0639\u0631\u0627\u0642|"
    r"\u0631\u0626\u064a\u0633|\u0648\u0632\u064a\u0631|\u0639\u0633\u0643\u0631|\u0627\u0646\u062a\u062e\u0627\u0628",
    re.I,
)


def download(url, path, max_bytes=250 * 1024**2):
    path = Path(path)
    if path.exists() and path.stat().st_size:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=(20, 60), headers={
                "User-Agent": "MultilingualNewsRecommender/1.0 (public academic corpus reconstruction)"
            }) as response:
                response.raise_for_status()
                total = 0
                with part.open("wb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        total += len(chunk)
                        if total > max_bytes:
                            raise ValueError("Download exceeds its explicit storage allowance")
                        handle.write(chunk)
            part.replace(path)
            return path
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def descriptors(archive, language):
    with zipfile.ZipFile(archive) as bundle:
        def table(name):
            member = next(p for p in bundle.namelist() if p.endswith(f"{language}/{name}.tsv"))
            return list(csv.DictReader(io.StringIO(bundle.read(member).decode("utf-8")), delimiter="\t"))
        docs = {row["curid"]: row for row in table("docs")}
        mentions = defaultdict(list)
        for row in table("mentions"):
            mentions[row["docid"]].append(row)
    return docs, mentions


def clean_wikitext(text):
    import mwparserfromhell
    # Sources and navigation are not article prose and must not become model labels.
    text = re.split(r"(?im)^==+\s*(?:sources|related news|external links|references|"
                    r"\u0645\u0635\u0627\u062f\u0631|\u0627\u0644\u0645\u0635\u0627\u062f\u0631)\s*==+", text)[0]
    text = re.sub(r"\[\[(?:Category|File|Image|\u062a\u0635\u0646\u064a\u0641|\u0645\u0644\u0641):[^\]]*\]\]", "", text, flags=re.I)
    value = mwparserfromhell.parse(text).strip_code(normalize=True, collapse=True)
    return "\n".join(line.strip() for line in value.splitlines() if line.strip())


def news_date(wikitext, revision_timestamp):
    match = re.search(r"\{\{date\|([^}]+)", wikitext, re.I)
    if match:
        for fmt in ("%B %d, %Y", "%d %B %Y", "%Y-%m-%d", "%B %d %Y"):
            try:
                return datetime.strptime(match[1].strip(), fmt).date().isoformat(), "publication_date"
            except ValueError:
                pass
    return revision_timestamp[:10], "revision_date_publication_unknown"


def reconstruct(dump_path, docs, annotations, language):
    with bz2.open(dump_path, "rb") as handle:
        for _, page in ET.iterparse(handle, events=("end",)):
            if page.tag.rsplit("}", 1)[-1] != "page":
                continue
            ns = page.tag.split("}")[0] + "}"
            page_id = page.findtext(ns + "id")
            descriptor = docs.get(page_id)
            if descriptor:
                revision = page.find(ns + "revision")
                raw = revision.findtext(ns + "text") or ""
                # A descriptor belongs to one revision. Do not silently use later text.
                if revision.findtext(ns + "id") != descriptor["revid"]:
                    page.clear()
                    continue
                body = clean_wikitext(raw)
                if len(body) < 200:
                    page.clear()
                    continue
                mentions = []
                seen = set()
                for mention in annotations[descriptor["docid"]]:
                    surface = mention["mention"]
                    key = (surface, mention["qid"])
                    if key in seen or not surface or body.count(surface) != 1:
                        continue
                    seen.add(key)
                    start = body.index(surface)
                    mentions.append({"text": surface, "qid": mention["qid"], "start": start,
                                     "end": start + len(surface), "label_source": "published_hyperlink",
                                     "alignment": "unique_surface", "source_link": mention["url"]})
                date, date_kind = news_date(raw, revision.findtext(ns + "timestamp") or "")
                yield {"article_id": descriptor["docid"], "title": descriptor["title"],
                       "body": body, "summary": "", "language": language, "date": date,
                       "date_kind": date_kind, "source_url": descriptor["url"],
                       "source_id": "wikinews_mewsli9", "author": "Wikinews contributors",
                       "license_id": "CC-BY-2.5", "license_url": "https://creativecommons.org/licenses/by/2.5/",
                       "source_revision": descriptor["revid"], "synthetic": False,
                       "annotation_completeness": "partial_hyperlinks_not_exhaustive_entities",
                       "mentions": mentions, "domain_hits": len(NEWS_TERMS.findall(body)),
                       "content_hash": digest(body)}
            page.clear()


def simhash(text):
    words = re.findall(r"\w+", text.casefold())
    grams = {" ".join(words[i:i+3]) for i in range(max(1, len(words)-2))}
    accum = [0] * 64
    for gram in grams:
        number = int.from_bytes(hashlib.blake2b(gram.encode(), digest_size=8).digest(), "big")
        for bit in range(64):
            accum[bit] += 1 if number & (1 << bit) else -1
    return sum(1 << bit for bit, score in enumerate(accum) if score >= 0)


def assign_groups(records):
    parents = list(range(len(records)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    def union(a, b):
        parents[find(b)] = find(a)
    bands, signatures, related = defaultdict(list), [], {}
    for i, record in enumerate(records):
        signature = simhash(record["body"])
        signatures.append(signature)
        candidates = set()
        for band in range(4):
            key = (band, (signature >> (16 * band)) & 65535)
            candidates.update(bands[key])
            bands[key].append(i)
        for j in candidates:
            if (signature ^ signatures[j]).bit_count() <= 3:
                union(i, j)
        qids = tuple(sorted({m["qid"] for m in record["mentions"]}))
        if len(qids) >= 3:
            key = (record["date"], qids)
            if key in related:
                union(i, related[key])
            related[key] = i
    groups = defaultdict(list)
    for i, record in enumerate(records):
        groups[find(i)].append(record["article_id"])
    for i, record in enumerate(records):
        group = digest(sorted(groups[find(i)]))
        bucket = int(group[:8], 16) % 100
        record.update(group_id=group, split="train" if bucket < 80 else "validation" if bucket < 90 else "test")
    return records


def build_corpus(target=5000):
    downloads = ARTIFACTS / "downloads"
    archive = download(DESCRIPTORS, downloads / "mewsli-9.zip")
    records, inputs = [], {"descriptors_sha256": file_digest(archive)}
    for language in ("ar", "en"):
        filename = f"{language}wikinews-20190101-pages-articles.xml.bz2"
        url = f"https://archive.org/download/{language}wikinews-20190101/{filename}"
        print(f"Downloading/reconstructing {language}", flush=True)
        dump = download(url, downloads / filename)
        inputs[language + "_dump_sha256"] = file_digest(dump)
        docs, mentions = descriptors(archive, language)
        records.extend(reconstruct(dump, docs, mentions, language))
    records.sort(key=lambda r: (r["language"] == "ar", r["domain_hits"], r["article_id"]), reverse=True)
    selected, hashes = [], set()
    for record in records:
        if record["content_hash"] in hashes:
            continue
        hashes.add(record["content_hash"])
        selected.append(record)
        if len(selected) == target:
            break
    assign_groups(selected)
    write_jsonl(ARTIFACTS / "corpus.jsonl", selected)
    manifest = {"version": digest([r["content_hash"] for r in selected]), "documents": len(selected),
                "reconstructed_documents": len(records), "languages": dict(Counter(r["language"] for r in selected)),
                "splits": dict(Counter(r["split"] for r in selected)),
                "domain_related_documents": sum(r["domain_hits"] > 0 for r in selected),
                "aligned_mentions": sum(len(r["mentions"]) for r in selected), "inputs": inputs,
                "source": DESCRIPTORS, "split_policy": "project_grouped_80_10_10",
                "limitations": ["Historical archive, not current news", "Hyperlink labels are incomplete",
                                "Custom text reconstruction, not the official Mewsli benchmark",
                                "Translation grouping is conservative; residual paraphrase overlap is possible"]}
    write_json(ARTIFACTS / "corpus_manifest.json", manifest)
    print(manifest, flush=True)
    return selected
