"""Portable paths and explicit identities for the portfolio release."""
import hashlib
import json
import os
from pathlib import Path
from filelock import FileLock

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get("NEWS_ARTIFACTS", ROOT / "output" / "portfolio"))
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
BASE_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
EMBEDDING_REVISION = "d128750597153bb5987e10b1c3493a34e5a4502a"
LEGACY_ADAPTER = ROOT / "output" / "entity_extraction_qwen25_3b" / "adapter"
PIPELINE_VERSION = "news-v1.0"
GPU_LOCK = FileLock(ARTIFACTS / ".gpu.lock")


def configure_cache():
    # Reuse the original cache on this PC; all other machines use HF's defaults.
    if "HF_HOME" not in os.environ and Path("D:/hf-cache/hub").is_dir():
        os.environ["HF_HOME"] = "D:/hf-cache"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)
