"""Download pinned cloud archives, verify their bytes, and replay runtime evidence."""
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.portfolio_config import ARTIFACTS, ROOT, file_digest, read_json, write_json
from scripts.verify_runtime_restore import verify


def main():
    from huggingface_hub import hf_hub_download, HfApi
    deadline = time.monotonic() + 3600
    folder = ROOT / ".tmp" / "phase5-remote-download"
    for profile in ("runtime", "evidence"):
        receipt_path = ARTIFACTS / "phase5" / f"{profile}_upload.json"
        while not receipt_path.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Upload receipt did not arrive: {profile}")
            time.sleep(15)
        receipt = read_json(receipt_path)
        info = HfApi().repo_info(receipt["repo"], repo_type="dataset", revision=receipt["revision"])
        if bool(info.private) != receipt["private"]:
            raise ValueError("Remote privacy does not match the upload receipt")
        archive = Path(hf_hub_download(receipt["repo"], receipt["path"], repo_type="dataset",
            revision=receipt["revision"], local_dir=folder / profile, force_download=True))
        if file_digest(archive) != receipt["archive_sha256"]:
            raise ValueError("Downloaded archive SHA-256 mismatch")
        with zipfile.ZipFile(archive) as bundle:
            manifest = json.loads(bundle.read("bundle_manifest.json"))
            for name, expected in manifest["files"].items():
                checksum = hashlib.sha256()
                with bundle.open(name) as source:
                    for block in iter(lambda: source.read(1024**2), b""):
                        checksum.update(block)
                if checksum.hexdigest() != expected:
                    raise ValueError(f"Downloaded member checksum mismatch: {name}")
            if profile == "evidence":
                checkpoints = [name for name in manifest["files"] if "/checkpoints/" in name]
                if not checkpoints or "output/portfolio/corpus.jsonl" not in manifest["files"]:
                    raise ValueError("Evidence backup is missing checkpoints or the corpus")
        if profile == "runtime":
            replay = verify(archive, ROOT / ".tmp" / "phase5-remote-restore")
        else:
            replay = {"checkpoint_members_verified": len(checkpoints), "all_members_verified": len(manifest["files"])}
        receipt.update(status="download_checksum_and_restore_verified", verification=replay)
        write_json(receipt_path, receipt)
        print(f"Verified pinned {profile} backup: {receipt['revision']}", flush=True)


if __name__ == "__main__":
    main()
