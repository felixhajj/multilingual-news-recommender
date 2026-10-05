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


def download_archive(receipt, destination, deadline):
    """Use bounded HTTP transfers; retain a partial file for power-loss recovery."""
    import requests
    from huggingface_hub import get_token, hf_hub_url
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and file_digest(destination) == receipt["archive_sha256"]:
        return destination
    partial = destination.with_suffix(destination.suffix + ".download")
    url = hf_hub_url(receipt["repo"], receipt["path"], repo_type="dataset", revision=receipt["revision"])
    failures = 0
    total_size = None
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError("Archive transfer deadline reached; partial file preserved")
        offset = partial.stat().st_size if partial.exists() else 0
        if total_size is not None and offset >= total_size:
            if offset != total_size or file_digest(partial) != receipt["archive_sha256"]:
                raise ValueError("Complete partial archive size or checksum mismatch")
            partial.replace(destination)
            return destination
        if offset and total_size is None and file_digest(partial) == receipt["archive_sha256"]:
            partial.replace(destination)
            return destination
        headers = {"Authorization": "Bearer " + get_token()} if receipt["private"] else {}
        end = offset + 8 * 1024**2 - 1
        if total_size is not None:
            end = min(end, total_size - 1)
        headers["Range"] = f"bytes={offset}-{end}"
        if offset % (128 * 1024**2) == 0:
            print(f"Downloading {receipt['version'][:16]}: {offset} bytes", flush=True)
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(30, 60)) as response:
                response.raise_for_status()
                if response.status_code != 206:
                    raise ValueError("Archive host did not honor bounded ranged download")
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {offset}-"):
                    raise ValueError("Remote range does not match preserved partial file")
                remote_size = int(content_range.rsplit("/", 1)[1])
                if total_size not in (None, remote_size) or offset >= remote_size:
                    raise ValueError("Remote size changed or partial file is oversized")
                total_size = remote_size
                expected_end = min(end, total_size - 1)
                if content_range != f"bytes {offset}-{expected_end}/{total_size}":
                    raise ValueError("Remote returned an unexpected byte interval")
                with partial.open("ab") as output:
                    for block in response.iter_content(64 * 1024):
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Archive transfer deadline reached; partial file preserved")
                        output.write(block)
                        if output.tell() > expected_end + 1:
                            raise ValueError("Transfer exceeded remote archive size; scratch download is invalid")
                if partial.stat().st_size != expected_end + 1:
                    raise requests.ConnectionError("Truncated ranged response")
            failures = 0
            if partial.stat().st_size == total_size:
                if file_digest(partial) != receipt["archive_sha256"]:
                    raise ValueError("Downloaded archive SHA-256 mismatch")
                partial.replace(destination)
                return destination
        except requests.RequestException as exc:
            print(f"Transfer interrupted ({type(exc).__name__}); partial bytes preserved", flush=True)
            failures += 1
            if failures == 5:
                raise
            time.sleep(min(2 ** failures, 15))


def verify_uploads():
    from huggingface_hub import HfApi
    deadline = time.monotonic() + 3600
    folder = ROOT / ".tmp" / "phase5-remote-download"
    for profile in ("runtime", "evidence"):
        receipt_path = ARTIFACTS / "phase5" / f"{profile}_upload.json"
        while not receipt_path.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Upload receipt did not arrive: {profile}")
            time.sleep(15)
        receipt = read_json(receipt_path)
        if receipt.get("status") == "download_checksum_and_restore_verified":
            print(f"Retained prior pinned {profile} verification", flush=True)
            continue
        info = HfApi().repo_info(receipt["repo"], repo_type="dataset", revision=receipt["revision"])
        if bool(info.private) != receipt["private"]:
            raise ValueError("Remote privacy does not match the upload receipt")
        archive = download_archive(receipt, folder / profile / receipt["version"] / Path(receipt["path"]).name,
                                   deadline)
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
            replay = verify(archive, ROOT / ".tmp" / ("phase5-remote-restore-" + receipt["version"][:16]))
        else:
            # Restore representative recovery artifacts, not only the ZIP directory.
            representative = next((name for name in checkpoints if name.endswith("adapter_model.safetensors")), None)
            if not representative:
                raise ValueError("Evidence has no resumable adapter checkpoint")
            restored = ROOT / ".tmp" / "phase5-evidence-sample"
            checkpoint_root = representative.split("/adapter/", 1)[0] + "/"
            sample_names = [name for name in checkpoints if name.startswith(checkpoint_root)]
            sample_names += ["output/portfolio/review/extraction.jsonl",
                             "output/portfolio/evaluation/extraction-only-v3/predictions.jsonl"]
            if checkpoint_root + "state.pt" not in sample_names:
                raise ValueError("Representative checkpoint is missing optimizer/resume state")
            with zipfile.ZipFile(archive) as bundle:
                for name in sample_names:
                    target = restored / name
                    if not target.resolve().is_relative_to(restored.resolve()) or "\\" in name or ":" in name:
                        raise ValueError("Unsafe representative evidence path")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(name) as source, target.open("wb") as output:
                        import shutil
                        shutil.copyfileobj(source, output)
                    if file_digest(target) != manifest["files"][name]:
                        raise ValueError("Representative evidence restore changed bytes")
            from safetensors import safe_open
            with safe_open(restored / representative, framework="pt", device="cpu") as weights:
                tensor_count = len(list(weights.keys()))
            replay = {"checkpoint_members_verified": len(checkpoints), "all_members_verified": len(manifest["files"]),
                      "representative_checkpoint": representative, "restored_adapter_tensors": tensor_count,
                      "checkpoint_resume_state_restored": True,
                      "review_and_prediction_files_restored": True}
        receipt.update(status="download_checksum_and_restore_verified", verification=replay)
        write_json(receipt_path, receipt)
        print(f"Verified pinned {profile} backup: {receipt['revision']}", flush=True)


def main():
    from filelock import FileLock
    with FileLock(ARTIFACTS / "phase5/.remote-verification.lock").acquire(timeout=0):
        verify_uploads()


if __name__ == "__main__":
    main()
