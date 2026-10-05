"""Upload immutable release/evidence artifacts and record pinned Hub revisions."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.portfolio_config import ARTIFACTS, file_digest, read_json, write_json


def upload(archive, repo, private):
    from huggingface_hub import HfApi
    archive = Path(archive)
    manifest_path = archive.with_suffix(".manifest.json")
    manifest = read_json(manifest_path)
    if not manifest or file_digest(archive) != manifest["archive_sha256"]:
        raise ValueError("Archive does not match its checksummed manifest")
    api = HfApi()
    api.create_repo(repo_id=repo, repo_type="dataset", private=private, exist_ok=True)
    info = api.repo_info(repo, repo_type="dataset")
    if bool(info.private) != private:
        raise ValueError("Repository visibility does not match the requested archive privacy")
    path = manifest["version"] + "/" + archive.name
    result = api.upload_file(path_or_fileobj=archive, path_in_repo=path,
        repo_id=repo, repo_type="dataset", commit_message=f"Archive {manifest['profile']} {manifest['version'][:16]}")
    result = api.upload_file(path_or_fileobj=manifest_path,
        path_in_repo=manifest["version"] + "/" + manifest_path.name,
        repo_id=repo, repo_type="dataset", commit_message="Record SHA-256 artifact manifest")
    receipt = {"status": "uploaded_download_verification_pending", "repo": repo,
        "private": private, "revision": result.oid, "path": path,
        "archive_sha256": manifest["archive_sha256"], "version": manifest["version"]}
    write_json(ARTIFACTS / "phase5" / f"{manifest['profile']}_upload.json", receipt)
    print(receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    upload(args.archive, args.repo, args.private)
