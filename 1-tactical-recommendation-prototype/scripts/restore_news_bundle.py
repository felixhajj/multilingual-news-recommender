"""Verify and restore a portable bundle; reject path traversal and changed checksums."""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.portfolio_config import ROOT
from src.portfolio_config import digest


def restore(archive, destination=ROOT):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as bundle:
        manifest = json.loads(bundle.read("bundle_manifest.json"))
        entries = [info.filename for info in bundle.infolist()]
        if len(entries) != len(set(entries)) or set(entries) != set(manifest["files"]) | {"bundle_manifest.json"}:
            raise ValueError("Unexpected or duplicate bundle entries")
        if sum(info.file_size for info in bundle.infolist()) > 5 * 1024**3:
            raise ValueError("Oversized artifact archive")
        for name, expected in manifest["files"].items():
            target = (destination / name).resolve()
            if not target.is_relative_to(destination) or Path(name).is_absolute() or "\\" in name or ":" in name:
                raise ValueError("Unsafe bundle path")
            actual = hashlib.sha256(bundle.read(name)).hexdigest()
            if actual != expected:
                raise ValueError(f"Artifact checksum mismatch: {name}")
        if "version" in manifest and digest({k: v for k, v in manifest.items() if k != "version"}) != manifest["version"]:
            raise ValueError("Bundle manifest version mismatch")
        # Verify every file before changing any local artifact.
        for name in manifest["files"]:
            target = destination / name
            if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != manifest["files"][name]:
                raise ValueError(f"Refusing to overwrite a different local artifact: {target}")
        for name in manifest["files"]:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bundle.read(name))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--destination", type=Path, default=ROOT)
    args = parser.parse_args()
    print(restore(args.archive, args.destination)["kind"])
