"""Split a complete archive for bounded cloud uploads, with reassembly checksums."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.portfolio_config import file_digest, write_json


def split(archive, part_bytes=90 * 1024**2):
    archive = Path(archive)
    parts = []
    with archive.open("rb") as source:
        index = 1
        while True:
            data = source.read(part_bytes)
            if not data:
                break
            path = archive.with_name(archive.name + f".part{index:03d}")
            with path.open("wb") as output:
                output.write(data)
            parts.append({"name": path.name, "bytes": len(data), "sha256": file_digest(path)})
            index += 1
    manifest = {"archive": archive.name, "archive_sha256": file_digest(archive), "parts": parts,
                "restore": "Concatenate parts in listed order as binary bytes; verify archive SHA-256 before restoring."}
    write_json(archive.with_name(archive.name + ".parts.json"), manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    print(split(parser.parse_args().archive))
