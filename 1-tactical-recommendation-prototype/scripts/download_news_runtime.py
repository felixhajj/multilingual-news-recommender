"""Download a version-pinned, checksummed runtime using only the Python standard library."""
import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_DESCRIPTOR = "https://huggingface.co/datasets/felixhajj/multilingual-news-runtime/resolve/main/release.json"


def download(descriptor_url, destination):
    if not descriptor_url.startswith("https://huggingface.co/datasets/"):
        raise ValueError("Use an HTTPS Hugging Face dataset release descriptor")
    with urllib.request.urlopen(descriptor_url, timeout=60) as response:
        descriptor = json.load(response)
    if (descriptor.get("private") is not False or not re.fullmatch(r"[0-9a-f]{40}", descriptor["revision"])
            or not re.fullmatch(r"[0-9a-f]{64}", descriptor["version"])):
        raise ValueError("The runtime must be public and pinned to a commit")
    if not re.fullmatch(r"[0-9a-f]{64}", descriptor["archive_sha256"]):
        raise ValueError("Missing archive checksum")
    expected_path = descriptor["version"] + "/runtime-" + descriptor["version"][:16] + ".zip"
    if descriptor["path"] != expected_path or not re.fullmatch(r"[\w-]+/[\w-]+", descriptor["repo"]):
        raise ValueError("Unexpected runtime archive identity")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / Path(expected_path).name
    partial = archive.with_suffix(".zip.download")
    url = (f"https://huggingface.co/datasets/{descriptor['repo']}/resolve/"
           f"{descriptor['revision']}/{expected_path}")
    def checksum(path):
        sha = hashlib.sha256()
        with path.open('rb') as source:
            for block in iter(lambda: source.read(1024**2), b''):
                sha.update(block)
        return sha.hexdigest()

    if archive.exists() and checksum(archive) == descriptor['archive_sha256']:
        print(json.dumps({'archive': str(archive), 'revision': descriptor['revision'],
                          'sha256': descriptor['archive_sha256']}))
        return archive
    deadline, failures, total = time.monotonic() + 3600, 0, None
    while True:
        offset = partial.stat().st_size if partial.exists() else 0
        if total is not None and offset >= total:
            if offset != total:
                raise ValueError('Oversized partial runtime download')
            break
        if total is None and offset and checksum(partial) == descriptor['archive_sha256']:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError('Download deadline reached; partial bytes retained')
        end = offset + 8 * 1024**2 - 1
        if total is not None:
            end = min(end, total - 1)
        request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-{end}'})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                value = response.headers.get('Content-Range', '')
                match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', value)
                if response.status != 206 or not match:
                    raise ValueError('Runtime host did not honor bounded byte ranges')
                start, last, size = map(int, match.groups())
                if start != offset or last != min(end, size-1) or total not in (None, size):
                    raise ValueError('Runtime byte interval or size changed')
                total = size
                with partial.open('ab') as output:
                    for block in iter(lambda: response.read(64*1024), b''):
                        if time.monotonic() >= deadline:
                            raise TimeoutError('Download deadline reached; partial bytes retained')
                        output.write(block)
                        if output.tell() > last+1:
                            raise ValueError('Runtime response exceeded its interval')
                if partial.stat().st_size != last+1:
                    raise urllib.error.URLError('Truncated runtime interval')
            failures = 0
        except (urllib.error.URLError, OSError):
            failures += 1
            if failures >= 5:
                raise
            time.sleep(min(2**failures, 15))
    if checksum(partial) != descriptor["archive_sha256"]:
        raise ValueError("Runtime download checksum failed; not restored")
    partial.replace(archive)
    print(json.dumps({"archive": str(archive), "revision": descriptor["revision"],
                      "sha256": descriptor['archive_sha256']}))
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor-url", default=DEFAULT_DESCRIPTOR)
    parser.add_argument("--destination", type=Path, default=Path("downloads"))
    args = parser.parse_args()
    download(args.descriptor_url, args.destination)
