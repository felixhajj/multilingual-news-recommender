"""Enable GitHub Actions Pages using the existing Git credential manager."""
import json
import subprocess
import urllib.error
import urllib.request


def main():
    result = subprocess.run(["git", "-c", "credential.interactive=false", "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n", text=True, capture_output=True, check=True, timeout=20)
    credentials = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    headers = {"Authorization": "Bearer " + credentials["password"],
               "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    url = "https://api.github.com/repos/felixhajj/multilingual-news-recommender/pages"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as response:
            existing = json.load(response)
        method = "PUT" if existing.get("build_type") != "workflow" else None
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        method = "POST"
    if method:
        request = urllib.request.Request(url, data=json.dumps({"build_type": "workflow"}).encode(),
                                         headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                existing = json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Pages configuration failed ({exc.code}): {exc.read().decode()}") from None
    print({"url": existing.get("html_url"), "build_type": existing.get("build_type"),
           "status": "configured_workflow_execution_pending"})


if __name__ == "__main__":
    main()
