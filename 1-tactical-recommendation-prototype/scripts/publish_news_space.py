"""Publish a gated release only to free ZeroGPU; never provision paid fallback hardware."""
import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.portfolio_config import ROOT, read_json, write_json
from scripts.package_news_release import package
from scripts.restore_news_bundle import restore


def main(repo):
    from huggingface_hub import HfApi, get_token, SpaceHardware
    if not get_token():
        raise RuntimeError("Authenticate on this machine with hf auth login. Do not paste credentials into chat or source files.")
    archive = package(preview=False)
    api = HfApi()
    api.whoami()
    temporary_root = ROOT / ".tmp"
    temporary_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
        folder = Path(temporary)
        restore(archive, folder)
        shutil.copytree(ROOT / "src", folder / "src", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(ROOT / "data" / "release", folder / "data" / "release", dirs_exist_ok=True)
        shutil.copyfile(ROOT / "portfolio_app.py", folder / "portfolio_app.py")
        shutil.copyfile(ROOT / "deployment" / "space_app.py", folder / "app.py")
        shutil.copyfile(ROOT / "deployment" / "requirements.txt", folder / "requirements.txt")
        (folder / "README.md").write_text("---\ntitle: Multilingual News Recommender\nemoji: \U0001f310\ncolorFrom: green\ncolorTo: blue\nsdk: gradio\nsdk_version: 5.49.1\npython_version: 3.12\napp_file: app.py\n---\n\nBuilt with Qwen. Research/evaluation only. Qwen Research License applies; see licenses/.\n\nArticle text is processed live, not saved for training. Free GPU quotas and cold starts apply.\n", encoding="utf-8")
        api.create_repo(repo_id=repo, repo_type="space", space_sdk="gradio", space_hardware=SpaceHardware.ZERO_A10G, exist_ok=True)
        api.request_space_hardware(repo, hardware=SpaceHardware.ZERO_A10G)
        api.upload_folder(repo_id=repo, repo_type="space", folder_path=folder, commit_message="Publish validated multilingual news research release")
    deployment_path = ROOT / "data" / "release" / "deployment.json"
    deployment = read_json(deployment_path, {})
    deployment.update(live_demo="https://huggingface.co/spaces/" + repo, status="uploaded_hosted_verification_pending")
    write_json(deployment_path, deployment)
    print(deployment)
    print("Run the hosted UI checks, including a new article, before marking the deployment verified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    main(parser.parse_args().repo)
