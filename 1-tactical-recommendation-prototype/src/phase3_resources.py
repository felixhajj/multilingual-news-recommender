"""Conservative disk headroom checks; never delete artifacts to make a job fit."""
import shutil

MIN_FREE_BYTES = 2 * 1024**3
ARTIFACT_LIMIT_BYTES = 5 * 1024**3
CHECKPOINT_HEADROOM_BYTES = 256 * 1024**2


def check_storage(directory):
    free = shutil.disk_usage(directory).free
    size = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
    if free < MIN_FREE_BYTES:
        raise RuntimeError("Storage guard: less than 2 GiB free on the artifact drive; existing checkpoints retained")
    if size + CHECKPOINT_HEADROOM_BYTES >= ARTIFACT_LIMIT_BYTES:
        raise RuntimeError("Storage guard: insufficient headroom within the 5 GiB artifact limit; no files deleted")
    return {"free_bytes": free, "portfolio_bytes": size,
            "minimum_free_bytes": MIN_FREE_BYTES, "artifact_limit_bytes": ARTIFACT_LIMIT_BYTES}
