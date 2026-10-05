"""Opt-in buffered safetensors loading for the Windows mmap crash boundary."""
import os
from contextlib import contextmanager
from functools import partial
from threading import RLock

_LOADER_LOCK = RLock()


@contextmanager
def checkpoint_loading():
    backend = os.getenv("NEWS_SAFETENSORS_BACKEND", "mmap")
    if backend not in {"mmap", "pread"}:
        raise ValueError("Unsupported safetensors storage backend")
    if backend == "mmap":
        yield
        return
    from transformers import modeling_utils
    # The installed Transformers predates safetensors' backend argument. Scope
    # its adapter to model loading only; restore the dependency on every exit.
    with _LOADER_LOCK:
        original = modeling_utils.safe_open
        modeling_utils.safe_open = partial(original, backend="pread")
        try:
            yield
        finally:
            modeling_utils.safe_open = original
