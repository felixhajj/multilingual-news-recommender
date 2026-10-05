import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.checkpoint_loading import checkpoint_loading


class CheckpointLoadingTests(unittest.TestCase):
    def test_buffered_loader_is_scoped_and_restored_after_error(self):
        calls = []
        def original(*args, **kwargs):
            calls.append(kwargs)
        modeling = SimpleNamespace(safe_open=original)
        with patch.dict(os.environ, {"NEWS_SAFETENSORS_BACKEND": "pread"}), \
             patch.dict(sys.modules, {"transformers": SimpleNamespace(modeling_utils=modeling)}):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                with checkpoint_loading():
                    modeling.safe_open("fixture", framework="pt")
                    raise RuntimeError("interrupted")
            self.assertIs(modeling.safe_open, original)
        self.assertEqual(calls, [{"framework": "pt", "backend": "pread"}])

    def test_unsupported_backend_refused(self):
        with patch.dict(os.environ, {"NEWS_SAFETENSORS_BACKEND": "unsafe"}):
            with self.assertRaises(ValueError):
                with checkpoint_loading():
                    self.fail("Unsupported backend entered")


if __name__ == "__main__":
    unittest.main()
