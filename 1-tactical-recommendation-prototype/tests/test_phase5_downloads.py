import hashlib
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

from scripts.verify_uploaded_artifacts import download_archive


class Response:
    def __init__(self, start, content, total, broken=False):
        self.status_code = 206
        self.headers = {'Content-Range': f'bytes {start}-{total-1}/{total}'}
        self.content = content
        self.broken = broken

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, _size):
        yield self.content
        if self.broken:
            raise requests.ConnectionError('Simulated interruption')


class RangedDownloadTests(unittest.TestCase):
    def test_resume_preserves_bytes_and_verifies_complete_archive(self):
        content = b'abcdefghijklmno'
        receipt = {'repo': 'user/data', 'path': 'version/file.zip', 'revision': 'a'*40,
                   'private': False, 'version': 'b'*64,
                   'archive_sha256': hashlib.sha256(content).hexdigest()}
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / 'archive.zip'
            archive.with_suffix('.zip.download').write_bytes(content[:5])
            with patch('requests.get', return_value=Response(5, content[5:], len(content))) as get:
                self.assertEqual(download_archive(receipt, archive, time.monotonic()+10), archive)
                self.assertEqual(get.call_args.kwargs['headers']['Range'], 'bytes=5-8388612')
            self.assertEqual(archive.read_bytes(), content)

    def test_interrupted_chunk_resumes_without_duplicate_bytes(self):
        content = b'abcdefghijklmno'
        receipt = {'repo': 'user/data', 'path': 'version/file.zip', 'revision': 'a'*40,
                   'private': False, 'version': 'b'*64,
                   'archive_sha256': hashlib.sha256(content).hexdigest()}
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / 'archive.zip'
            responses = [Response(0, content[:5], len(content), broken=True),
                         Response(5, content[5:], len(content))]
            with patch('requests.get', side_effect=responses), patch('time.sleep'):
                download_archive(receipt, archive, time.monotonic()+10)
            self.assertEqual(archive.read_bytes(), content)

    def test_range_mismatch_is_rejected(self):
        receipt = {'repo': 'user/data', 'path': 'version/file.zip', 'revision': 'a'*40,
                   'private': False, 'version': 'b'*64, 'archive_sha256': 'c'*64}
        with tempfile.TemporaryDirectory() as folder:
            with patch('requests.get', return_value=Response(5, b'bad', 8)):
                with self.assertRaisesRegex(ValueError, 'range does not match'):
                    download_archive(receipt, Path(folder)/'archive.zip', time.monotonic()+10)

    def test_disconnect_after_final_payload_promotes_verified_partial(self):
        content = b'abcdefghijklmno'
        receipt = {'repo': 'user/data', 'path': 'version/file.zip', 'revision': 'a'*40,
                   'private': False, 'version': 'b'*64,
                   'archive_sha256': hashlib.sha256(content).hexdigest()}
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder)/'archive.zip'
            with patch('requests.get', return_value=Response(0, content, len(content), broken=True)) as get:
                with patch('time.sleep'):
                    download_archive(receipt, archive, time.monotonic()+10)
                self.assertEqual(get.call_count, 1)
            self.assertEqual(archive.read_bytes(), content)
