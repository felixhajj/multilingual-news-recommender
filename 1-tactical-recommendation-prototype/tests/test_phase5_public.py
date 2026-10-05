import unittest

from portfolio_app import clear_analysis, gpu_duration, begin_analysis, end_analysis, FILTER_FIELDS


class PublicInferenceTests(unittest.TestCase):
    def test_short_requests_do_not_reserve_the_entire_anonymous_quota(self):
        self.assertEqual(gpu_duration("Test", "France met Lebanon."), 30)
        self.assertLess(gpu_duration("Test", "a" * 3000), 120)

    def test_long_requests_are_bounded(self):
        self.assertEqual(gpu_duration("Test", "a" * 50000), 120)
        self.assertEqual(gpu_duration("Test", None), 30)

    def test_new_request_clears_old_json_links_and_explanation(self):
        html, extraction, links, explanation, trace = clear_analysis()
        self.assertIn("no saved answer is substituted", html)
        self.assertEqual(extraction, {})
        self.assertEqual(links, [])
        self.assertEqual(explanation, {})
        self.assertEqual(trace, {"status": "requested"})

    def test_form_locks_during_analysis_and_unlocks_after_the_chain(self):
        self.assertEqual(len(begin_analysis()), 5 + 4 + len(FILTER_FIELDS))
        self.assertTrue(all(row['interactive'] is False for row in begin_analysis()[5:]))
        self.assertTrue(all(row['interactive'] is True for row in end_analysis()))

    def test_unlock_runs_after_failure_not_only_after_success(self):
        from portfolio_app import build_app
        dependencies = build_app().config['dependencies']
        analyze = next(row for row in dependencies if row.get('api_name') == 'analyze')
        unlock = next(row for row in dependencies if row.get('trigger_after') == analyze['id'])
        self.assertFalse(unlock['trigger_only_on_success'])

    def test_evidence_interpreter_keeps_virtual_environment_symlink_path(self):
        from pathlib import Path
        from unittest.mock import patch
        from scripts.verify_runtime_restore import interpreter_path
        with patch.object(Path, 'resolve', side_effect=AssertionError('Do not dereference venv symlinks')):
            result = Path(interpreter_path('.venv-evidence/bin/python'))
        self.assertTrue(result.is_absolute())
        self.assertEqual(result.parts[-3:], ('.venv-evidence', 'bin', 'python'))
