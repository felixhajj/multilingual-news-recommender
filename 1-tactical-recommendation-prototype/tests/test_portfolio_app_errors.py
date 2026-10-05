import unittest
from unittest.mock import patch

import portfolio_app


class NoOpProgress:
    def __call__(self, *_args, **_kwargs):
        pass


class PortfolioAppFailureTests(unittest.TestCase):
    def test_gpu_lease_rejection_returns_visible_failure_and_empty_outputs(self):
        with patch.object(portfolio_app, '_run_analysis',
                          side_effect=portfolio_app.gr.Error('GPU quota exceeded')):
            status, generated, links, explanation, trace = portfolio_app.analyze(
                'Title', 'A sufficiently long article body for this test.', 'Follow diplomacy',
                '', '', '', '', '', '', '', progress=NoOpProgress())
        self.assertIn('Analysis unavailable', status)
        self.assertEqual((generated, links, explanation), ({}, [], {}))
        self.assertEqual(trace['status'], 'failed')
        self.assertIn('GPU quota', trace['error'])

    def test_empty_article_does_not_acquire_a_gpu_lease(self):
        with patch.object(portfolio_app, '_run_analysis') as run:
            result = portfolio_app.analyze('Title', '', 'Follow diplomacy',
                                          '', '', '', '', '', '', '')
        run.assert_not_called()
        self.assertEqual(result[4]['status'], 'failed')

    def test_empty_interest_does_not_acquire_a_gpu_lease(self):
        with patch.object(portfolio_app, '_run_analysis') as run:
            result = portfolio_app.analyze('Title', 'A sufficiently long article body.', '',
                                          '', '', '', '', '', '', '')
        run.assert_not_called()
        self.assertEqual(result[4]['status'], 'failed')

    def test_unavailable_model_is_visible_without_fabricated_output(self):
        with patch.object(portfolio_app, "get_pipeline",
                          side_effect=OSError("Qwen model files are unavailable")):
            status, generated, links, explanation, trace = portfolio_app.analyze(
                "Title", "A sufficiently long article body for this test.", "Follow diplomacy",
                "", "", "", "", "", "", "", progress=NoOpProgress())

        self.assertIn("Analysis unavailable", status)
        self.assertEqual(generated, {})
        self.assertEqual(links, [])
        self.assertEqual(explanation, {})
        self.assertEqual(trace["status"], "failed")
        self.assertIn("unavailable", trace["error"])

    def test_malformed_model_output_is_visible_in_failure_trace(self):
        class FailingPipeline:
            def analyze_article(self, *_args, **_kwargs):
                error = ValueError("Invalid generated JSON")
                error.raw_outputs = ['{"countries": [']
                error.chunk_metadata = [{"input_tokens": 123, "output_tokens": 4}]
                raise error

        with patch.object(portfolio_app, "get_pipeline", return_value=FailingPipeline()):
            status, generated, links, explanation, trace = portfolio_app.analyze(
                "Title", "A sufficiently long article body for this test.", "Follow diplomacy",
                "", "", "", "", "", "", "", progress=NoOpProgress())

        self.assertIn("Analysis unavailable", status)
        self.assertEqual(generated, {})
        self.assertEqual(links, [])
        self.assertEqual(explanation, {})
        self.assertEqual(trace["status"], "failed")
        self.assertEqual(trace["raw_model_outputs"], ['{"countries": ['])
        self.assertEqual(trace["chunk_metadata"][0]["input_tokens"], 123)


if __name__ == "__main__":
    unittest.main()
