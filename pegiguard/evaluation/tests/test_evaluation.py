"""Regression checks for rating metrics and resumable evaluation without a model."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pegiguard.evaluation import eval_rating
from pegiguard.evaluation.metrics_calculator import (
    EvaluationMetricsCalculator, get_binary_metrics, normalize_rating,
)


class MetricsTests(unittest.TestCase):
    def test_canonical_ratings_are_strings(self):
        for value in ("12", 12, 12.0, "PEGI 12", "12+", " 12 "):
            with self.subTest(value=value):
                self.assertEqual(normalize_rating(value), "12")
        for value in (None, "12.5", "12 or 18", "120", True, "PEGI 4", "nan"):
            with self.subTest(value=value):
                self.assertIsNone(normalize_rating(value))

    def test_rating_metrics_and_numeric_order(self):
        calculator = EvaluationMetricsCalculator()
        for sample_id, (prediction, reference) in enumerate(
            [("3", "3"), ("12", "7"), ("7", "12"), ("18", "18"), ("invalid", "16")]
        ):
            calculator.add_sample("age_rating", sample_id,
                                  {"PEGI rating": prediction}, json.dumps({"PEGI rating": reference}))
        with contextlib.redirect_stdout(io.StringIO()):
            metrics = calculator.compute("age_rating")
        self.assertEqual(metrics["Accuracy"], 0.5)
        self.assertEqual(metrics["micro_f1"], 0.5)
        self.assertEqual(metrics["One-vs-Rest Accuracy"], 0.8)
        self.assertEqual(metrics["attempted_samples"], 5)
        self.assertEqual(metrics["total_invalid"], 1)
        self.assertEqual(metrics["ratings_over_rate"], 1)
        self.assertEqual(metrics["ratings_under_rate"], 1)
        self.assertEqual(metrics["rating_per_label"]["7"]["FP"], 1)
        self.assertEqual(metrics["rating_per_label"]["7"]["FN"], 1)
        self.assertEqual(metrics["rating_per_label"]["3"]["Recall"], 1.0)
        self.assertEqual(metrics["rating_confusion_matrix"]["7"]["12"], 1)

    def test_invalid_outputs_remain_visible(self):
        calculator = EvaluationMetricsCalculator()
        calculator.add_sample("age_rating", "invalid", {}, {"PEGI rating": "3"})
        with contextlib.redirect_stdout(io.StringIO()):
            metrics = calculator.compute("age_rating")
        self.assertEqual(metrics["number_of_samples"], 0)
        self.assertEqual(metrics["attempted_samples"], 1)
        self.assertEqual(metrics["invalid_list"], ["invalid"])
        self.assertIn("rating_confusion_matrix", metrics)

    def test_invalid_ground_truth_does_not_mutate_metrics(self):
        calculator = EvaluationMetricsCalculator()
        with self.assertRaises(ValueError):
            calculator.add_sample("age_rating", "bad", {"PEGI rating": "3"}, {"PEGI rating": "4"})
        self.assertEqual(calculator.n_samples, 0)
        self.assertEqual(calculator.invalid_ratings, [])

    def test_safety_and_fractional_f1(self):
        calculator = EvaluationMetricsCalculator()
        for i, (pred, true) in enumerate([("Safe", "Safe"), ("Unsafe", "Unsafe"),
                                         ("Safe", "Unsafe"), ("Unsafe", "Safe"), (None, "Safe")]):
            calculator.add_sample("safety", i, {"assessment": pred}, json.dumps({"assessment": true}))
        with contextlib.redirect_stdout(io.StringIO()):
            metrics = calculator.compute("safety")
        for key in ("TP", "FP", "TN", "FN", "Invalid"):
            self.assertEqual(metrics[key], 1)
        self.assertEqual(metrics["F1"], 0.5)
        self.assertEqual(get_binary_metrics(1, 2, 0, 3)["F1"], 0.2857)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data.json"
        self.responses = self.root / "responses.json"
        self.output = self.root / "metrics.json"
        self.items = [self.item("a"), self.item("b")]
        self.data.write_text(json.dumps(self.items))

    @staticmethod
    def item(sample_id):
        return {"id": sample_id, "image path": "unused.jpg", "conversations": [
            {"value": "prompt"}, {"value": json.dumps({"PEGI rating": "12"})},
        ]}

    @staticmethod
    def record(sample_id, response=None):
        return {"id": sample_id, "model_response": response}

    def run_evaluation(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return eval_rating.evaluate_dataset("fake", "age_rating", str(self.data),
                                                str(self.output), str(self.responses), **kwargs)

    def test_cache_is_deduplicated_and_incomplete_entries_are_retried(self):
        good = {"PEGI rating": "12"}
        self.responses.write_text(json.dumps([
            self.record("a", good), self.record("a", good), self.record("a"), self.record("b"),
        ]))
        with patch.object(eval_rating, "call_vllm", return_value=good) as inference:
            metrics = self.run_evaluation()
        inference.assert_called_once()
        self.assertEqual(metrics["number_of_samples"], 2)
        self.assertEqual(metrics["Accuracy"], 1.0)
        self.assertEqual(len(json.loads(self.responses.read_text())), 2)

    def test_responses_only_never_calls_model_or_changes_cache(self):
        self.responses.write_text(json.dumps([self.record("a", {"PEGI rating": "12"})]))
        before = self.responses.read_bytes()
        with patch.object(eval_rating, "call_vllm") as inference:
            metrics = self.run_evaluation(responses_only=True)
        inference.assert_not_called()
        self.assertEqual(self.responses.read_bytes(), before)
        self.assertEqual(metrics["missing_response_ids"], ["b"])
        self.assertFalse(metrics["complete"])
        self.assertEqual(metrics["number_of_samples"], 1)

    def test_invalid_prediction_is_counted_without_regenerating(self):
        self.responses.write_text(json.dumps([self.record("a", {})]))
        with patch.object(eval_rating, "call_vllm") as inference:
            metrics = self.run_evaluation(responses_only=True)
        inference.assert_not_called()
        self.assertEqual(metrics["invalid_list"], ["a"])
        self.assertEqual(metrics["missing_response_ids"], ["b"])

    def test_bad_cache_is_not_overwritten(self):
        self.responses.write_text("broken JSON")
        with patch.object(eval_rating, "call_vllm") as inference:
            with self.assertRaises(json.JSONDecodeError):
                self.run_evaluation()
        inference.assert_not_called()
        self.assertEqual(self.responses.read_text(), "broken JSON")
        self.assertFalse(self.output.exists())

    def test_duplicate_dataset_ids_fail_before_inference(self):
        self.data.write_text(json.dumps([self.item("a"), self.item("a")]))
        with patch.object(eval_rating, "call_vllm") as inference:
            with self.assertRaisesRegex(ValueError, "Duplicate dataset ID"):
                self.run_evaluation()
        inference.assert_not_called()

    def test_failed_inference_is_reported_and_success_is_saved(self):
        with patch.object(eval_rating, "call_vllm", side_effect=[RuntimeError("unavailable"), {"PEGI rating": "12"}]):
            metrics = self.run_evaluation()
        self.assertEqual(metrics["failed_response_ids"], ["a"])
        self.assertFalse(metrics["complete"])
        self.assertEqual(metrics["number_of_samples"], 1)
        self.assertEqual(json.loads(self.responses.read_text())[0]["id"], "b")

    def test_numeric_and_string_ids_match(self):
        self.data.write_text(json.dumps([self.item(1)]))
        self.responses.write_text(json.dumps([self.record("1", {"PEGI rating": "12"})]))
        with patch.object(eval_rating, "call_vllm") as inference:
            metrics = self.run_evaluation(responses_only=True)
        inference.assert_not_called()
        self.assertEqual(metrics["number_of_samples"], 1)
        self.assertTrue(metrics["complete"])

    def test_empty_dataset_has_a_report(self):
        self.data.write_text("[]")
        metrics = self.run_evaluation(responses_only=True)
        self.assertEqual(metrics["dataset_samples"], 0)
        self.assertEqual(metrics["number_of_samples"], 0)
        self.assertTrue(metrics["complete"])


class JsonExtractionTests(unittest.TestCase):
    def test_json_formats_and_trailing_commas(self):
        for text in ('{"PEGI rating":"12"}', '```json\n{"PEGI rating":"12",}\n```',
                     'Explanation. {"PEGI rating":"12"} Done.'):
            self.assertEqual(eval_rating.extract_json(text)["PEGI rating"], "12")

    def test_rationale_text_is_preserved(self):
        text = '{"rationale":"A comma,} inside text", "PEGI rating":"12",}'
        self.assertEqual(eval_rating.extract_json(text)["rationale"], "A comma,} inside text")

    def test_non_object_json_is_rejected(self):
        for text in ('[]', 'null', '"12"', '[{"PEGI rating":"12"}]', 'No answer'):
            with self.assertRaises(ValueError):
                eval_rating.extract_json(text)


if __name__ == "__main__":
    unittest.main()
