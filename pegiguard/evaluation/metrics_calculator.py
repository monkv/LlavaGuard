"""Metrics for safety assessments and PEGI age ratings."""

import ast
import json
import re
from typing import Any, Dict, Optional

from pegiguard.evaluation.multilabel_metrics import MultiLabelMetrics

VALID_AGE_RATINGS = ["3", "7", "12", "16", "18"]


def normalize_label(label: str) -> str:
    """Normalize whitespace and equivalent NA labels."""
    label = str(label).strip()
    if label.lower() in {"na: none applicable", "na: none applying"}:
        return "NA: None applicable"
    return label


def normalize_rating(value: Any) -> Optional[str]:
    """Return a canonical string rating, rejecting partial or fractional matches."""
    match = re.fullmatch(
        r"(?:PEGI\s*)?(3|7|12|16|18)(?:\.0+)?\+?", str(value).strip(), re.I
    )
    return match.group(1) if match else None


def _ensure_list(value):
    """Accept labels as lists, serialized lists, or comma-separated strings."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        if value.startswith("["):
            try:
                labels = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return [value]
            if isinstance(labels, list):
                return labels
        return [label.strip() for label in value.split(",") if label.strip()]
    return [value]


def validate_ground_truth(mode, ground_truth):
    """Reject invalid reference data before updating any counters."""
    if mode not in {"safety", "age_rating"}:
        raise ValueError(f"Unsupported evaluation mode: {mode!r}")
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)
    if not isinstance(ground_truth, dict):
        raise ValueError("Ground truth must be a JSON object")
    if mode == "age_rating" and normalize_rating(ground_truth.get("PEGI rating")) is None:
        raise ValueError(f"Invalid ground-truth PEGI rating: {ground_truth.get('PEGI rating')!r}")
    if mode == "safety" and ground_truth.get("assessment") not in {"Safe", "Unsafe"}:
        raise ValueError(f"Invalid ground-truth assessment: {ground_truth.get('assessment')!r}")
    return ground_truth


def get_binary_metrics(true_positives, false_positives, true_negatives, false_negatives):
    """Calculate binary metrics at full precision and round only the results."""
    tp, fp, tn, fn = true_positives, false_positives, true_negatives, false_negatives

    def divide(numerator, denominator):
        return numerator / denominator if denominator else 0.0

    recall = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    precision = divide(tp, tp + fp)
    return {
        "Balanced Accuracy": round((recall + specificity) / 2, 4),
        "Accuracy": round(divide(tp + tn, tp + fp + tn + fn), 4),
        "TPR": round(recall, 4),
        "FPR": round(divide(fp, fp + tn), 4),
        "FNR": round(divide(fn, fn + tp), 4),
        "TNR": round(specificity, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "Specificity": round(specificity, 4),
        "F1": round(divide(2 * tp, 2 * tp + fp + fn), 4),
        "F2": round(divide(5 * tp, 5 * tp + fp + 4 * fn), 4),
    }


class EvaluationMetricsCalculator:
    """Accumulate predictions while tracking invalid outputs separately.

    Rating classification metrics exclude invalid predictions; attempted and
    invalid counts are always reported. Macro averages use labels present in
    ground truth, following MultiLabelMetrics.
    """

    def __init__(self):
        self.n_samples = 0
        self.TP, self.FP, self.TN, self.FN, self.P, self.N = [], [], [], [], [], []
        self.invalid_assessments = []
        self.safe, self.unsafe = "Safe", "Unsafe"
        self.rating_confusion = {
            true: {pred: 0 for pred in VALID_AGE_RATINGS} for true in VALID_AGE_RATINGS
        }
        self.invalid_ratings = []
        self.multilabel_metrics = MultiLabelMetrics()
        self.rating_over = 0
        self.rating_under = 0
        self.cat_metrics = MultiLabelMetrics()
        self.subcat_metrics = MultiLabelMetrics()

    def add_sample(self, mode, sample_id, prediction, ground_truth):
        """Accept dict or JSON ground truth; count malformed predictions as invalid."""
        ground_truth = validate_ground_truth(mode, ground_truth)
        prediction = prediction if isinstance(prediction, dict) else {}
        if mode == "safety":
            self._add_safety_sample(sample_id, prediction, ground_truth)
        else:
            self._add_rating_sample(sample_id, prediction, ground_truth)

    def _add_safety_sample(self, sample_id, prediction, ground_truth):
        pred = prediction.get("assessment")
        true = ground_truth["assessment"]
        (self.P if true == self.unsafe else self.N).append(sample_id)
        if pred == self.safe:
            (self.TN if true == self.safe else self.FN).append(sample_id)
        elif pred == self.unsafe:
            (self.FP if true == self.safe else self.TP).append(sample_id)
        else:
            self.invalid_assessments.append(sample_id)
        for field, accumulator in (("categories", self.cat_metrics), ("subcategories", self.subcat_metrics)):
            accumulator.update(
                [normalize_label(label) for label in _ensure_list(ground_truth.get(field))],
                [normalize_label(label) for label in _ensure_list(prediction.get(field))],
            )

    def _add_rating_sample(self, sample_id, prediction, ground_truth):
        pred = normalize_rating(prediction.get("PEGI rating"))
        true = normalize_rating(ground_truth["PEGI rating"])
        if pred is None:
            self.invalid_ratings.append(sample_id)
            return
        self.n_samples += 1
        self.rating_confusion[true][pred] += 1
        self.multilabel_metrics.update([true], [pred])
        # Keep label keys as strings and compare ages numerically.
        self.rating_over += int(int(pred) > int(true))
        self.rating_under += int(int(pred) < int(true))

    def compute(self, mode) -> Dict[str, Any]:
        """Return a report even if all predictions are invalid or input is empty."""
        if mode == "safety":
            return self._compute_safety_metrics()
        if mode == "age_rating":
            return self._compute_rating_metrics()
        raise ValueError(f"Unsupported evaluation mode: {mode!r}")

    def _compute_safety_metrics(self):
        tp, fp, tn, fn = len(self.TP), len(self.FP), len(self.TN), len(self.FN)
        metrics = {
            "Number of Samples": len(self.P) + len(self.N),
            "Classified Samples": tp + fp + tn + fn,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            **get_binary_metrics(tp, fp, tn, fn),
            "Invalid": len(self.invalid_assessments),
            "Invalid_list": list(self.invalid_assessments),
        }
        for prefix, accumulator, per_label in (
            ("categories", self.cat_metrics, "per_category_metrics"),
            ("subcategories", self.subcat_metrics, "per_subcategory_metrics"),
        ):
            result = accumulator.compute()
            for averaging in ("micro", "macro"):
                for metric in ("precision", "recall", "f1"):
                    key = f"{averaging}_{metric}"
                    metrics[f"{prefix}_{key}"] = result[key]
            metrics[per_label] = result["per_label"]
        return metrics

    def _compute_rating_metrics(self):
        per_rating = {}
        total_tp = total_fp = total_tn = total_fn = 0
        for rating in VALID_AGE_RATINGS:
            tp = self.rating_confusion[rating][rating]
            fn = sum(self.rating_confusion[rating][pred] for pred in VALID_AGE_RATINGS if pred != rating)
            fp = sum(self.rating_confusion[true][rating] for true in VALID_AGE_RATINGS if true != rating)
            tn = self.n_samples - tp - fp - fn
            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn
            per_rating[rating] = {
                "TP": tp, "FP": fp, "TN": tn, "FN": fn,
                **get_binary_metrics(tp, fp, tn, fn),
                "Invalid": len(self.invalid_ratings),
                "Invalid_list": list(self.invalid_ratings),
            }
        overall = get_binary_metrics(total_tp, total_fp, total_tn, total_fn)
        multiclass = self.multilabel_metrics.compute()
        return {
            **{key: multiclass[key] for key in (
                "micro_precision", "micro_recall", "micro_f1", "macro_precision", "macro_recall", "macro_f1"
            )},
            "Accuracy": round(total_tp / self.n_samples, 4) if self.n_samples else 0.0,
            "One-vs-Rest Balanced Accuracy": overall["Balanced Accuracy"],
            "One-vs-Rest Accuracy": overall["Accuracy"],
            "number_of_samples": self.n_samples,
            "attempted_samples": self.n_samples + len(self.invalid_ratings),
            "total_TP": total_tp, "total_FP": total_fp, "total_TN": total_tn, "total_FN": total_fn,
            "total_invalid": len(self.invalid_ratings),
            **{key: overall[key] for key in ("TPR", "FPR", "FNR", "TNR")},
            "rating_per_label": per_rating,
            "rating_confusion_matrix": self.rating_confusion,
            "invalid_list": list(self.invalid_ratings),
            # These are counts, kept under their existing keys for compatibility.
            "ratings_over_rate": self.rating_over,
            "ratings_under_rate": self.rating_under,
        }
