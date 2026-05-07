from collections import defaultdict
from typing import Iterable, Dict, Set

class MultiLabelMetrics:
    """
    Multi-label metrics over a label set:
      - per-label TP/FP/FN
      - micro-precision/recall/F1
      - macro-precision/recall/F1 over labels that appear in GT
    """

    def __init__(self):
        # label -> {"tp":int, "fp":int, "fn":int}
        self.stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        self.per_label = {}
        self.n_samples = 0

    @staticmethod
    def _to_set(labels: Iterable[str]) -> Set[str]:
        return {str(l).strip() for l in labels if str(l).strip()}

    def update(self, y_true: Iterable[str], y_pred: Iterable[str]) -> None:
        """
        y_true, y_pred: iterables of label strings for one sample.
        """
    
        self.n_samples += 1

        t = self._to_set(y_true)
        p = self._to_set(y_pred)
        #print(f"y_true: {t}, y_pred: {p}")
        inter = t & p
        #print(f"inter: {inter}")
        only_p = p - t
        #print(f"only_p: {only_p}")
        only_t = t - p
        #print(f"only_t: {only_t}")

        for lab in inter:
            self.stats[lab]["tp"] += 1
        for lab in only_p:
            self.stats[lab]["fp"] += 1
        for lab in only_t:
            self.stats[lab]["fn"] += 1

    def compute(self) -> Dict:
        """
        Compute micro/macro precision/recall/F1 average over all and per-label metrics.
        """
        # micro
        tp_sum = sum(v["tp"] for v in self.stats.values())
        fp_sum = sum(v["fp"] for v in self.stats.values())
        fn_sum = sum(v["fn"] for v in self.stats.values())

        micro_p = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) else 0.0
        micro_r = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) else 0.0
        micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) else 0.0

        # per-label + macro
        p_sum = r_sum = f1_sum = 0.0
        n_labels_with_gt = 0

        total = self.n_samples

        for lab, s in self.stats.items():
            tp, fp, fn = s["tp"], s["fp"], s["fn"]
            tn = total - tp - fp - fn

            # precision / recall / f1
            if tp + fn == 0:
                prec = rec = f1 = 0.0
            else:
                prec = tp / (tp + fp) if (tp + fp) else 0.0
                rec  = tp / (tp + fn) if (tp + fn) else 0.0
                f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
                p_sum += prec
                r_sum += rec
                f1_sum += f1
                n_labels_with_gt += 1

            # additional metrics
            tpr = rec # recall = TPR
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            tnr = tn / (tn + fp) if (tn + fp) else 0.0
            fnr = fn / (fn + tp) if (fn + tp) else 0.0

            self.per_label[lab] = {
                "precision": prec,
                "recall": rec, # tpr
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "tpr": tpr,
                "fpr": fpr,
                "tnr": tnr,
                "fnr": fnr,
            }

        if n_labels_with_gt > 0:
            macro_p = p_sum / n_labels_with_gt
            macro_r = r_sum / n_labels_with_gt
            macro_f1 = f1_sum / n_labels_with_gt
        else:
            macro_p = macro_r = macro_f1 = 0.0

        return {
            "micro_precision": micro_p,
            "micro_recall": micro_r,
            "micro_f1": micro_f1,
            "macro_precision": macro_p,
            "macro_recall": macro_r,
            "macro_f1": macro_f1,
            "per_label": self.per_label,
            "n_samples": self.n_samples,
        }