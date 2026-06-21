import json
import re
import sys
sys.path.append('/pfss/mlde/workspaces/mlde_wsp_PI_Kersting/am84fuxo/LlavaGuard')

from typing import Dict

from llavaguard_config import local_data_dir
from pegiguard.evaluation.multilabel_metrics import MultiLabelMetrics

VALID_AGE_RATINGS = {"3", "7", "12", "16", "18"}

def normalize_label(lbl: str) -> str:
    """docstring"""
    lbl = lbl.strip()

    # Normalize all NA variants
    if lbl.lower() in {
        "na: none applicable",
        "na: none applying"
    }:
        return "NA: None applicable"

    return lbl


def normalize_rating(lbl: str) -> str:
    """Normalize age rating strings for comparison.

    Accept numeric ratings and formats like "PEGI 12", "12", "16+", etc.
    """
    if lbl is None:
        return None
    raw = str(lbl).strip()
    if not raw:
        return None

    raw = raw.upper().replace("PEGI", "").strip()
    digits = re.findall(r"\d+", raw)
    if not digits:
        return None
    rating = digits[0]
    return rating if rating in VALID_AGE_RATINGS else None


def _ensure_list(x):
    """for the subcategories and categoiries String"""
    if x is None: return []
    if isinstance(x, list): return x
    if isinstance(x, str):
        # handle comma-separated strings
        try:
            j = json.loads(x)
            if isinstance(j, list): return j
        except Exception:
            pass
        return [s.strip() for s in x.split(",") if s.strip()]
    return [x]

class EvaluationMetricsCalculator:
    """docstring"""
    def __init__(self):
        self.n_samples = 0
        self.TP, self.FP, self.TN, self.FN, self.P, self.N = [], [], [], [], [], []
        self.per_label = {}
        self.invalid_assessments = []
        self.safe = "Safe"
        self.unsafe = "Unsafe"
        # for age rating
        #self.rating_correct = TP
        self.rating_over = 0
        self.rating_under = 0
        self.rating_per_label = {rating: {"support": 0, "correct": 0} for rating in VALID_AGE_RATINGS}
        self.rating_confusion = {true: {pred: 0 for pred in VALID_AGE_RATINGS} for true in VALID_AGE_RATINGS}
        self.invalid_ratings = []
        # multi-label metrics for categories + subcategories
        self.cat_metrics = MultiLabelMetrics()
        self.subcat_metrics = MultiLabelMetrics()

    def add_sample(self, mode, sample_id, pred_dict, ground_truth):
        """docstring"""
        gt_dict = json.loads(ground_truth)
        if mode == "safety":
            # update assessment metrics and category metrics together
            pred_assessment = pred_dict['assessment']
            true_assessment = gt_dict['assessment']

            if true_assessment == self.safe:
                self.N.append(sample_id)
            elif true_assessment == self.unsafe:
                self.P.append(sample_id)
            else:
                raise ValueError(f'Invalid ground truth for sample {sample_id} with safety rating: {true_assessment}')
            
            if pred_assessment == self.safe and true_assessment == self.safe:
                self.TN.append(sample_id)
            elif pred_assessment == self.safe and true_assessment == self.unsafe:
                self.FN.append(sample_id)
            elif pred_assessment == self.unsafe and true_assessment == self.safe:
                self.FP.append(sample_id)
            elif pred_assessment == self.unsafe and true_assessment == self.unsafe:
                self.TP.append(sample_id)
            else:
                self.invalid_assessments.append(sample_id)

            gt_cats = [normalize_label(x) for x in _ensure_list(gt_dict.get("categories"))]
            gt_subs = [normalize_label(x) for x in _ensure_list(gt_dict.get("subcategories"))]
            pred_cats = [normalize_label(x) for x in _ensure_list(pred_dict.get("categories"))]
            pred_subs = [normalize_label(x) for x in _ensure_list(pred_dict.get("subcategories"))]
            self.cat_metrics.update(gt_cats, pred_cats)
            self.subcat_metrics.update(gt_subs, pred_subs)

        elif mode == "age_rating":
            #print("*** ADDING SAMPLE FOR AGE RATING ***")
            pred_rating = normalize_rating(pred_dict.get("PEGI rating"))
            true_rating = normalize_rating(gt_dict.get("PEGI rating"))

            if pred_rating not in VALID_AGE_RATINGS:
                self.invalid_ratings.append(sample_id)
                return

            self.n_samples += 1
            #self.rating_per_label[true_rating]["support"] += 1

            self.rating_confusion[true_rating][pred_rating] += 1

            if pred_rating > true_rating:
                self.rating_over += 1
            elif pred_rating < true_rating:
                self.rating_under += 1

        else:
             raise ValueError("Wrong mode of evaluation")

    def get_metrics(self, true_positives, false_positives, true_negatives, false_negatives):
        '''
        function to calculate the evaluation metrics
        :param true_positives: number of true positives
        :param false_positives: number of false positives
        :param true_negatives: number of true negatives
        :param false_negatives: number of false negatives
        :return: TPR, FPR, FNR, TNR, precision, acc, bal_acc, F1, F
        '''
        TPR = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
        FPR = false_positives / (false_positives + true_negatives) if false_positives + true_negatives > 0 else 0
        FNR = false_negatives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
        TNR = true_negatives / (false_positives + true_negatives) if false_positives + true_negatives > 0 else 0
        TPR, FPR, FNR, TNR = round(TPR, 4), round(FPR, 4), round(FNR, 4), round(TNR, 4)
        precision = round(true_positives / (true_positives + false_positives), 4) if true_positives + false_positives > 0 else 0
        num_samples = true_positives + false_positives + true_negatives + false_negatives
        acc = round((true_positives + true_negatives) / num_samples, 4) if num_samples > 0 else 0
        bal_acc = round((TPR + TNR) / 2, 4) if num_samples > 0 else 0
        F1 = 2 * (precision * TPR) / (precision + TPR) if precision + TPR > 0 else 0
        F2 = 5 * (precision * TPR) / (4 * precision + TPR) if 4 * precision + TPR > 0 else 0
        return TPR, FPR, FNR, TNR, precision, acc, bal_acc, F1, F2

    def compute(self, mode) -> Dict[str, float]:
        """docstring"""
        metrics = {}  
        if mode == "safety":
            true_positives, false_positives, true_negatives, false_negatives = len(self.TP), len(self.FP), len(self.TN), len(self.FN)
            P = self.P
            N = self.N
            all_samples = P + N
            if len(all_samples) == 0:
                print('No samples to evaluate.')
                return {}
            num_samples = true_negatives + false_negatives + true_positives + false_positives
            TPR, FPR, FNR, TNR, precision, acc, bal_acc, f1, f2 = self.get_metrics(true_positives, false_positives, true_negatives, false_negatives)

            metrics.update({
                'Balanced Accuracy': bal_acc,
                'Accuracy': acc,
                'Number of Samples': len(all_samples),
                'Classified Samples': num_samples,
                'TP': len(self.TP),
                'FP': len(self.FP),
                'TN': len(self.TN),
                'FN': len(self.FN),
                'Invalid': len(self.invalid_assessments),
                'TPR': TPR,
                'FPR': FPR,
                'FNR': FNR,
                'TNR': TNR,
                'Precision': precision,
                'Recall': TPR,
                'Specificity': TNR,
                'F1': f1,
                'F2': f2,
                'Invalid_list': self.invalid_assessments,
            })

            cat_res = self.cat_metrics.compute()
            metrics.update({
                "categories_micro_precision": cat_res["micro_precision"],
                "categories_micro_recall": cat_res["micro_recall"],
                "categories_micro_f1": cat_res["micro_f1"],
                "categories_macro_precision": cat_res["macro_precision"],
                "categories_macro_recall": cat_res["macro_recall"],
                "categories_macro_f1": cat_res["macro_f1"],
                "per_category_metrics": cat_res["per_label"]
            })

            sub_res = self.subcat_metrics.compute()
            metrics.update({
                "subcategories_micro_precision": sub_res["micro_precision"],
                "subcategories_micro_recall": sub_res["micro_recall"],
                "subcategories_micro_f1": sub_res["micro_f1"],
                "subcategories_macro_precision": sub_res["macro_precision"],
                "subcategories_macro_recall": sub_res["macro_recall"],
                "subcategories_macro_f1": sub_res["macro_f1"],
                "per_subcategory_metrics": sub_res["per_label"]
            })
        elif mode == "age_rating":
            if self.n_samples == 0:
                print('No rating samples to evaluate.')
                return {}

            per_rating_metrics = {}          
            total = self.n_samples
            total_tp = 0
            total_fp = 0
            total_tn = 0
            total_fn = 0
            #balanced_accum = 0.0
            #valid_label_count = 0

            for rating in sorted(VALID_AGE_RATINGS, key=lambda x: int(x)):
                # for the per label metrics
                true_positives = self.rating_confusion[rating][rating]
                false_negatives = sum(self.rating_confusion[rating][pred] for pred in VALID_AGE_RATINGS if pred != rating)
                false_positives = sum(self.rating_confusion[true][rating] for true in VALID_AGE_RATINGS if true != rating)
                true_negatives = total - true_positives - false_positives - false_negatives

                TPR, FPR, FNR, TNR, precision, acc, bal_acc, f1, f2 = self.get_metrics(true_positives, false_positives, true_negatives, false_negatives)

                # for overall metrics
                total_tp += true_positives
                total_fp += false_positives
                total_tn += true_negatives
                total_fn += false_negatives
                #balanced_accum += bal_acc

                per_rating_metrics[rating] = {
                    'Balanced Accuracy': bal_acc,
                    'Accuracy': acc,                
                    #"support": true_positives + false_negatives,
                    #"correct": true_positives,
                    'TP': true_positives,
                    'FP': false_positives,
                    'TN': true_negatives,
                    'FN': false_negatives,
                    'Invalid': len(self.invalid_assessments),
                    'TPR': TPR,
                    'FPR': FPR,
                    'FNR': FNR,
                    'TNR': TNR,
                    'Precision': precision,
                    'Recall': TPR,
                    'Specificity': TNR,
                    'F1': f1,
                    'F2': f2,
                    'Invalid_list': self.invalid_assessments
                }
            
            TPR, FPR, FNR, TNR, precision, accuracy, bal_acc, f1, f2 = self.get_metrics(total_tp, total_fp, total_tn, total_fn)
            
            metrics.update({
                'Total Balanced Accuracy': bal_acc,
                'Total Accuracy': accuracy,
                "number_of_samples": self.n_samples,
                'total_TP': total_tp,
                'total_FP': total_fp,
                'total_TN': total_tn,
                'total_FN': total_fn,
                'total_invalid': len(self.invalid_ratings),
                'TPR': TPR,
                'FPR': FPR,
                'FNR': FNR,
                'TNR': TNR,
                'Precision': precision,
                'Total F1': f1, 
                'Total F2': f2,
                'rating_per_label': per_rating_metrics,
                'rating_confusion_matrix': self.rating_confusion,
                'invalid_list': self.invalid_ratings,
                'ratings_over_rate': round(self.rating_over),
                'ratings_under_rate': round(self.rating_under),
            })
        return metrics