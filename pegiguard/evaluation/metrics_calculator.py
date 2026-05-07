import json
import sys
sys.path.append('/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/am84fuxo/LlavaGuard')

from typing import Dict

from llavaguard_config import local_data_dir
from pegiguard.evaluation.multilabel_metrics import MultiLabelMetrics

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
        # multi-label metrics for categories + subcategories
        self.cat_metrics = MultiLabelMetrics()
        self.subcat_metrics = MultiLabelMetrics()

    def add_sample(self, mode, sample_id, pred_dict, ground_truth):
        """docstring"""
        gt_dict = json.loads(ground_truth)

        #print(f"**Adding a sample to emc {sample_id}**")
        if mode == "assessment":
        
            pred_assessment = pred_dict['assessment']
            true_assessment = gt_dict['assessment']

            # evaluating the assessment-label
            if true_assessment == self.safe:
                self.N.append(sample_id)
            elif true_assessment == self.unsafe:
                self.P.append(sample_id)
            else:
                raise ValueError(f'Invalid ground truth for sample {sample_id} with safety rating: {true_assessment}')
            if pred_assessment == self.safe and true_assessment == self.safe:
                    #print('Added to TN sample id:', sample_id)
                    self.TN.append(sample_id)
                    #print(f"List:{self.TN}")
            elif pred_assessment == self.safe and true_assessment == self.unsafe:
                    self.FN.append(sample_id)
            elif pred_assessment == self.unsafe and true_assessment == self.safe:
                    self.FP.append(sample_id)
            elif pred_assessment == self.unsafe and true_assessment == self.unsafe:
                    self.TP.append(sample_id)
            else:
                self.invalid_assessments.append(sample_id)
                
        elif mode == "categories":
            gt_cats = [normalize_label(x) for x in _ensure_list(gt_dict.get("categories"))]
            gt_subs = [normalize_label(x) for x in _ensure_list(gt_dict.get("subcategories"))]
            pred_cats = [normalize_label(x) for x in _ensure_list(pred_dict.get("categories"))]
            pred_subs = [normalize_label(x) for x in _ensure_list(pred_dict.get("subcategories"))]
            # evaluating the sub- category multilabels
            #print("**Updating the multilabel metrics**")
            self.cat_metrics.update(gt_cats, pred_cats)
            self.subcat_metrics.update(gt_subs, pred_subs)
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
        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
        num_samples = true_positives + false_positives + true_negatives + false_negatives
        acc = round((true_positives + true_negatives) / num_samples, 4) if num_samples > 0 else 0
        bal_acc = round((TPR + TNR) / 2, 4) if num_samples > 0 else 0
        F1 = 2 * (precision * TPR) / (precision + TPR) if precision + TPR > 0 else 0
        F2 = 5 * (precision * TPR) / (4 * precision + TPR) if 4 * precision + TPR > 0 else 0
        return TPR, FPR, FNR, TNR, precision, acc, bal_acc, F1, F2

    def compute(self, mode) -> Dict[str, float]:
        """docstring"""
        metrics = {}
        if mode == "assessment": 
            true_positives, false_positives, true_negatives, false_negatives = len(self.TP), len(self.FP), len(self.TN), len(self.FN)
            P = self.P
            N = self.N
            all_samples = P + N
            if len(all_samples) == 0:
                print('No samples to evaluate.')
                return None, None
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
                'TPR': TPR, #rated unsafe, is unsafe
                'FPR': FPR, #rated unsafe, is safe
                'FNR': FNR, #rated safe, is unsafe
                'TNR': TNR, #rated safe is safe
                'Precision': precision,
                'Recall': TPR,
                'Specificity': TNR,
                'F1': f1,
                'F2': f2,
                'Invalid_list': self.invalid_assessments,
                #'TP_list': self.TP,
                #'FP_list': self.FP,
                #'TN_list': self.TN,
                #'FN_list': self.FN,
            })
            #print(metrics_table)
            # multi-label metrics for over-categories
        elif mode == "categories":
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

            # multi-label metrics for subcategories
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
        return metrics