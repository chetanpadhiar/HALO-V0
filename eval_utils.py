"""
HALO ML — full evaluation utility.

The previous report leaned on raw accuracy, which is misleading here:
malicious (37,036) outnumbers legitimate (2,501) about 14.8:1 overall,
and the TEST split specifically is 93.6% malicious -- so a classifier
that always predicts "malicious" scores 93.6% raw accuracy while
having ZERO ability to recognize a legitimate email. Every model from
here on is judged primarily on balanced accuracy, macro-F1, PR-AUC,
and per-class recall, with raw accuracy reported only alongside the
majority-class baseline for context.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    f1_score, matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix,
)


def majority_baseline(y_true):
    """Accuracy of always predicting the majority class -- the floor any
    real model must clear to be worth using."""
    majority = 1 if y_true.mean() >= 0.5 else 0
    preds = np.full_like(y_true, majority)
    return {
        "majority_class": "malicious" if majority == 1 else "legitimate",
        "accuracy": accuracy_score(y_true, preds),
        "balanced_accuracy": balanced_accuracy_score(y_true, preds),
        "macro_f1": f1_score(y_true, preds, average="macro", zero_division=0),
    }


def full_metrics(y_true, scores, threshold):
    """scores: predicted probability/score of the MALICIOUS class."""
    preds = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

    prec_per_class, rec_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_true, preds, labels=[0, 1], zero_division=0
    )

    specificity = tn / (tn + fp) if (tn + fp) else 0.0  # = legitimate recall = TNR
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    try:
        roc_auc = roc_auc_score(y_true, scores)
    except ValueError:
        roc_auc = float("nan")
    try:
        pr_auc = average_precision_score(y_true, scores)
    except ValueError:
        pr_auc = float("nan")

    return {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "accuracy": accuracy_score(y_true, preds),
        "balanced_accuracy": balanced_accuracy_score(y_true, preds),
        "mcc": matthews_corrcoef(y_true, preds) if len(set(preds)) > 1 else 0.0,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "macro_f1": f1_score(y_true, preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, preds, average="weighted", zero_division=0),
        "legitimate": {  # class 0
            "precision": prec_per_class[0], "recall": rec_per_class[0],
            "f1": f1_per_class[0], "support": int(support[0]),
        },
        "malicious": {  # class 1
            "precision": prec_per_class[1], "recall": rec_per_class[1],
            "f1": f1_per_class[1], "support": int(support[1]),
        },
        "specificity_legitimate_recall": specificity,
        "fpr": fpr,
        "fnr": fnr,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def print_metrics(m, label=""):
    print(f"--- {label} (n={m['n']}, threshold={m['threshold']:.3f}) ---")
    print(f"  accuracy={m['accuracy']:.4f}   balanced_accuracy={m['balanced_accuracy']:.4f}   MCC={m['mcc']:.4f}")
    print(f"  ROC-AUC={m['roc_auc']:.4f}   PR-AUC={m['pr_auc']:.4f}")
    print(f"  macro-F1={m['macro_f1']:.4f}   weighted-F1={m['weighted_f1']:.4f}")
    print(f"  legitimate: precision={m['legitimate']['precision']:.4f} recall(specificity)={m['legitimate']['recall']:.4f} f1={m['legitimate']['f1']:.4f} (n={m['legitimate']['support']})")
    print(f"  malicious : precision={m['malicious']['precision']:.4f} recall={m['malicious']['recall']:.4f} f1={m['malicious']['f1']:.4f} (n={m['malicious']['support']})")
    print(f"  FPR={m['fpr']:.4f}  FNR={m['fnr']:.4f}  confusion_matrix={m['confusion_matrix']}")


def best_threshold_by_balanced_accuracy(y_true, scores, min_malicious_recall=0.0):
    """Choose the threshold maximizing balanced accuracy on the given
    (validation) set, optionally requiring a minimum malicious recall
    so the operating point doesn't quietly let too much phishing through
    in exchange for a nicer-looking legitimate-class number."""
    candidates = np.unique(scores)
    if len(candidates) > 500:
        candidates = candidates[np.linspace(0, len(candidates) - 1, 500).astype(int)]
    rows = [full_metrics(y_true, scores, t) for t in candidates]
    pool = [r for r in rows if r["malicious"]["recall"] >= min_malicious_recall] or rows
    return max(pool, key=lambda r: r["balanced_accuracy"])
