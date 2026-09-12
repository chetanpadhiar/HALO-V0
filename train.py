"""
HALO ML — Phase 2: train & compare candidate models.

Candidates:
  A. Structured-only  Logistic Regression
  B. Structured-only  Random Forest
  C. Structured-only  HistGradientBoosting
  D. Text-only         TF-IDF + Logistic Regression
  E. Hybrid (struct+text) Logistic Regression
  F. Hybrid (struct+text) Linear SVM

All candidates are compared on the VALIDATION set only. The TEST set
is touched exactly once, at the end, for the single selected model.
"""
import json
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                               recall_score, accuracy_score, confusion_matrix)

from features import STRUCTURED_FEATURE_NAMES

RANDOM_SEED = 42

df = pd.read_pickle("/home/claude/halo_ml/data/features.pkl")
idx = np.load("/home/claude/halo_ml/data/split_indices.npz")
train_idx, val_idx, test_idx = idx["train"], idx["val"], idx["test"]

train = df.iloc[train_idx].reset_index(drop=True)
val = df.iloc[val_idx].reset_index(drop=True)
test = df.iloc[test_idx].reset_index(drop=True)

y_train, y_val, y_test = train["label_bin"].values, val["label_bin"].values, test["label_bin"].values

X_struct_train = train[STRUCTURED_FEATURE_NAMES].values.astype(float)
X_struct_val = val[STRUCTURED_FEATURE_NAMES].values.astype(float)
X_struct_test = test[STRUCTURED_FEATURE_NAMES].values.astype(float)

scaler = StandardScaler().fit(X_struct_train)
X_struct_train_s = scaler.transform(X_struct_train)
X_struct_val_s = scaler.transform(X_struct_val)
X_struct_test_s = scaler.transform(X_struct_test)

tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
X_text_train = tfidf.fit_transform(train["text"])
X_text_val = tfidf.transform(val["text"])
X_text_test = tfidf.transform(test["text"])

X_hybrid_train = hstack([X_text_train, csr_matrix(X_struct_train_s)]).tocsr()
X_hybrid_val = hstack([X_text_val, csr_matrix(X_struct_val_s)]).tocsr()
X_hybrid_test = hstack([X_text_test, csr_matrix(X_struct_test_s)]).tocsr()


def eval_at(y_true, scores, threshold):
    preds = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "fnr": fn / (fn + tp) if (fn + tp) else 0.0,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def best_threshold(y_true, scores):
    """Pick the threshold on the F1-optimal point, but reject thresholds
    with FPR > 5% if any such threshold exists (false positives -- a
    legitimate campus email flagged as malicious -- are costlier for
    HALO than a missed low-confidence spam email, per the design brief)."""
    candidates = np.unique(scores)
    candidates = candidates[np.linspace(0, len(candidates) - 1, min(400, len(candidates))).astype(int)]
    rows = [eval_at(y_true, scores, t) for t in candidates]
    low_fpr = [r for r in rows if r["fpr"] <= 0.05]
    pool = low_fpr if low_fpr else rows
    best = max(pool, key=lambda r: r["f1"])
    return best


results = {}

print("=== A. Structured-only Logistic Regression ===")
clf_a = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)
clf_a.fit(X_struct_train_s, y_train)
scores_a_val = clf_a.predict_proba(X_struct_val_s)[:, 1]
auc_a = roc_auc_score(y_val, scores_a_val)
best_a = best_threshold(y_val, scores_a_val)
print(f"val AUC={auc_a:.4f}  best-threshold-on-val: {best_a}")
results["A_structured_logreg"] = {"model": clf_a, "val_auc": auc_a, "val_best": best_a,
                                    "val_scores": scores_a_val, "kind": "struct"}

print("\n=== B. Structured-only Random Forest ===")
clf_b = RandomForestClassifier(n_estimators=300, max_depth=None, class_weight="balanced",
                                 random_state=RANDOM_SEED, n_jobs=-1)
clf_b.fit(X_struct_train, y_train)
scores_b_val = clf_b.predict_proba(X_struct_val)[:, 1]
auc_b = roc_auc_score(y_val, scores_b_val)
best_b = best_threshold(y_val, scores_b_val)
print(f"val AUC={auc_b:.4f}  best-threshold-on-val: {best_b}")
results["B_structured_rf"] = {"model": clf_b, "val_auc": auc_b, "val_best": best_b,
                                "val_scores": scores_b_val, "kind": "struct_raw"}

print("\n=== C. Structured-only HistGradientBoosting ===")
# HGB has no class_weight param in this sklearn version's stable API path we use manually via sample_weight
sw = np.where(y_train == 1, 1.0, (y_train == 1).sum() / max((y_train == 0).sum(), 1))
clf_c = HistGradientBoostingClassifier(random_state=RANDOM_SEED, max_depth=6)
clf_c.fit(X_struct_train, y_train, sample_weight=sw)
scores_c_val = clf_c.predict_proba(X_struct_val)[:, 1]
auc_c = roc_auc_score(y_val, scores_c_val)
best_c = best_threshold(y_val, scores_c_val)
print(f"val AUC={auc_c:.4f}  best-threshold-on-val: {best_c}")
results["C_structured_hgb"] = {"model": clf_c, "val_auc": auc_c, "val_best": best_c,
                                 "val_scores": scores_c_val, "kind": "struct_raw"}

print("\n=== D. Text-only TF-IDF + Logistic Regression ===")
clf_d = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)
clf_d.fit(X_text_train, y_train)
scores_d_val = clf_d.predict_proba(X_text_val)[:, 1]
auc_d = roc_auc_score(y_val, scores_d_val)
best_d = best_threshold(y_val, scores_d_val)
print(f"val AUC={auc_d:.4f}  best-threshold-on-val: {best_d}")
results["D_text_logreg"] = {"model": clf_d, "val_auc": auc_d, "val_best": best_d,
                              "val_scores": scores_d_val, "kind": "text"}

print("\n=== E. Hybrid (struct+text) Logistic Regression ===")
clf_e = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)
clf_e.fit(X_hybrid_train, y_train)
scores_e_val = clf_e.predict_proba(X_hybrid_val)[:, 1]
auc_e = roc_auc_score(y_val, scores_e_val)
best_e = best_threshold(y_val, scores_e_val)
print(f"val AUC={auc_e:.4f}  best-threshold-on-val: {best_e}")
results["E_hybrid_logreg"] = {"model": clf_e, "val_auc": auc_e, "val_best": best_e,
                                "val_scores": scores_e_val, "kind": "hybrid"}

print("\n=== F. Hybrid (struct+text) Linear SVM (calibrated) ===")
base_svm = LinearSVC(class_weight="balanced", random_state=RANDOM_SEED, max_iter=5000)
clf_f = CalibratedClassifierCV(base_svm, method="sigmoid", cv=3)
clf_f.fit(X_hybrid_train, y_train)
scores_f_val = clf_f.predict_proba(X_hybrid_val)[:, 1]
auc_f = roc_auc_score(y_val, scores_f_val)
best_f = best_threshold(y_val, scores_f_val)
print(f"val AUC={auc_f:.4f}  best-threshold-on-val: {best_f}")
results["F_hybrid_linearsvm"] = {"model": clf_f, "val_auc": auc_f, "val_best": best_f,
                                   "val_scores": scores_f_val, "kind": "hybrid"}

print("\n\n=== VALIDATION SUMMARY (sorted by F1 at chosen threshold) ===")
summary_rows = []
for name, r in results.items():
    b = r["val_best"]
    summary_rows.append((name, r["val_auc"], b["f1"], b["precision"], b["recall"], b["fpr"], b["threshold"]))
summary_rows.sort(key=lambda x: -x[2])
for row in summary_rows:
    print(f"{row[0]:22s} AUC={row[1]:.4f}  F1={row[2]:.4f}  P={row[3]:.4f}  R={row[4]:.4f}  FPR={row[5]:.4f}  thr={row[6]:.3f}")

import pickle
with open("/home/claude/halo_ml/data/candidate_results.pkl", "wb") as f:
    pickle.dump({
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "model"} for k, v in results.items()},
    }, f)

# stash fitted objects for the selection step
import joblib
joblib.dump({
    "clf_a": clf_a, "clf_b": clf_b, "clf_c": clf_c, "clf_d": clf_d, "clf_e": clf_e, "clf_f": clf_f,
    "scaler": scaler, "tfidf": tfidf,
}, "/home/claude/halo_ml/data/fitted_candidates.joblib")
print("\nSaved fitted candidates -> /home/claude/halo_ml/data/fitted_candidates.joblib")
