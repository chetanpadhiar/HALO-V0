"""
HALO ML — Phase 2, v2: retrain text-based candidates on sanitized text
(see text_clean.py) after diagnosing a corpus-source leak in v1.
Structured-only candidates (A/B/C) are carried over unchanged from v1
since they don't use free text.
"""
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from features import STRUCTURED_FEATURE_NAMES
from text_clean import clean_for_text_model

RANDOM_SEED = 42

df = pd.read_pickle("/home/claude/halo_ml/data/features.pkl")
idx = np.load("/home/claude/halo_ml/data/split_indices.npz")
train_idx, val_idx, test_idx = idx["train"], idx["val"], idx["test"]

train = df.iloc[train_idx].reset_index(drop=True)
val = df.iloc[val_idx].reset_index(drop=True)
test = df.iloc[test_idx].reset_index(drop=True)
y_train, y_val, y_test = train["label_bin"].values, val["label_bin"].values, test["label_bin"].values

print("Cleaning text (stripping HTML markup, dates/years, corpus jargon)...")
train_text_clean = train["text"].apply(clean_for_text_model)
val_text_clean = val["text"].apply(clean_for_text_model)
test_text_clean = test["text"].apply(clean_for_text_model)

cand = joblib.load("/home/claude/halo_ml/data/fitted_candidates.joblib")
scaler = cand["scaler"]
X_struct_train_s = scaler.transform(train[STRUCTURED_FEATURE_NAMES].values.astype(float))
X_struct_val_s = scaler.transform(val[STRUCTURED_FEATURE_NAMES].values.astype(float))
X_struct_test_s = scaler.transform(test[STRUCTURED_FEATURE_NAMES].values.astype(float))

# require the token to appear in at least 3 documents AND contain a
# letter (drops leftover stray numerics/punctuation fragments)
tfidf2 = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=3,
                           sublinear_tf=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b")
X_text_train = tfidf2.fit_transform(train_text_clean)
X_text_val = tfidf2.transform(val_text_clean)
X_text_test = tfidf2.transform(test_text_clean)

X_hybrid_train = hstack([X_text_train, csr_matrix(X_struct_train_s)]).tocsr()
X_hybrid_val = hstack([X_text_val, csr_matrix(X_struct_val_s)]).tocsr()
X_hybrid_test = hstack([X_text_test, csr_matrix(X_struct_test_s)]).tocsr()


def eval_at(y_true, scores, threshold):
    preds = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold), "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "fnr": fn / (fn + tp) if (fn + tp) else 0.0,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def best_threshold(y_true, scores):
    candidates = np.unique(scores)
    candidates = candidates[np.linspace(0, len(candidates) - 1, min(400, len(candidates))).astype(int)]
    rows = [eval_at(y_true, scores, t) for t in candidates]
    low_fpr = [r for r in rows if r["fpr"] <= 0.05]
    pool = low_fpr if low_fpr else rows
    return max(pool, key=lambda r: r["f1"])


results = {}

print("\n=== D2. Text-only (sanitized) TF-IDF + Logistic Regression ===")
clf_d2 = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)
clf_d2.fit(X_text_train, y_train)
s = clf_d2.predict_proba(X_text_val)[:, 1]
auc = roc_auc_score(y_val, s); best = best_threshold(y_val, s)
print(f"val AUC={auc:.4f}  {best}")
results["D2_text_logreg_clean"] = {"model": clf_d2, "val_auc": auc, "val_best": best}

print("\n=== E2. Hybrid (sanitized text + struct) Logistic Regression ===")
clf_e2 = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)
clf_e2.fit(X_hybrid_train, y_train)
s = clf_e2.predict_proba(X_hybrid_val)[:, 1]
auc = roc_auc_score(y_val, s); best = best_threshold(y_val, s)
print(f"val AUC={auc:.4f}  {best}")
results["E2_hybrid_logreg_clean"] = {"model": clf_e2, "val_auc": auc, "val_best": best}

print("\n=== F2. Hybrid (sanitized text + struct) Linear SVM (calibrated) ===")
base_svm = LinearSVC(class_weight="balanced", random_state=RANDOM_SEED, max_iter=5000)
clf_f2 = CalibratedClassifierCV(base_svm, method="sigmoid", cv=3)
clf_f2.fit(X_hybrid_train, y_train)
s = clf_f2.predict_proba(X_hybrid_val)[:, 1]
auc = roc_auc_score(y_val, s); best = best_threshold(y_val, s)
print(f"val AUC={auc:.4f}  {best}")
results["F2_hybrid_linearsvm_clean"] = {"model": clf_f2, "val_auc": auc, "val_best": best}

print("\n\n=== Inspect top tokens again (sanitized) — is the leak gone? ===")
fnames = np.array(tfidf2.get_feature_names_out())
coefs = clf_d2.coef_[0]
print("-- toward MALICIOUS --")
for i in np.argsort(coefs)[-20:][::-1]:
    print(f"  {fnames[i]:25s} {coefs[i]:.3f}")
print("-- toward LEGITIMATE --")
for i in np.argsort(coefs)[:20]:
    print(f"  {fnames[i]:25s} {coefs[i]:.3f}")

joblib.dump({"tfidf2": tfidf2, "clf_d2": clf_d2, "clf_e2": clf_e2, "clf_f2": clf_f2},
            "/home/claude/halo_ml/data/fitted_candidates_v2.joblib")

import pickle
with open("/home/claude/halo_ml/data/candidate_results_v2.pkl", "wb") as f:
    pickle.dump({k: {kk: vv for kk, vv in v.items() if kk != "model"} for k, v in results.items()}, f)
print("\nSaved v2 candidates.")
