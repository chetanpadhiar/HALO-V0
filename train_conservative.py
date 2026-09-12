"""
HALO ML — conservative candidate.

Feature-importance inspection of the full structured Random Forest
showed html_present, subject/body length, and reply-style features
dominating (~70% of importance combined) -- all of which plausibly
reflect "which of the two source corpora is this from" (a 2002 tech
mailing-list archive vs. a mass-produced phishing/test-payload corpus)
rather than phishing mechanics specifically.

This script retrains on a restricted feature set containing ONLY
features that are semantically about how phishing operates (identity
mismatch, credential/payment/OTP/urgency requests, URL structure),
dropping every feature whose importance is plausibly genre/style
rather than intent. This is expected to score lower -- that's the
point: it's the number we can actually defend.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

CONSERVATIVE_FEATURES = [
    "url_count", "attachment_count",
    "has_credential_request", "has_otp_request", "has_payment_request", "has_urgency",
    "auth_headers_present",
    "from_missing", "reply_to_present", "reply_to_mismatch", "from_local_has_digits",
    "urgency_word_hits", "credential_word_hits", "payment_word_hits",
    "url_extracted_count", "url_ip_based_count", "url_punycode_count",
    "url_domain_mismatch_count", "url_subdomain_heavy_count",
]
EXCLUDED_AS_LIKELY_CORPUS_ARTIFACT = [
    "html_present", "script_present", "subject_len", "body_len",
    "body_len_log", "subject_has_exclaim", "subject_is_reply_or_fwd",
]

RANDOM_SEED = 42
df = pd.read_pickle("/home/claude/halo_ml/data/features.pkl")
idx = np.load("/home/claude/halo_ml/data/split_indices.npz")
train_idx, val_idx, test_idx = idx["train"], idx["val"], idx["test"]
train, val, test = df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]
y_train, y_val, y_test = train["label_bin"].values, val["label_bin"].values, test["label_bin"].values

X_train = train[CONSERVATIVE_FEATURES].values.astype(float)
X_val = val[CONSERVATIVE_FEATURES].values.astype(float)
X_test = test[CONSERVATIVE_FEATURES].values.astype(float)


def eval_at(y_true, scores, threshold):
    preds = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    return {"threshold": float(threshold), "accuracy": accuracy_score(y_true, preds),
            "precision": precision_score(y_true, preds, zero_division=0),
            "recall": recall_score(y_true, preds, zero_division=0),
            "f1": f1_score(y_true, preds, zero_division=0),
            "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
            "fnr": fn / (fn + tp) if (fn + tp) else 0.0,
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def best_threshold(y_true, scores):
    cands = np.unique(scores)
    cands = cands[np.linspace(0, len(cands) - 1, min(400, len(cands))).astype(int)]
    rows = [eval_at(y_true, scores, t) for t in cands]
    low_fpr = [r for r in rows if r["fpr"] <= 0.05]
    pool = low_fpr if low_fpr else rows
    return max(pool, key=lambda r: r["f1"])


print("=== Conservative Random Forest (mechanism-only features) ===")
rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)
rf.fit(X_train, y_train)
s_val = rf.predict_proba(X_val)[:, 1]
print("val AUC:", roc_auc_score(y_val, s_val))
print("val best:", best_threshold(y_val, s_val))
imp_order = np.argsort(rf.feature_importances_)[::-1]
print("\nFeature importances:")
for i in imp_order:
    print(f"  {CONSERVATIVE_FEATURES[i]:28s} {rf.feature_importances_[i]:.4f}")

print("\n=== Conservative Logistic Regression (mechanism-only features) ===")
scaler = StandardScaler().fit(X_train)
lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)
lr.fit(scaler.transform(X_train), y_train)
s_val_lr = lr.predict_proba(scaler.transform(X_val))[:, 1]
print("val AUC:", roc_auc_score(y_val, s_val_lr))
print("val best:", best_threshold(y_val, s_val_lr))
print("\nCoefficients:")
for name, c in sorted(zip(CONSERVATIVE_FEATURES, lr.coef_[0]), key=lambda x: -abs(x[1])):
    print(f"  {name:28s} {c:+.3f}")

import joblib
joblib.dump({"rf": rf, "lr": lr, "scaler": scaler, "features": CONSERVATIVE_FEATURES},
            "/home/claude/halo_ml/data/conservative_candidates.joblib")
