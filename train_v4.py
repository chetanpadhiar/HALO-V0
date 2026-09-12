"""
HALO ML — v4: remove stylistic/length features proven to be corpus
artifacts, not phishing signal.

Trigger: test_routine_email_does_not_score_extreme caught the v3
structured-only RandomForest scoring a genuinely benign, short email
("Class rescheduled to Friday") at P(malicious)=0.91, driven entirely
by subject_len=27 / body_len=51 -- because EPVME's malicious set is
full of short template/test payloads while SpamAssassin ham is full of
long, quoted mailing-list threads. Short-and-plain is not inherently
suspicious; this is the same corpus-genre problem as html_present,
now with a concrete reproducing example.

Removed: subject_len, body_len, body_len_log, subject_has_exclaim,
subject_is_reply_or_fwd -- all stylistic/length features with no
direct mechanism tying them to phishing risk, as opposed to e.g.
has_payment_request (mechanism) or url_domain_mismatch_count (mechanism).

html_present is KEPT despite being the largest single confound,
because unlike raw length it has at least a plausible causal story
(templated phishing kits are disproportionately HTML) -- but it
remains flagged as the top item under known_limitations and should be
the first thing revalidated against real campus HTML mail.
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from eval_utils import majority_baseline, full_metrics, print_metrics, best_threshold_by_balanced_accuracy

RANDOM_SEED = 42
MIN_MALICIOUS_RECALL = 0.85

FEATURES_V4 = [
    "url_count", "attachment_count",
    "has_credential_request", "has_otp_request", "has_payment_request", "has_urgency",
    "html_present", "script_present", "auth_headers_present",
    "from_missing", "reply_to_present", "reply_to_mismatch", "from_local_has_digits",
    "urgency_word_hits", "credential_word_hits", "payment_word_hits",
    "url_extracted_count", "url_ip_based_count", "url_punycode_count",
    "url_domain_mismatch_count", "url_subdomain_heavy_count",
]
EXCLUDED_V4 = ["subject_len", "body_len", "body_len_log", "subject_has_exclaim", "subject_is_reply_or_fwd"]

df = pd.read_pickle("/home/claude/halo_ml/data/features.pkl")
idx = np.load("/home/claude/halo_ml/data/split_indices.npz")
train, val, test = df.iloc[idx["train"]], df.iloc[idx["val"]], df.iloc[idx["test"]]
y_train, y_val, y_test = train["label_bin"].values, val["label_bin"].values, test["label_bin"].values
X_train = train[FEATURES_V4].values.astype(float)
X_val = val[FEATURES_V4].values.astype(float)
X_test = test[FEATURES_V4].values.astype(float)


class _HGBWrapper:
    def __init__(self):
        self.model = HistGradientBoostingClassifier(random_state=RANDOM_SEED, max_depth=6)
    def fit(self, X, y):
        pos, neg = (y == 1).sum(), (y == 0).sum()
        sw = np.where(y == 1, 1.0, pos / max(neg, 1))
        self.model.fit(X, y, sample_weight=sw)
        return self
    def predict_proba(self, X):
        return self.model.predict_proba(X)


print("=== RandomForest, corpus-artifact features removed ===")
rf = RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)
rf.fit(X_train, y_train)
val_scores_rf = rf.predict_proba(X_val)[:, 1]
best_rf = best_threshold_by_balanced_accuracy(y_val, val_scores_rf, min_malicious_recall=MIN_MALICIOUS_RECALL)
print_metrics(best_rf, "RF-v4 on VALIDATION")
imp = sorted(zip(FEATURES_V4, rf.feature_importances_), key=lambda x: -x[1])
for f, v in imp:
    print(f"  {f:28s} {v:.4f}")

print("\n=== HistGradientBoosting, corpus-artifact features removed ===")
hgb = _HGBWrapper()
hgb.fit(X_train, y_train)
val_scores_hgb = hgb.predict_proba(X_val)[:, 1]
best_hgb = best_threshold_by_balanced_accuracy(y_val, val_scores_hgb, min_malicious_recall=MIN_MALICIOUS_RECALL)
print_metrics(best_hgb, "HGB-v4 on VALIDATION")

# regression check: the exact case that broke v3
from features import extract_structured_features
benign_row = {"subject": "Class rescheduled to Friday",
              "body_text": "Hi all, our Monday class moves to Friday same time.",
              "from_address": "prof.sharma@college.edu", "reply_to": "prof.sharma@college.edu",
              "url_count": 0, "attachment_count": 0}
f = extract_structured_features(benign_row)
x = np.array([f[k] for k in FEATURES_V4], dtype=float).reshape(1, -1)
print(f"\nRegression check (the email that broke v3): P(malicious) = {rf.predict_proba(x)[0,1]:.4f} (RF), {hgb.predict_proba(x)[0,1]:.4f} (HGB)")

joblib.dump({"rf": rf, "hgb": hgb, "features": FEATURES_V4},
            "/home/claude/halo_ml/data/v4_candidates.joblib")
