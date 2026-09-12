"""
HALO ML — finalize & save the production model artifact.

Retrains the selected model (moderate-feature Random Forest) on
TRAIN + VALIDATION combined (standard practice once model selection
and threshold selection are both finished), keeping the untouched
TEST set purely for the reported metrics already computed in
train_conservative.py / the moderate-feature evaluation.
"""
import json
import datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

MODERATE_FEATURES = [
    "url_count", "attachment_count", "has_credential_request", "has_otp_request",
    "has_payment_request", "has_urgency", "auth_headers_present", "html_present", "script_present",
    "from_missing", "reply_to_present", "reply_to_mismatch", "from_local_has_digits",
    "urgency_word_hits", "credential_word_hits", "payment_word_hits",
    "url_extracted_count", "url_ip_based_count", "url_punycode_count",
    "url_domain_mismatch_count", "url_subdomain_heavy_count",
]
THRESHOLD = 0.4108509505814136
RANDOM_SEED = 42

df = pd.read_pickle("/home/claude/halo_ml/data/features.pkl")
idx = np.load("/home/claude/halo_ml/data/split_indices.npz")
trainval_idx = np.concatenate([idx["train"], idx["val"]])
trainval = df.iloc[trainval_idx]

X = trainval[MODERATE_FEATURES].values.astype(float)
y = trainval["label_bin"].values

model = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                 random_state=RANDOM_SEED, n_jobs=-1)
model.fit(X, y)

joblib.dump({"model": model, "features": MODERATE_FEATURES, "threshold": THRESHOLD},
            "/home/claude/halo_ml/models/halo_trust_model_v1.joblib")

with open("/home/claude/halo_ml/data/final_test_metrics.json") as f:
    test_metrics = json.load(f)

metadata = {
    "model_name": "HALO-Trust-RF-v1",
    "model_type": "RandomForestClassifier (scikit-learn), 300 trees, class_weight=balanced",
    "trained_on": "train+validation splits (22466+11277=33743 rows) after model/threshold selection",
    "feature_version": "moderate_v1",
    "features": MODERATE_FEATURES,
    "decision_threshold": THRESHOLD,
    "threshold_selection_method": "F1-optimal among thresholds with validation FPR <= 5%, chosen on the validation set only",
    "training_date": datetime.date.today().isoformat(),
    "dataset": {
        "epvme_rows": 37036,
        "spamassassin_legitimate_rows": 2501,
        "combined_rows": 39537,
        "dedup_groups": 22340,
        "split": {"train": 22466, "val": 11277, "test": 5794},
        "split_method": "GroupShuffleSplit on normalized-body dedup groups (70/15/15 by group), seed=42, "
                          "so no exact/near-duplicate email spans two splits",
    },
    "final_test_metrics": test_metrics,
    "known_limitations": [
        "The legitimate-email class comes entirely from a single 2002 tech mailing-list archive "
        "(SpamAssassin Easy Ham). The malicious class (EPVME) is a mixed corpus including many "
        "automated XSS/injection test payloads. This means the two classes differ in writing genre "
        "and era, not just in maliciousness -- the model can partly separate them via corpus style "
        "rather than phishing mechanics alone.",
        "html_present is the single largest feature (~58% of importance in the shipped model). It is "
        "heavily confounded with source corpus (72.5% of EPVME is HTML vs 0.3% of SpamAssassin ham). "
        "Modern legitimate campus email (LMS notices, official announcements) is frequently HTML too, "
        "so this feature MUST be revalidated against real legitimate HTML campus email before trusting "
        "it in production -- otherwise the model may over-flag legitimate HTML mail.",
        "A raw-text TF-IDF model reached ~99.9% validation F1 but was found (via coefficient inspection) "
        "to be substantially driven by corpus-identifying tokens (years, mailing-list jargon, HTML tag "
        "tokens) rather than phishing content -- that model is NOT shipped for this reason. See README.",
        "A conservative model using ONLY unambiguous phishing-mechanism features (credential/payment/OTP/"
        "urgency signals, sender/reply-to mismatch, URL structure) reached only ~0.86 validation ROC-AUC "
        "on its own -- meaningfully below target -- showing those mechanism features alone are not yet "
        "sufficient on this dataset. The shipped model's 88.4% test accuracy sits between these two, "
        "and should be read as 'partially real signal, partially corpus-era-confounded' rather than a "
        "clean 90%+ result.",
        "Recommended before production use: collect real (or carefully simulated) campus email samples "
        "for the legitimate class -- official notices, LMS mail, T&P Cell announcements -- so the "
        "legitimate class matches the deployment distribution instead of a 2002 mailing-list archive.",
    ],
}
with open("/home/claude/halo_ml/models/halo_trust_model_v1.metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Saved model  -> /home/claude/halo_ml/models/halo_trust_model_v1.joblib")
print("Saved metadata -> /home/claude/halo_ml/models/halo_trust_model_v1.metadata.json")
print(json.dumps(test_metrics, indent=2))
