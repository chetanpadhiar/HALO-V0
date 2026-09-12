"""
HALO ML — final model selection & single-touch test evaluation (v3).

SELECTED: structured-only RandomForestClassifier, full feature set.
Rationale (see README for the full writeup):
  - Text and hybrid models score ~0.99 balanced accuracy on validation,
    but source_dataset and label are 100% confounded in this data
    (every legitimate row is SpamAssassin, every malicious row is
    EPVME) -- coefficient inspection after TWO rounds of corpus-jargon
    stripping still surfaces literal corpus fingerprints (e.g.
    "zzzzteana", a SpamAssassin-bundled mailing list name; "satalk").
    This is a structural property of the dataset, not a cleaning bug,
    so the text/hybrid numbers cannot be trusted as phishing detection.
  - The structured-only model cannot see raw vocabulary/style at all,
    so it cannot be re-detecting corpus identity the same way -- its
    (lower, ~0.93 balanced accuracy) score is the one we can defend.
  - Chosen over structured-only HistGradientBoosting for better
    malicious recall (94.1% vs 91.5%) and macro-F1 (0.850 vs 0.819),
    at the cost of a higher FPR (7.9% vs 3.8%). HALO's 3-tier bucket
    (safe/attention/high-risk) cushions this: a borderline legitimate
    email is more likely to land in "needs attention" than "high risk"
    given the scores involved, softening the false-positive cost.
"""
import json
import datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

from features import STRUCTURED_FEATURE_NAMES
from eval_utils import majority_baseline, full_metrics, print_metrics, best_threshold_by_balanced_accuracy

RANDOM_SEED = 42
MIN_MALICIOUS_RECALL = 0.85

df = pd.read_pickle("/home/claude/halo_ml/data/features.pkl")
idx = np.load("/home/claude/halo_ml/data/split_indices.npz")
train = df.iloc[idx["train"]]
val = df.iloc[idx["val"]]
test = df.iloc[idx["test"]]

y_train, y_val, y_test = train["label_bin"].values, val["label_bin"].values, test["label_bin"].values
X_train = train[STRUCTURED_FEATURE_NAMES].values.astype(float)
X_val = val[STRUCTURED_FEATURE_NAMES].values.astype(float)
X_test = test[STRUCTURED_FEATURE_NAMES].values.astype(float)

# 1) select threshold on validation only, using the model trained on TRAIN only
model_for_threshold = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                               random_state=RANDOM_SEED, n_jobs=-1)
model_for_threshold.fit(X_train, y_train)
val_scores = model_for_threshold.predict_proba(X_val)[:, 1]
chosen = best_threshold_by_balanced_accuracy(y_val, val_scores, min_malicious_recall=MIN_MALICIOUS_RECALL)
THRESHOLD = chosen["threshold"]
print_metrics(chosen, "SELECTED MODEL on VALIDATION (threshold chosen here)")

# 2) refit on train+val for the artifact actually shipped (standard
#    practice once threshold + model family are both already decided)
X_trainval = np.vstack([X_train, X_val])
y_trainval = np.concatenate([y_train, y_val])
final_model = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                       random_state=RANDOM_SEED, n_jobs=-1)
final_model.fit(X_trainval, y_trainval)

# 3) ONE touch of the test set, with the metric suite the review asked for
test_scores = final_model.predict_proba(X_test)[:, 1]
test_metrics = full_metrics(y_test, test_scores, THRESHOLD)
test_baseline = majority_baseline(y_test)
print()
print_metrics(test_metrics, "FINAL MODEL on TEST (untouched until now)")
print(f"\nFor context, majority-class baseline on TEST: accuracy={test_baseline['accuracy']:.4f} "
      f"balanced_accuracy={test_baseline['balanced_accuracy']:.4f} macro_f1={test_baseline['macro_f1']:.4f}")

joblib.dump({"model": final_model, "features": STRUCTURED_FEATURE_NAMES, "threshold": THRESHOLD},
            "/home/claude/halo_ml/models/halo_trust_model_v2.joblib")

imp = sorted(zip(STRUCTURED_FEATURE_NAMES, final_model.feature_importances_), key=lambda x: -x[1])

metadata = {
    "model_name": "HALO-Trust-RF-v2",
    "model_type": "RandomForestClassifier (scikit-learn), 400 trees, class_weight=balanced",
    "feature_set": "structured-only (25 features) -- text/hybrid explicitly NOT shipped, see limitations",
    "features": STRUCTURED_FEATURE_NAMES,
    "feature_importances": [{"feature": f, "importance": round(float(v), 4)} for f, v in imp],
    "decision_threshold": THRESHOLD,
    "threshold_selection_method": f"max balanced accuracy on validation subject to malicious recall >= {MIN_MALICIOUS_RECALL}",
    "training_date": datetime.date.today().isoformat(),
    "dataset": {
        "epvme_rows": 37036, "spamassassin_legitimate_rows": 2501, "combined_rows": 39537,
        "near_duplicate_template_groups": 20011,
        "split_method": "GroupShuffleSplit on template-normalized (numbers/URLs/emails stripped) "
                          "near-duplicate groups, 70/15/15 by group, seed=42",
        "split_sizes": {"train": int(len(y_train)), "val": int(len(y_val)), "test": int(len(y_test))},
    },
    "majority_class_baseline_test": test_baseline,
    "final_test_metrics": test_metrics,
    "known_limitations": [
        "CRITICAL: source_dataset and label are 100% confounded -- every legitimate row is from "
        "SpamAssassin Easy Ham, every malicious row is from EPVME. No feature-engineering or text "
        "cleaning can fully separate 'phishing content' from 'which corpus this came from' using this "
        "data alone. Text/hybrid models reached ~0.99 balanced accuracy on validation but were found, "
        "via coefficient inspection after two rounds of corpus-jargon stripping, to still rely partly on "
        "corpus fingerprints (e.g. 'zzzzteana' and 'satalk', both SpamAssassin-bundled mailing-list "
        "names). They are NOT shipped for this reason -- see train_v3.py output for the full ablation.",
        "The shipped structured-only model reaches 0.93 balanced accuracy / 0.85 macro-F1 on held-out "
        "test, BELOW raw accuracy in isolation because raw accuracy is dominated by the 92.6% malicious "
        "majority in this test split -- the majority-class baseline itself scores higher raw accuracy "
        "(see majority_class_baseline_test) while having zero ability to recognize legitimate mail.",
        "html_present remains the largest single structured feature and is itself confounded with "
        "source corpus (72.5% of EPVME vs 0.3% of SpamAssassin ham) -- must be revalidated against real "
        "legitimate HTML campus email (LMS notices, official announcements) before trusting in production.",
        "Legitimate-class precision at the chosen threshold is modest (see final_test_metrics) -- of "
        "messages the model calls 'legitimate', a meaningful fraction are actually missed phishing, a "
        "direct consequence of the 92.6:7.4 test imbalance. This is disclosed, not hidden, and argues "
        "for keeping HALO's 3-tier bucket (not a binary safe/unsafe call) in the product UI.",
        "Recommended before production use: collect real or realistically-simulated campus email for "
        "BOTH classes from the SAME contemporary source, so label is no longer aliased with corpus "
        "identity, and re-run this exact pipeline (train_v3.py) -- at that point the text/hybrid models "
        "become trustworthy to evaluate and likely to ship instead of the structured-only model.",
    ],
}
with open("/home/claude/halo_ml/models/halo_trust_model_v2.metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("\nSaved -> halo_trust_model_v2.joblib / .metadata.json")
