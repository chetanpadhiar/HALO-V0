"""
HALO ML — Phase 2 v3: rigorous comparison.

Fixes applied vs. the previous pass, per review:
  - Report the majority-class baseline alongside every result.
  - Judge models on balanced accuracy / macro-F1 / PR-AUC / per-class
    recall, not raw accuracy (misleading under 14.8:1 imbalance).
  - Split on template-normalized near-duplicate groups (not just exact
    duplicates) so substituted-variable phishing templates can't leak.
  - Handle imbalance ONLY inside training: class_weight='balanced' vs.
    controlled downsampling of the malicious class, compared head to
    head. The val/test class balance is left exactly as collected.
  - Text/structured/hybrid ablation to see what's actually carrying
    the signal.
  - Threshold chosen on validation only, by balanced accuracy, with a
    minimum-malicious-recall floor -- not by raw accuracy or F1 alone.
"""
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from features import STRUCTURED_FEATURE_NAMES
from text_clean import clean_for_text_model
from eval_utils import majority_baseline, full_metrics, print_metrics, best_threshold_by_balanced_accuracy

RANDOM_SEED = 42
MIN_MALICIOUS_RECALL = 0.85  # operating-point floor: HALO must still catch most phishing

df = pd.read_pickle("/home/claude/halo_ml/data/features.pkl")
idx = np.load("/home/claude/halo_ml/data/split_indices.npz")
train = df.iloc[idx["train"]].reset_index(drop=True)
val = df.iloc[idx["val"]].reset_index(drop=True)
test = df.iloc[idx["test"]].reset_index(drop=True)
y_train, y_val, y_test = train["label_bin"].values, val["label_bin"].values, test["label_bin"].values

print("=" * 70)
print("MAJORITY-CLASS BASELINE (context for every result below)")
print("=" * 70)
for name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
    b = majority_baseline(y)
    print(f"{name:6s}: always predict '{b['majority_class']}' -> "
          f"accuracy={b['accuracy']:.4f}  balanced_accuracy={b['balanced_accuracy']:.4f}  macro_f1={b['macro_f1']:.4f}")
print()

# ---------- text (cleaned) ----------
print("Cleaning text...")
train_subj = train["subject_text"].apply(clean_for_text_model)
train_body = train["body_text_raw"].apply(clean_for_text_model)
val_subj = val["subject_text"].apply(clean_for_text_model)
val_body = val["body_text_raw"].apply(clean_for_text_model)
test_subj = test["subject_text"].apply(clean_for_text_model)
test_body = test["body_text_raw"].apply(clean_for_text_model)

WORD_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z']+\b"
subj_vec = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2, token_pattern=WORD_PATTERN)
body_word_vec = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), min_df=3, sublinear_tf=True, token_pattern=WORD_PATTERN)
body_char_vec = TfidfVectorizer(max_features=8000, ngram_range=(3, 5), min_df=3, analyzer="char_wb")

X_subj_train = subj_vec.fit_transform(train_subj); X_subj_val = subj_vec.transform(val_subj); X_subj_test = subj_vec.transform(test_subj)
X_bw_train = body_word_vec.fit_transform(train_body); X_bw_val = body_word_vec.transform(val_body); X_bw_test = body_word_vec.transform(test_body)
X_bc_train = body_char_vec.fit_transform(train_body); X_bc_val = body_char_vec.transform(val_body); X_bc_test = body_char_vec.transform(test_body)

X_text_train = hstack([X_subj_train, X_bw_train, X_bc_train]).tocsr()
X_text_val = hstack([X_subj_val, X_bw_val, X_bc_val]).tocsr()
X_text_test = hstack([X_subj_test, X_bw_test, X_bc_test]).tocsr()
print(f"Text feature dims: subject={X_subj_train.shape[1]}  body_word={X_bw_train.shape[1]}  body_char={X_bc_train.shape[1]}  total={X_text_train.shape[1]}")

# ---------- structured (full feature set; ablation will show what matters) ----------
X_struct_train_raw = train[STRUCTURED_FEATURE_NAMES].values.astype(float)
X_struct_val_raw = val[STRUCTURED_FEATURE_NAMES].values.astype(float)
X_struct_test_raw = test[STRUCTURED_FEATURE_NAMES].values.astype(float)
scaler = StandardScaler().fit(X_struct_train_raw)
X_struct_train = scaler.transform(X_struct_train_raw)
X_struct_val = scaler.transform(X_struct_val_raw)
X_struct_test = scaler.transform(X_struct_test_raw)

X_hybrid_train = hstack([X_text_train, csr_matrix(X_struct_train)]).tocsr()
X_hybrid_val = hstack([X_text_val, csr_matrix(X_struct_val)]).tocsr()
X_hybrid_test = hstack([X_text_test, csr_matrix(X_struct_test)]).tocsr()

# ---------- imbalance handling: controlled downsampling of TRAINING malicious only ----------
def downsample_train(X, y, ratio, seed=RANDOM_SEED):
    """ratio = desired malicious:legitimate ratio in the training data
    only. Downsamples the majority (malicious) class; never touches
    val/test, and never duplicates the minority class."""
    rng = np.random.RandomState(seed)
    mal_idx = np.where(y == 1)[0]
    leg_idx = np.where(y == 0)[0]
    n_keep_mal = min(len(mal_idx), int(len(leg_idx) * ratio))
    keep_mal = rng.choice(mal_idx, size=n_keep_mal, replace=False)
    keep = np.concatenate([keep_mal, leg_idx])
    rng.shuffle(keep)
    return X[keep], y[keep]


results = {}


class _HGBWrapper:
    """Thin wrapper computing a class-balanced sample_weight from
    whatever y it's actually fit on (works for full or downsampled
    training data alike), since HistGradientBoostingClassifier in this
    sklearn version has no class_weight= param."""
    def __init__(self):
        self.model = HistGradientBoostingClassifier(random_state=RANDOM_SEED, max_depth=6)

    def fit(self, X, y):
        pos, neg = (y == 1).sum(), (y == 0).sum()
        sw = np.where(y == 1, 1.0, pos / max(neg, 1))
        self.model.fit(X, y, sample_weight=sw)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def run_candidate(name, X_tr, y_tr, X_va, model_ctor):
    model = model_ctor()
    model.fit(X_tr, y_tr)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_va)[:, 1]
    else:
        scores = model.decision_function(X_va)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    best = best_threshold_by_balanced_accuracy(y_val, scores, min_malicious_recall=MIN_MALICIOUS_RECALL)
    print_metrics(best, name)
    results[name] = {"model": model, "val_scores": scores, "val_best": best}
    return model, scores, best


print("\n" + "=" * 70)
print("CANDIDATE MODELS -- validation set, threshold chosen on validation only")
print("=" * 70)

# --- text only, class_weight balanced ---
run_candidate("TEXT-ONLY LogReg (class_weight=balanced)", X_text_train, y_train, X_text_val,
              lambda: LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_SEED))

run_candidate("TEXT-ONLY LinearSVM (class_weight=balanced, calibrated)", X_text_train, y_train, X_text_val,
              lambda: CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=RANDOM_SEED, max_iter=5000), method="sigmoid", cv=3))

# --- structured only ---
run_candidate("STRUCTURED-ONLY RandomForest (class_weight=balanced)", X_struct_train_raw, y_train, X_struct_val_raw,
              lambda: RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1))

run_candidate("STRUCTURED-ONLY HistGradientBoosting (sample_weight)", X_struct_train_raw, y_train, X_struct_val_raw,
              lambda: _HGBWrapper())

# --- hybrid, class_weight balanced ---
run_candidate("HYBRID LogReg (class_weight=balanced)", X_hybrid_train, y_train, X_hybrid_val,
              lambda: LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_SEED))

run_candidate("HYBRID LinearSVM (class_weight=balanced, calibrated)", X_hybrid_train, y_train, X_hybrid_val,
              lambda: CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=RANDOM_SEED, max_iter=5000), method="sigmoid", cv=3))

# --- hybrid, controlled downsampling instead of class_weight, at two ratios ---
for ratio in [3.0, 5.0]:
    X_ds, y_ds = downsample_train(X_hybrid_train, y_train, ratio)
    run_candidate(f"HYBRID LogReg (downsampled train, malicious:legit={ratio:.0f}:1)", X_ds, y_ds, X_hybrid_val,
                  lambda: LogisticRegression(max_iter=3000, random_state=RANDOM_SEED))

joblib.dump({"results": {k: {kk: vv for kk, vv in v.items() if kk != "model"} for k, v in results.items()}},
            "/home/claude/halo_ml/data/v3_results_summary.joblib")
joblib.dump({k: v["model"] for k, v in results.items()},
            "/home/claude/halo_ml/data/v3_fitted_models.joblib")
joblib.dump({"subj_vec": subj_vec, "body_word_vec": body_word_vec, "body_char_vec": body_char_vec, "scaler": scaler},
            "/home/claude/halo_ml/data/v3_vectorizers.joblib")

print("\n" + "=" * 70)
print("RANKED BY BALANCED ACCURACY (validation)")
print("=" * 70)
ranked = sorted(results.items(), key=lambda kv: -kv[1]["val_best"]["balanced_accuracy"])
for name, r in ranked:
    b = r["val_best"]
    print(f"{name:55s} bal_acc={b['balanced_accuracy']:.4f}  macro_f1={b['macro_f1']:.4f}  "
          f"PR-AUC={b['pr_auc']:.4f}  malicious_recall={b['malicious']['recall']:.4f}  legit_recall={b['legitimate']['recall']:.4f}")
