"""
HALO ML — Phase 1 (leakage-safe split).

21,245 of the 37,036 EPVME rows are exact-duplicate template blasts
(same normalized body reused across many email_ids). A naive random
row-level split would put copies of the same template in both train
and test, which inflates test accuracy without the model having
learned anything generalizable. We split on `dedup_group` instead, so
every row sharing a body (or, for empty bodies, a sender+subject) with
another row is guaranteed to land in the same split.

Split target: 70% train / 15% validation / 15% test, by GROUP, then
checked for label balance per split.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

RANDOM_SEED = 42


def group_split(df, seed=RANDOM_SEED, group_col="template_group"):
    groups = df[group_col].values
    y = df["label_bin"].values

    # first: hold out 15% of groups for test
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    trainval_idx, test_idx = next(gss1.split(df, y, groups))

    # second: of the remaining 85%, hold out ~17.65% for validation
    # (0.15 / 0.85 ≈ 0.1765) so val ends up ~15% of the ORIGINAL data
    df_trainval = df.iloc[trainval_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=seed)
    train_sub_idx, val_sub_idx = next(gss2.split(
        df_trainval, df_trainval["label_bin"].values, df_trainval[group_col].values
    ))
    train_idx = trainval_idx[train_sub_idx]
    val_idx = trainval_idx[val_sub_idx]

    # sanity: no group appears in more than one split
    g_train = set(df.iloc[train_idx][group_col])
    g_val = set(df.iloc[val_idx][group_col])
    g_test = set(df.iloc[test_idx][group_col])
    assert not (g_train & g_val), "leak: group in both train and val"
    assert not (g_train & g_test), "leak: group in both train and test"
    assert not (g_val & g_test), "leak: group in both val and test"

    return train_idx, val_idx, test_idx


def summarize_split(df, train_idx, val_idx, test_idx):
    def stats(idx, name):
        sub = df.iloc[idx]
        n = len(sub)
        mal = int(sub["label_bin"].sum())
        leg = n - mal
        print(f"{name:>10}: {n:6d} rows  |  malicious={mal:6d} ({mal/n:5.1%})  legitimate={leg:6d} ({leg/n:5.1%})")

    stats(train_idx, "train")
    stats(val_idx, "val")
    stats(test_idx, "test")


if __name__ == "__main__":
    df = pd.read_pickle("/home/claude/halo_ml/data/combined_raw.pkl")
    train_idx, val_idx, test_idx = group_split(df)
    summarize_split(df, train_idx, val_idx, test_idx)
    np.savez("/home/claude/halo_ml/data/split_indices.npz",
             train=train_idx, val=val_idx, test=test_idx)
    print("Saved -> /home/claude/halo_ml/data/split_indices.npz")
