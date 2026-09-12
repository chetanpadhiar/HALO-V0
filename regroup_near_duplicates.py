"""
HALO ML — near-duplicate regrouping.

The original dedup_group used an exact normalized-body hash, which
correctly caught the 21,245 literal duplicate EPVME rows. But template
phishing kits often vary a name, amount, or link per recipient while
reusing the same boilerplate -- those would NOT hash identically and
could still leak a template family across train/val/test. This script
builds a coarser "template_group" by stripping numbers, URLs, and
email addresses before hashing, so substituted-variable near-duplicates
group together too, and uses THAT as the split key going forward.
"""
import re
import hashlib
import pandas as pd

URL_RE = re.compile(r"https?://\S+")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+")
DIGIT_RE = re.compile(r"\d+")
WS_RE = re.compile(r"\s+")


def template_normalize(text):
    t = (text or "").lower()
    t = URL_RE.sub(" <url> ", t)
    t = EMAIL_RE.sub(" <email> ", t)
    t = DIGIT_RE.sub("#", t)
    t = WS_RE.sub(" ", t).strip()
    return t


if __name__ == "__main__":
    df = pd.read_pickle("/home/claude/halo_ml/data/combined_raw.pkl")

    df["template_norm"] = df["body_norm"].apply(template_normalize)
    empty_mask = df["body_norm"] == ""

    df["template_group"] = df["template_norm"].apply(lambda t: "T_" + hashlib.sha1(t.encode("utf-8")).hexdigest())
    # empty-body rows: group by sender+subject template instead (same logic as before)
    df.loc[empty_mask, "template_group"] = (
        "EMPTYBODY_" + df.loc[empty_mask, "from_address"].fillna("") + "|" +
        df.loc[empty_mask, "subject_norm"].apply(template_normalize)
    ).apply(lambda t: "T_" + hashlib.sha1(t.encode("utf-8")).hexdigest())

    n_before = df["dedup_group"].nunique()
    n_after = df["template_group"].nunique()
    print(f"Exact-duplicate groups (body_hash):        {n_before}")
    print(f"Template groups (numbers/URLs/emails-normalized): {n_after}")
    print(f"Additional rows pulled together by template normalization: {n_before - n_after} fewer groups")

    # sanity: show a few template groups that contain >1 exact-dup group,
    # to prove this is actually catching near-duplicates, not just
    # relabeling the same groups
    grp_sizes = df.groupby("template_group")["dedup_group"].nunique()
    multi = grp_sizes[grp_sizes > 1].sort_values(ascending=False)
    print(f"\nTemplate groups spanning >1 distinct exact-dup group: {len(multi)}")
    if len(multi):
        example_tg = multi.index[0]
        examples = df[df["template_group"] == example_tg][["email_id", "subject", "label"]].head(4)
        print("Example near-duplicate family:")
        print(examples.to_string(index=False))

    df.to_pickle("/home/claude/halo_ml/data/combined_raw.pkl")
    print("\nUpdated -> combined_raw.pkl (added template_group column)")
