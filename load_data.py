"""
HALO ML — Phase 1: Data loading & normalization.

Both source files already share one schema (email_id, source_dataset,
source_file, label, from_address, to_address, reply_to, subject, date,
body_text, url_count, attachment_count, has_credential_request,
has_otp_request, has_payment_request, has_urgency, html_present,
script_present, authentication_headers_present, notes).

pandas' C/python engines choke on this file: HTML fragments embedded in
body_text contain long runs of doubled quote characters that produce an
ambiguous parse for those engines. The stdlib csv module parses it
correctly, so we load with that and hand the result to pandas.
"""
import csv
import re
import hashlib
import pandas as pd

csv.field_size_limit(10_000_000)

BOOL_COLS = [
    "has_credential_request", "has_otp_request", "has_payment_request",
    "has_urgency", "html_present", "script_present",
    "authentication_headers_present",
]
INT_COLS = ["url_count", "attachment_count"]


def _load_raw(path):
    rows = []
    bad = 0
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        ncols = len(header)
        for row in reader:
            if len(row) != ncols:
                bad += 1
                continue
            rows.append(dict(zip(header, row)))
    return rows, bad


def _to_bool(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def _to_int(v, default=0):
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return default


def _extract_source_zip(notes):
    m = re.search(r"source_zip=([^;]+)", notes or "")
    return m.group(1) if m else None


def _norm_text(s):
    """Canonicalize text for duplicate detection: lowercase, collapse whitespace."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def load_dataset(epvme_path, spam_path, verbose=True):
    ep_rows, ep_bad = _load_raw(epvme_path)
    sp_rows, sp_bad = _load_raw(spam_path)

    for r in ep_rows:
        r["_source_zip"] = _extract_source_zip(r.get("notes", ""))
    for r in sp_rows:
        r["_source_zip"] = "spamassassin_easy_ham"

    all_rows = ep_rows + sp_rows
    df = pd.DataFrame(all_rows)

    for c in BOOL_COLS:
        df[c] = df[c].apply(_to_bool)
    for c in INT_COLS:
        df[c] = df[c].apply(lambda v: _to_int(v, 0))

    df["label_bin"] = (df["label"].str.strip().str.lower() == "malicious").astype(int)
    df["body_norm"] = df["body_text"].apply(_norm_text)
    df["subject_norm"] = df["subject"].apply(_norm_text)
    df["body_hash"] = df["body_norm"].apply(
        lambda t: hashlib.sha1(t.encode("utf-8")).hexdigest() if t else "EMPTY_" + hashlib.sha1(str(id(t)).encode()).hexdigest()[:8]
    )
    # dedupe grouping key: identical normalized body -> same group, so a
    # group can never be split across train/val/test.
    df["dedup_group"] = df["body_hash"]
    # rows with a genuinely empty body get their own singleton group
    # (their body_hash is already unique above), grouped instead by
    # sender+subject so template blasts with empty bodies still group.
    empty_mask = df["body_norm"] == ""
    df.loc[empty_mask, "dedup_group"] = (
        "EMPTYBODY_" + df.loc[empty_mask, "from_address"].fillna("") + "|" + df.loc[empty_mask, "subject_norm"]
    ).apply(lambda t: hashlib.sha1(t.encode("utf-8")).hexdigest())

    if verbose:
        print(f"EPVME: loaded {len(ep_rows)} rows, skipped {ep_bad} malformed rows")
        print(f"SpamAssassin: loaded {len(sp_rows)} rows, skipped {sp_bad} malformed rows")
        print(f"Combined: {len(df)} rows")
        print(df["label_bin"].value_counts().rename({0: "legitimate", 1: "malicious"}))
        print(f"Unique dedup groups: {df['dedup_group'].nunique()} (of {len(df)} rows)")

    return df


if __name__ == "__main__":
    df = load_dataset(
        "/mnt/user-data/uploads/epvme_emails.csv",
        "/mnt/user-data/uploads/spamassassin_legitimate.csv",
    )
    df.to_pickle("/home/claude/halo_ml/data/combined_raw.pkl")
    print("Saved -> /home/claude/halo_ml/data/combined_raw.pkl")
