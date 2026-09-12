import pandas as pd
from features import extract_structured_features, extract_text, STRUCTURED_FEATURE_NAMES

if __name__ == "__main__":
    df = pd.read_pickle("/home/claude/halo_ml/data/combined_raw.pkl")

    print(f"Extracting structured features for {len(df)} rows...")
    struct_records = df.to_dict("records")
    struct_feats = [extract_structured_features(r) for r in struct_records]
    struct_df = pd.DataFrame(struct_feats, columns=STRUCTURED_FEATURE_NAMES, index=df.index)

    text_series = df.apply(extract_text, axis=1)

    out = pd.concat([
        df[["email_id", "label_bin", "dedup_group", "template_group", "source_dataset"]],
        struct_df,
    ], axis=1)
    out["subject_text"] = df["subject"].fillna("")
    out["body_text_raw"] = df["body_text"].fillna("")
    out["text"] = text_series

    out.to_pickle("/home/claude/halo_ml/data/features.pkl")
    print("Saved -> /home/claude/halo_ml/data/features.pkl")
    print(out.shape)
    print(out.head(3))
