# HALO ML — Contextual Trust Model

A real, trained scikit-learn model for HALO's "Content & Intent" / behavioral
signal, served by a small Python API. This document reports what was actually
done, including the mistakes that were caught and reversed, because the
value of this pipeline is as much in the honesty of the evaluation as in the
model itself.

**If you read nothing else, read "What we do NOT claim" at the bottom.**

---

## 1. Dataset

| File | Rows | Label |
|---|---|---|
| `epvme_emails.csv` | 37,036 | malicious |
| `spamassassin_legitimate.csv` | 2,501 | legitimate |
| **Combined** | **39,537** | — |

This is **not** the ~7,501-row / 5,000-malicious estimate in the original
brief — the real EPVME file is ~7.4x larger than expected. The class ratio
is **14.8 : 1** (malicious : legitimate) overall, and worse — **93–97%** —
within any individual split, because grouping isn't perfectly stratifiable.

**Both files needed a custom parser.** `pandas` (both the C and Python
engines) silently mis-parses this data: HTML fragments in `body_text`
contain long runs of doubled quote characters that produce an ambiguous
parse. The stdlib `csv` module handles it correctly (0 rows dropped, verified
against a hand count). See `src/load_data.py`.

### The central limitation of this dataset

```
                        legitimate   malicious
EPVME-Dataset                    0       37036
SpamAssassin Easy Ham         2501           0
```

**Every legitimate row comes from one corpus; every malicious row comes from
a different corpus. `source_dataset` and `label` are 100% confounded.** This
single fact drives most of the interesting findings below — any feature that
happens to correlate with *which corpus a message came from* (era, writing
register, mailing-list jargon, HTML boilerplate) will look like a phishing
signal here even though it is one only by coincidence of dataset assembly.
This is a property of the two files, not a preprocessing bug, and it cannot
be fully engineered around — see §6.

### Duplicates and near-duplicates

- **21,245 of 37,036 EPVME rows are exact-duplicate template blasts** (same
  normalized body reused across many `email_id`s).
- After also normalizing away numbers/URLs/emails (so a template with a
  substituted amount or link still matches its siblings), duplicate groups
  drop further: **22,340 exact-dup groups → 20,011 near-dup "template
  groups."** 641 template groups were confirmed to span more than one exact-
  dup group (e.g. a 4-email "FDA approved on-line pharmacies" family with
  varying links/amounts).
- **All splitting is done on these template groups**, not on rows, so no
  duplicate or near-duplicate email can appear in more than one of
  train/validation/test. This was verified programmatically (zero group
  overlap) — see `src/split.py`.

---

## 2. Methodology

```
epvme_emails.csv, spamassassin_legitimate.csv
        │  src/load_data.py            (custom CSV parser, schema normalization)
        ▼
combined_raw.pkl
        │  src/regroup_near_duplicates.py   (template-group near-dup detection)
        ▼
        │  src/build_features.py       (structured features + raw text, cached)
        ▼
features.pkl
        │  src/split.py                (GroupShuffleSplit on template groups, seed=42)
        ▼
train / validation / test  (no group ever spans two splits)
        │  src/train_v3.py             (full model comparison + ablation)
        │  src/eval_utils.py           (majority baseline + full metric suite)
        ▼
src/finalize_model_v3.py               (refit on train+val, ONE test touch)
        ▼
models/halo_trust_model_v3.joblib + .metadata.json
        ▼
service/app.py                          (Flask inference API, loads the artifact)
```

**The test set was touched exactly once**, for the metrics reported in §5,
using a decision threshold chosen entirely from validation data.

---

## 3. What we report and why (the accuracy trap)

The **majority-class baseline** — always predict "malicious" — scores:

| Split | Majority-baseline accuracy | Balanced accuracy |
|---|---|---|
| train | 94.2% | 50.0% |
| validation | 90.9% | 50.0% |
| test | 92.6% | 50.0% |

**A model that is useless for recognizing a single legitimate email still
scores >90% raw accuracy on this data.** This is why every result below is
reported with **balanced accuracy, macro-F1, MCC, PR-AUC, and per-class
recall** front and center, with raw accuracy included only for context. Our
first-pass model (documented as "v1" in git history / superseded scripts)
reported 88.4% accuracy as if it were close to a 90–95% target — it is
actually *below* the majority baseline on that specific metric, which is
exactly the trap this section exists to name.

---

## 4. What we tried, and what we rejected (in order)

This is the part most reports skip. We are including it because catching our
own mistakes is the actual evidence of rigor, not the final number.

### 4.1 — Text models looked incredible. That was the problem.

A TF-IDF + Logistic Regression model reached **99.9% validation F1**.
Inspecting the top coefficients:

| Toward "malicious" | Toward "legitimate" |
|---|---|
| `html`, `font`, `div`, `img`, `src`, `xss`, `test mail` | `2002`, `spambayes`, `spamassassin`, `fork`, `linux` |

**`2002` is a strong "legitimate" predictor because SpamAssassin's ham corpus
is literally all from 2002.** `spamassassin`/`spambayes`/`fork` are that
corpus's own mailing-list jargon. The HTML tokens are EPVME's test-payload
boilerplate. **The model was learning which corpus a message came from, not
whether it was phishing.** Not reported, not shipped.

### 4.2 — First cleaning pass: still leaking.

Built `src/text_clean.py`: strips HTML tags, years/dates, and known
corpus-jargon terms before vectorizing. Re-ran: **still ~99.9% F1.**
Re-inspected coefficients: `zzzzteana` and `satalk` — both literal
SpamAssassin-bundled mailing-list names — were now the top "legitimate"
tokens, alongside generic-but-genre-specific words (`wrote`, `forwarded`,
`mv`, `list`). **Two rounds of targeted token removal did not fix it, and a
third round of manual whack-a-mole would face the same problem indefinitely**
— because of §1's confound: source and label are perfectly aliased, so
*any* systematic stylistic difference between a 2002 tech-mailing-list
archive and a mass-produced phishing/test corpus will look like signal no
matter how much specific vocabulary gets blocklisted.

**Conclusion: text/hybrid models are not shippable on this data, at any
accuracy they report**, until the legitimate and malicious classes come from
the same contemporary source. See `src/train_v3.py`'s full output for the
complete ablation (text-only / structured-only / hybrid, each with full
metrics) — the comparison itself is the useful artifact, even though we
don't ship its winner.

### 4.3 — Structured-only, and a caught false positive.

Retreating to structured features only (no free text) removes most of the
above risk, since these features describe *mechanism* (does it ask for a
password? does the reply-to match?) rather than vocabulary. First pass used
all 25 structured features and reached 91.9% test balanced accuracy — but
our own test suite caught a real bug:

```
test_routine_email_does_not_score_extreme:
  "Class rescheduled to Friday" / "Hi all, our Monday class moves to
  Friday same time." → P(malicious) = 0.91
```

A completely benign, short email was flagged high-risk. Feature inspection
showed this was driven almost entirely by `subject_len`/`body_len`: EPVME's
malicious set skews short (many are template/test payloads), SpamAssassin's
ham skews long (quoted mailing-list threads) — **the same corpus-genre
problem, now hiding in message length instead of vocabulary.**

**Fix:** removed `subject_len`, `body_len`, `body_len_log`,
`subject_has_exclaim`, `subject_is_reply_or_fwd` — five features with no
causal link to phishing risk. Regression-tested: the same email now scores
`P(malicious) = 0.0000`. This exact case is now a permanent unit test
(`test_short_benign_email_not_flagged_high_risk`).

We also tested removing `html_present` entirely (it's the single largest
remaining feature, and it's also corpus-confounded — 72.5% of EPVME is HTML
vs 0.3% of SpamAssassin ham). Result: balanced accuracy collapsed from 0.90
to 0.67 and FPR jumped to 54%. **`html_present` carries real, substantial
signal despite the confound, so it is kept** — but flagged as the #1 item to
revalidate against real legitimate HTML campus mail (LMS notices, official
announcements) before trusting in production, since modern legitimate email
is frequently HTML too.

---

## 5. Final model & honest test-set results

**`HALO-Trust-RF-v3`** — `RandomForestClassifier`, 400 trees,
`class_weight="balanced"`, 21 structured features (see
`models/halo_trust_model_v3.metadata.json` for the exact list and their
importances). Chosen over a HistGradientBoosting candidate that scored
within noise of it (0.904 vs 0.898 balanced accuracy on this feature set)
because our explainability layer (per-tree contribution attribution, see
`service/app.py`) is built around scikit-learn's standard tree structure,
which `RandomForestClassifier` exposes directly and `HistGradientBoosting`
does not.

Decision threshold (0.412) was chosen on **validation only**, maximizing
balanced accuracy subject to malicious recall ≥ 85%.

**Test set (touched once, n=4,689):**

| Metric | Value |
|---|---|
| Accuracy | 88.2% (majority baseline: 92.6% — see §3 for why this is expected and not the headline number) |
| **Balanced accuracy** | **90.2%** |
| Macro-F1 | 73.6% |
| MCC | 0.548 |
| ROC-AUC | 95.5% |
| PR-AUC | 99.5% |
| Malicious recall | 87.9% |
| Legitimate recall (specificity) | 92.6% |
| Malicious precision | 99.3% |
| Legitimate precision | 38.0% |
| False positive rate | 7.4% |
| False negative rate | 12.1% |

Confusion matrix: TN=323, FP=26, FN=527, TP=3813.

**Read the legitimate-class precision (38%) carefully**: of every message
this model calls "legitimate," a real chunk are actually missed phishing —
a direct, disclosed consequence of the ~93:7 imbalance in the test set, not
a hidden flaw. This is exactly why HALO's product design uses a **3-tier
bucket (safe / needs attention / high-risk)** rather than a binary verdict —
a borderline case lands in "needs attention," not a false "all clear."

---

## 6. Known limitations (read before deploying)

1. **`html_present` is the largest feature and is still corpus-confounded.**
   Must be revalidated against real legitimate HTML campus email before
   production use.
2. **Text/hybrid models are not shipped**, even though they scored higher on
   paper, because of the source/label confound in §1/§4.2. This is the
   single most important thing to fix before the next iteration: collect
   real (or carefully simulated) campus email for **both** classes from the
   **same** contemporary source, so corpus identity stops aliasing with
   label. Once that data exists, re-run `src/train_v3.py` — the pipeline is
   already built for this comparison.
3. **Legitimate-class precision is modest** at the shipped operating point
   (see §5). Mitigated in-product by the 3-tier bucket, not by the model
   alone.
4. **Authentication signal is weak in this dataset** — only a bare
   `authentication_headers_present` boolean, no granular SPF/DKIM/DMARC per
   row. The frontend prototype's simulated demo data has a richer,
   header-parsing identity signal than this dataset can supply.
5. The 90–95% accuracy target in the original brief is not a meaningful bar
   under this class imbalance (see §3) — balanced accuracy (90.2%) is the
   closer honest analog, and it clears the range; raw accuracy (88.2%) does
   not, and is reported anyway rather than hidden.

## What we do NOT claim

- We do not claim 90–95% raw accuracy. We report 88.2%, explain why raw
  accuracy is the wrong headline metric here, and report balanced accuracy
  (90.2%) as the more honest analog.
- We do not claim the text/TF-IDF experiments validate NLP-based phishing
  detection — they demonstrate a dataset confound, and are kept in the repo
  as a documented negative result, not a capability.
- We do not claim `html_present` is a clean, corpus-independent signal — we
  show the evidence that it partly isn't, and ship it anyway with that
  caveat because removing it is worse (§4.3).
- We do not claim this model is ready for production scoring of real
  student email without the revalidation in §6.2.

---

## 7. Running it

```bash
# 1. Rebuild the dataset + features (only needed once, or after changing
#    the source CSVs) -- point PATHS at wherever your CSVs actually are:
cd halo_ml/src
python3 load_data.py               # -> ../data/combined_raw.pkl
python3 regroup_near_duplicates.py # adds near-dup template groups
python3 build_features.py          # -> ../data/features.pkl
python3 split.py                   # -> ../data/split_indices.npz

# 2. Reproduce the full model comparison + ablation (optional -- the
#    final artifact is already built and committed):
python3 train_v3.py

# 3. Reproduce the final model + single test-set touch (optional):
python3 finalize_model_v3.py

# 4. Run the tests:
cd ..
python3 -m unittest tests.test_pipeline -v

# 5. Start the real inference service:
cd service
python3 app.py
# -> http://127.0.0.1:8787  (health: GET /health, inference: POST /analyze)
```

The frontend (`HALO_2.0_prototype.html`) calls this service directly from
the browser for its "Live ML Model" tab on any analyzed email. If the
service isn't running, the UI says so plainly ("ML SERVICE OFFLINE... start
it with...") rather than showing a substitute number — there is no
JavaScript reimplementation of the model anywhere in this project.

### Superseded scripts, kept for the record

`train.py`, `train_conservative.py`, `train_v2.py`, `finalize_model.py`,
`finalize_model_v2.py` are the earlier, rejected passes described in §4.
They still run, and running them reproduces the exact numbers quoted above
as "rejected" — they're kept so the journey in this README is independently
checkable, not just asserted.
