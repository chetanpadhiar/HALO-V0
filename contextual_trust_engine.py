"""
HALO contextual layer — contextual_trust_engine.py

This is the layer the product brief calls for: the ML model's
prediction is ONE signal among several, not the final verdict. This
module fuses:

  - identity / authentication  (real SPF/DKIM/DMARC + homograph + Reply-To
                                  mismatch, parsed from raw headers)
  - content / intent            (the trained ML model's probability --
                                  this is where the real model plugs in)
  - URL / domain                (IP-based links, punycode, domain mismatch)
  - behavioural                 (sender history baseline, explicit
                                  "insufficient_history" when there is none)
  - infrastructure               (observed IP/ASN reference lookup,
                                  explicitly not attacker location)

into: HALO Trust Score (0-100), bucket (safe/attention/high-risk),
confidence, plain-language reasons, and full technical evidence per
signal family.

Pipeline stage: email -> email_parser -> features.py -> ML model ->
                THIS MODULE -> verdict
"""
import os
import joblib

from email_parser import parse_email_source
from features import extract_structured_features
import infrastructure_intel
import behavioral_baseline

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "halo_trust_model_v3.joblib")
_artifact = joblib.load(MODEL_PATH)
ML_MODEL = _artifact["model"]
ML_FEATURES = _artifact["features"]
ML_THRESHOLD = _artifact["threshold"]


def _identity_signal(parsed):
    auth = parsed["auth"]
    score = 100
    evidence = []

    for name, val, weight in [("SPF", auth["spf"], 35), ("DKIM", auth["dkim"], 30), ("DMARC", auth["dmarc"], 25)]:
        if val == "PASS":
            evidence.append(f"{name} check: PASS")
        elif val in ("FAIL", "REJECT"):
            score -= weight
            evidence.append(f"{name} check: {val}")
        else:
            evidence.append(f"{name} check: {val}")

    if parsed["homograph_of"]:
        score -= 25
        evidence.append(f"Sending domain \"{parsed['from_domain']}\" resembles protected brand "
                          f"\"{parsed['homograph_of']}\" (possible homograph/lookalike)")

    reply_to_mismatch = bool(parsed["reply_to_domain"] and parsed["from_domain"]
                               and parsed["reply_to_domain"] != parsed["from_domain"])
    if reply_to_mismatch:
        score -= 10
        evidence.append(f"Reply-To domain ({parsed['reply_to_domain']}) does not match "
                          f"From domain ({parsed['from_domain']})")

    score = max(0, min(100, score))
    flagged = score < 60 or bool(parsed["homograph_of"])  # a homograph hit is disqualifying on its own
    return {"score": score, "evidence": evidence, "flagged": flagged,
            "reply_to_mismatch": reply_to_mismatch, "homograph_of": parsed["homograph_of"]}


def _url_domain_signal(struct_feats):
    evidence = []
    score = 0
    if struct_feats["url_ip_based_count"] > 0:
        score += 35
        evidence.append("One or more links point directly to a numeric IP address")
    if struct_feats["url_punycode_count"] > 0:
        score += 30
        evidence.append("One or more links use lookalike (punycode) domain characters")
    if struct_feats["url_domain_mismatch_count"] > 0:
        score += 30
        evidence.append("One or more links point to a different domain than the sender")
    if struct_feats["url_subdomain_heavy_count"] > 0:
        score += 15
        evidence.append("One or more links use an unusually long chain of subdomains")
    if not evidence:
        evidence.append("No suspicious link structure detected" if struct_feats["url_extracted_count"] > 0
                         else "No links found in the message")
    score = min(100, score)
    return {"score": score, "evidence": evidence, "flagged": score >= 40}


def _ml_content_signal(struct_feats):
    x = [[struct_feats[f] for f in ML_FEATURES]]
    proba_malicious = float(ML_MODEL.predict_proba(x)[0][1])
    flagged = proba_malicious >= ML_THRESHOLD
    evidence = []
    if struct_feats.get("has_credential_request"):
        evidence.append("The message asks for a password, login, or verification code")
    if struct_feats.get("has_payment_request"):
        evidence.append("The message asks for a payment or financial details")
    if struct_feats.get("has_urgency"):
        evidence.append("The message creates urgency or a tight deadline")
    if struct_feats.get("credential_word_hits"):
        evidence.append("Wording matches common credential-request phrases")
    if struct_feats.get("payment_word_hits"):
        evidence.append("Wording matches common payment-request phrases")
    if not evidence:
        if flagged:
            # The model can flag a message on structural features (URL
            # shape, sender/reply-to mismatch, etc. -- all separately
            # inputs to this same model) even when no urgency/credential/
            # payment keyword fired. Say so plainly instead of showing
            # the reassuring "nothing found" line next to a red flag.
            evidence.append("The trained model's overall assessment flags this message, "
                              "though no single keyword stands out on its own")
        else:
            evidence.append("No urgency, credential, or payment-request language detected")
    return {
        "score": round((1 - proba_malicious) * 100),
        "probability_malicious": round(proba_malicious, 4),
        "model_threshold": ML_THRESHOLD,
        "model_name": _artifact.get("model_name", "HALO-Trust-RF-v3"),
        "evidence": evidence,
        "flagged": flagged,
        "has_credential_request": bool(struct_feats.get("has_credential_request")),
        "has_payment_request": bool(struct_feats.get("has_payment_request")),
        "has_urgency": bool(struct_feats.get("has_urgency")),
    }


def _confidence(behavior, infra, ml):
    conf = 100
    notes = []
    if behavior["status"] == "insufficient_history":
        conf -= 15
        notes.append("no behavioural history for this sender")
    if infra.get("confidence") in ("none", "low"):
        conf -= 10
        notes.append("infrastructure signal has low confidence")
    if abs(ml["probability_malicious"] - ml["model_threshold"]) < 0.1:
        conf -= 15
        notes.append("ML signal is close to its decision boundary")
    conf = max(40, conf)
    label = "high" if conf >= 85 else "medium" if conf >= 65 else "low"
    return {"score": conf, "label": label, "notes": notes}


def analyze(raw_source=None, from_address=None, reply_to=None, subject=None, body_text=None,
            record_behavior=True):
    """Run the full contextual trust engine on one email. Accepts either
    a raw email source string OR individual fields (the demo dataset
    ships as individual fields; live mail would supply raw_source)."""
    if raw_source:
        parsed = parse_email_source(raw_source)
    else:
        from email_parser import is_homograph_of
        fdom = _domain(from_address)
        parsed = {
            "raw": None, "header_lines": [], "body": body_text or "",
            "from_field": from_address or "", "reply_to_field": reply_to or "",
            "subject": subject or "", "date": None,
            "sender_addr": from_address, "from_domain": fdom,
            "reply_to_domain": _domain(reply_to),
            "auth": {"spf": "NOT PRESENT", "dkim": "NOT PRESENT", "dmarc": "NOT PRESENT", "present": False},
            "origin_ip": None,
            "homograph_of": is_homograph_of(fdom),
            "received_count": 0,
        }

    row = {
        "from_address": parsed["from_field"] or parsed.get("sender_addr") or "",
        "reply_to": parsed["reply_to_field"],
        "subject": parsed["subject"],
        "body_text": parsed["body"],
        "url_count": len(__import__("re").findall(r"https?://", parsed["body"] or "")),
        "attachment_count": 0,
        "html_present": _looks_like_html(parsed["body"]), "script_present": _has_script_tag(parsed["body"]),
        # NOTE: authentication_headers_present is intentionally always False
        # here, not derived from whether this email has an Authentication-
        # Results header. In training data this feature is ~99.9% False in
        # BOTH classes (near-constant / uninformative) -- but the rare
        # True rows that exist are overwhelmingly EPVME (malicious), so the
        # model learned "True" as a malicious-only artifact from a handful
        # of noisy examples. A real, well-configured legitimate mail server
        # commonly DOES have this header -- feeding that through as True
        # would trigger the artifact and misclassify good mail. The real
        # SPF/DKIM/DMARC richness is already captured independently by the
        # identity signal family below, so no real signal is lost by
        # holding this one ML input at its training-distribution baseline.
        "authentication_headers_present": False,
    }
    # Live/demo email has no pre-existing human annotation for has_*
    # (unlike the training CSVs) -- derive it from the same keyword
    # detection used for *_word_hits, rather than silently leaving it
    # False (which would understate risk on real mail).
    text_lower = f"{row['subject']} {row['body_text']}".lower()
    row["has_credential_request"] = any(w in text_lower for w in
        ["password", "otp", "verification code", "login", "credentials", "security code"])
    row["has_otp_request"] = "otp" in text_lower or "one-time" in text_lower
    row["has_payment_request"] = any(w in text_lower for w in
        ["wire transfer", "bank details", "processing fee", "gift card", "account number", "payment", "refundable fee"])
    row["has_urgency"] = any(w in text_lower for w in
        ["urgent", "immediately", "act now", "within 24 hours", "asap", "final notice"])
    struct_feats = extract_structured_features(row)

    identity = _identity_signal(parsed)
    ml_content = _ml_content_signal(struct_feats)
    url_domain = _url_domain_signal(struct_feats)
    behavior = behavioral_baseline.STORE.lookup(parsed.get("sender_addr"), parsed["subject"], parsed["body"])
    infra = infrastructure_intel.lookup(parsed.get("origin_ip"))

    if record_behavior and parsed.get("sender_addr"):
        behavioral_baseline.STORE.record(
            parsed["sender_addr"], parsed["subject"],
            bool(struct_feats.get("has_credential_request")), bool(struct_feats.get("has_payment_request")),
        )

    flags = {
        "identity": identity["flagged"],
        "ml_content": ml_content["flagged"],
        "url_domain": url_domain["flagged"],
        "behavior": behavior["flagged"],
        "infra": infra["flagged"],
    }
    flagged_count = sum(flags.values())
    high_risk = flagged_count >= 2

    technical_legitimacy_score = identity["score"]
    ml_risk = ml_content["probability_malicious"] * 100
    contextual_risk = (0.40 * ml_risk + 0.25 * behavior.get("score", 0)
                        + 0.20 * url_domain["score"] + 0.15 * infra["score"])
    contextual_legitimacy_score = round(max(0, min(100, 100 - contextual_risk)))

    if not identity["flagged"] and high_risk:
        verdict = "Possible account takeover or social engineering"
        bucket = "high-risk"
    elif high_risk:
        verdict = "High risk -- likely malicious"
        bucket = "high-risk"
    elif flagged_count == 1:
        verdict = "Needs attention -- one signal flagged"
        bucket = "attention"
    else:
        verdict = "Looks safe"
        bucket = "safe"

    def _label(score):
        return "HIGH" if score >= 70 else "MEDIUM" if score >= 40 else "LOW"

    technical_legitimacy_label = _label(technical_legitimacy_score)
    contextual_legitimacy_label = _label(contextual_legitimacy_score)

    trust_score = contextual_legitimacy_score
    confidence = _confidence(behavior, infra, ml_content)

    reasons = []
    for fam_name, fam in [("identity", identity), ("ml_content", ml_content),
                            ("url_domain", url_domain), ("behavior", behavior), ("infra", infra)]:
        if fam.get("flagged") and fam.get("evidence"):
            reasons.append(fam["evidence"][0])

    return {
        "verdict": verdict,
        "bucket": bucket,
        "trust_score": trust_score,
        "technical_legitimacy": {"score": technical_legitimacy_score, "label": technical_legitimacy_label},
        "contextual_legitimacy": {"score": contextual_legitimacy_score, "label": contextual_legitimacy_label},
        "confidence": confidence,
        "reasons": reasons[:4],
        "flagged_signal_count": flagged_count,
        "signals": {
            "identity": identity,
            "content_intent": ml_content,
            "url_domain": url_domain,
            "behavior": behavior,
            "infrastructure": infra,
        },
        "parsed": {
            "from": parsed["from_field"], "reply_to": parsed["reply_to_field"],
            "subject": parsed["subject"], "from_domain": parsed["from_domain"],
            "reply_to_domain": parsed.get("reply_to_domain"), "date": parsed.get("date"),
            "sender_addr": parsed.get("sender_addr"), "origin_ip": parsed.get("origin_ip"),
            "auth": parsed["auth"], "header_lines": parsed["header_lines"],
            "homograph_of": parsed.get("homograph_of"), "body": parsed.get("body", ""),
        },
    }


def _domain(addr):
    import re
    if not addr:
        return None
    m = re.search(r"@([a-zA-Z0-9.-]+)", addr)
    return m.group(1).lower() if m else None


def _looks_like_html(text):
    import re
    return bool(re.search(r"<\s*(html|div|table|span|img|a\s|body|br\s*/?>|font)", text or "", re.IGNORECASE))


def _has_script_tag(text):
    import re
    return bool(re.search(r"<\s*script", text or "", re.IGNORECASE))
