"""
HALO ML — Phase 1 (feature extraction).

Turns one raw email record (a dict or a pandas row with the normalized
schema from load_data.py) into:
  - a dict of structured/tabular features (numeric + boolean)
  - the raw text used for the TF-IDF branch (subject + body)

This module is imported by BOTH train.py and the inference service, so
training and serving can never drift apart.
"""
import re
import math

URL_RE = re.compile(r"https?://[^\s\"'<>\)\]]+", re.IGNORECASE)
IP_HOST_RE = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")
PUNYCODE_RE = re.compile(r"xn--", re.IGNORECASE)

URGENCY_WORDS = ["urgent", "immediately", "action required", "within 24 hours",
                  "verify now", "suspended", "act now", "asap", "expire", "final notice"]
CREDENTIAL_WORDS = ["password", "otp", "verification code", "login", "credentials",
                     "confirm your account", "security code", "click here to verify"]
PAYMENT_WORDS = ["wire transfer", "bank details", "processing fee", "gift card",
                  "account number", "payment", "routing number", "refundable fee"]


def _domain_of(addr):
    if not addr:
        return None
    m = re.search(r"@([a-zA-Z0-9.-]+)", addr)
    return m.group(1).lower() if m else None


def _extract_urls(text):
    return URL_RE.findall(text or "")


def _url_domain(url):
    m = re.match(r"https?://([^/]+)", url)
    if not m:
        return None
    return m.group(1).split(":")[0].split("@")[-1].lower()


def _count_hits(text, words):
    lower = (text or "").lower()
    return sum(1 for w in words if w in lower)


def extract_structured_features(row):
    """row: dict-like with keys from the normalized schema (see load_data.py)."""
    subject = row.get("subject") or ""
    body = row.get("body_text") or ""
    from_addr = row.get("from_address") or ""
    reply_to = row.get("reply_to") or ""

    from_domain = _domain_of(from_addr)
    reply_domain = _domain_of(reply_to)

    urls = _extract_urls(body)
    url_domains = [_url_domain(u) for u in urls if _url_domain(u)]
    ip_based = sum(1 for d in url_domains if IP_HOST_RE.match(d))
    punycode = sum(1 for d in url_domains if PUNYCODE_RE.search(d))
    mismatched = sum(1 for d in url_domains if from_domain and d != from_domain)
    subdomain_heavy = sum(1 for d in url_domains if d.count(".") >= 3)

    feats = {
        # already provided by the source datasets — real signals, not derived
        "url_count": int(row.get("url_count") or 0),
        "attachment_count": int(row.get("attachment_count") or 0),
        "has_credential_request": int(bool(row.get("has_credential_request"))),
        "has_otp_request": int(bool(row.get("has_otp_request"))),
        "has_payment_request": int(bool(row.get("has_payment_request"))),
        "has_urgency": int(bool(row.get("has_urgency"))),
        "html_present": int(bool(row.get("html_present"))),
        "script_present": int(bool(row.get("script_present"))),
        "auth_headers_present": int(bool(row.get("authentication_headers_present"))),

        # derived sender features
        "from_missing": int(from_domain is None),
        "reply_to_present": int(bool(reply_to)),
        "reply_to_mismatch": int(bool(reply_domain and from_domain and reply_domain != from_domain)),
        "from_local_has_digits": int(bool(re.search(r"\d", from_addr.split("@")[0] if "@" in from_addr else ""))),

        # derived content features (independent re-detection, not just a
        # copy of the provided has_* flags, so the model has signal even
        # if those flags are noisy or absent for new campus mail)
        "urgency_word_hits": _count_hits(subject + " " + body, URGENCY_WORDS),
        "credential_word_hits": _count_hits(subject + " " + body, CREDENTIAL_WORDS),
        "payment_word_hits": _count_hits(subject + " " + body, PAYMENT_WORDS),
        "subject_len": len(subject),
        "body_len": len(body),
        "body_len_log": math.log1p(len(body)),
        "subject_has_exclaim": int("!" in subject),
        "subject_is_reply_or_fwd": int(bool(re.match(r"^(re|fwd)\s*:", subject.strip().lower()))),

        # derived URL/domain features
        "url_extracted_count": len(urls),
        "url_ip_based_count": ip_based,
        "url_punycode_count": punycode,
        "url_domain_mismatch_count": mismatched,
        "url_subdomain_heavy_count": subdomain_heavy,
    }
    return feats


STRUCTURED_FEATURE_NAMES = list(extract_structured_features({
    "subject": "", "body_text": "", "from_address": "", "reply_to": "",
    "url_count": 0, "attachment_count": 0,
}).keys())


def extract_text(row):
    return (row.get("subject") or "") + " \n " + (row.get("body_text") or "")
