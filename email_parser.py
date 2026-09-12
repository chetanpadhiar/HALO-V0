"""
HALO contextual layer — email_parser.py

Parses a raw email source (headers + body) into structured fields:
From/Reply-To domains, SPF/DKIM/DMARC, origin IP, homograph check.

This is the "parser" stage of the pipeline described in the product
brief: email -> parser -> feature extraction -> ML model -> contextual
trust engine -> verdict. It is intentionally independent of the ML
training-data schema (features.py) because live/incoming email doesn't
arrive as a CSV row -- it arrives as raw source text, same as the
frontend's simulated demo emails.
"""
import re

PROTECTED_BRANDS = [
    "paypal.com", "google.com", "microsoft.com", "amazon.com",
    "apple.com", "accounts.google.com", "outlook.com",
]


def domain_of(addr):
    if not addr:
        return None
    m = re.search(r"@([a-zA-Z0-9.-]+)", addr)
    return m.group(1).lower() if m else None


def sender_address_of(from_field):
    m = re.search(r"<?([\w.+\-']+@[\w.-]+)>?", from_field or "")
    return m.group(1) if m else None


def _unfold_headers(header_block):
    lines = header_block.split("\n")
    out = []
    for line in lines:
        if line.startswith((" ", "\t")) and out:
            out[-1] += " " + line.strip()
        elif line.strip():
            out.append(line.rstrip("\r"))
    return out


def _get_header(lines, name):
    pattern = re.compile(rf"^{re.escape(name)}:\s*(.*)$", re.IGNORECASE)
    return [m.group(1) for l in lines if (m := pattern.match(l))]


def parse_auth_results(value):
    if not value:
        return {"spf": "NOT PRESENT", "dkim": "NOT PRESENT", "dmarc": "NOT PRESENT", "present": False}
    spf = re.search(r"spf=(\w+)", value, re.IGNORECASE)
    dkim = re.search(r"dkim=(\w+)", value, re.IGNORECASE)
    dmarc = re.search(r"dmarc=(\w+)", value, re.IGNORECASE)
    return {
        "spf": spf.group(1).upper() if spf else "NOT PRESENT",
        "dkim": dkim.group(1).upper() if dkim else "NOT PRESENT",
        "dmarc": dmarc.group(1).upper() if dmarc else "NOT PRESENT",
        "present": True,
    }


def extract_origin_ip(received_headers):
    if not received_headers:
        return None
    earliest = received_headers[-1]  # last Received = earliest hop
    m = re.search(r"\[?(\d{1,3}(?:\.\d{1,3}){3})\]?", earliest)
    return m.group(1) if m else None


def is_homograph_of(domain):
    if not domain:
        return None
    for brand in PROTECTED_BRANDS:
        name = brand.split(".")[0]
        if domain != brand and name in domain:
            return brand
    return None


def parse_email_source(raw):
    """raw: full email source (headers + blank line + body), OR just a
    headers block with no body. Returns a dict of parsed fields."""
    raw = raw or ""
    idx = re.search(r"\r?\n\r?\n", raw)
    header_block = raw[: idx.start()] if idx else raw
    body = raw[idx.end():].strip() if idx else ""

    lines = _unfold_headers(header_block)
    from_field = (_get_header(lines, "From") or [""])[0].strip()
    reply_to_field = (_get_header(lines, "Reply-To") or [""])[0].strip()
    subject = (_get_header(lines, "Subject") or [""])[0].strip()
    date = (_get_header(lines, "Date") or [""])[0].strip()
    auth_raw = (_get_header(lines, "Authentication-Results") or [""])[0]
    received = _get_header(lines, "Received")

    from_domain = domain_of(from_field)
    reply_to_domain = domain_of(reply_to_field)
    sender_addr = sender_address_of(from_field)

    return {
        "raw": raw,
        "header_lines": lines,
        "body": body,
        "from_field": from_field,
        "reply_to_field": reply_to_field,
        "subject": subject,
        "date": date,
        "sender_addr": sender_addr,
        "from_domain": from_domain,
        "reply_to_domain": reply_to_domain,
        "auth": parse_auth_results(auth_raw),
        "origin_ip": extract_origin_ip(received),
        "homograph_of": is_homograph_of(from_domain),
        "received_count": len(received),
    }
