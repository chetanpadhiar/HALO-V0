"""
HALO ML — artifact-aware text cleaning.

Diagnostic on the naive TF-IDF model showed it was substantially keying
on WHICH SOURCE CORPUS a message came from rather than phishing intent:
  - "2002" / "sep" / date tokens  -> legitimate  (SpamAssassin ham is
    literally all from 2002; this is a corpus era leak, not signal)
  - "spamassassin" / "spambayes" / "fork" / "linux" -> legitimate
    (SpamAssassin ham is a tech mailing-list archive; these are that
    list's own jargon, not general markers of a safe email)
  - raw HTML tag tokens ("br", "font", "div", "img", "src", "href")
    -> malicious (EPVME's malicious set is HTML/marketing-heavy; but
    "this email contains HTML" is already captured by the separate
    html_present structured feature, so letting markup tokens into the
    vocabulary just lets the text model re-detect that instead of
    reading content)

This module removes exactly those categories of token before
vectorizing, so the remaining vocabulary reflects message content.
"""
import re

TAG_RE = re.compile(r"<[^>]+>")
ENTITY_RE = re.compile(r"&[a-zA-Z]+;|&#\d+;")
YEAR_OR_DATE_RE = re.compile(r"\b(19|20)\d{2}\b|\b\d{1,2}[:/-]\d{1,2}([:/-]\d{2,4})?\b")
PURE_NUMERIC_RE = re.compile(r"\b\d+\b")

# proper nouns specific to the SpamAssassin/tech-mailing-list corpus that
# leak the source of the *ham* half of the data, found via coefficient
# inspection during development (see README "Known limitations").
CORPUS_JARGON = [
    "spamassassin", "spambayes", "exmh", "razor", "netnews",
    "sourceforge", "listinfo", "mailman", "rpm",
]

# EPVME-specific test-harness boilerplate (automated XSS/injection probes
# submitted as "malicious" samples) -- these phrases identify the
# *collection method*, not phishing content, and would otherwise let a
# text model "detect phishing" by literally detecting the word "test".
TEST_HARNESS_PHRASES = [
    "xss test", "test xss", "this is a test mail", "test mail for xss",
]


def clean_for_text_model(raw):
    text = raw or ""
    text = TAG_RE.sub(" ", text)
    text = ENTITY_RE.sub(" ", text)
    text = YEAR_OR_DATE_RE.sub(" ", text)
    text = PURE_NUMERIC_RE.sub(" ", text)
    lower = text.lower()
    for phrase in TEST_HARNESS_PHRASES:
        lower = lower.replace(phrase, " ")
    for term in CORPUS_JARGON:
        lower = lower.replace(term, " ")
    lower = re.sub(r"\s+", " ", lower).strip()
    return lower
