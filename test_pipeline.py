"""
HALO ML — automated tests (stdlib unittest, no extra dependencies).

Run with:  cd halo_ml && python3 -m unittest tests.test_pipeline -v
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np

from features import extract_structured_features, extract_text, STRUCTURED_FEATURE_NAMES
from text_clean import clean_for_text_model

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "halo_trust_model_v3.joblib")


class FeatureExtractionTests(unittest.TestCase):

    def test_structured_features_on_normal_email(self):
        row = {"subject": "Meeting tomorrow", "body_text": "Let's meet at 10am.",
               "from_address": "prof@college.edu", "reply_to": "", "url_count": 0, "attachment_count": 0}
        feats = extract_structured_features(row)
        self.assertEqual(feats["has_credential_request"], 0)
        self.assertEqual(feats["from_missing"], 0)
        self.assertEqual(set(feats.keys()), set(STRUCTURED_FEATURE_NAMES))

    def test_structured_features_on_empty_email(self):
        row = {"subject": "", "body_text": "", "from_address": "", "reply_to": "", "url_count": 0, "attachment_count": 0}
        feats = extract_structured_features(row)
        self.assertEqual(feats["from_missing"], 1)
        self.assertEqual(feats["subject_len"], 0)
        self.assertEqual(feats["body_len"], 0)

    def test_reply_to_mismatch_detected(self):
        row = {"subject": "x", "body_text": "y", "from_address": "a@bank.com",
               "reply_to": "b@totally-different.com", "url_count": 0, "attachment_count": 0}
        self.assertEqual(extract_structured_features(row)["reply_to_mismatch"], 1)

    def test_reply_to_match_not_flagged(self):
        row = {"subject": "x", "body_text": "y", "from_address": "a@bank.com",
               "reply_to": "b@bank.com", "url_count": 0, "attachment_count": 0}
        self.assertEqual(extract_structured_features(row)["reply_to_mismatch"], 0)

    def test_url_ip_based_detected(self):
        row = {"subject": "x", "body_text": "Click http://192.168.1.5/verify now",
               "from_address": "a@x.com", "reply_to": "", "url_count": 1, "attachment_count": 0}
        self.assertGreaterEqual(extract_structured_features(row)["url_ip_based_count"], 1)

    def test_url_domain_mismatch_detected(self):
        row = {"subject": "x", "body_text": "See http://totally-unrelated-domain.example/verify",
               "from_address": "alerts@paypal.com", "reply_to": "", "url_count": 1, "attachment_count": 0}
        self.assertGreaterEqual(extract_structured_features(row)["url_domain_mismatch_count"], 1)

    def test_malformed_email_no_at_sign_does_not_crash(self):
        row = {"subject": "weird", "body_text": "body", "from_address": "not-an-email",
               "reply_to": "also not an email", "url_count": 0, "attachment_count": 0}
        feats = extract_structured_features(row)  # should not raise
        self.assertEqual(feats["from_missing"], 1)

    def test_extract_text_handles_none_fields(self):
        self.assertEqual(extract_text({"subject": None, "body_text": None}), " \n ")


class TextSanitizationTests(unittest.TestCase):

    def test_html_tags_stripped(self):
        cleaned = clean_for_text_model("<div><b>hello</b></div>")
        self.assertNotIn("<", cleaned)
        self.assertNotIn("div", cleaned)

    def test_years_stripped(self):
        cleaned = clean_for_text_model("This was sent in 2002 and again in 2024")
        self.assertNotIn("2002", cleaned)
        self.assertNotIn("2024", cleaned)

    def test_corpus_jargon_stripped(self):
        cleaned = clean_for_text_model("posted to spamassassin and spambayes lists")
        self.assertNotIn("spamassassin", cleaned)
        self.assertNotIn("spambayes", cleaned)


@unittest.skipUnless(os.path.exists(MODEL_PATH), "model artifact not built yet")
class ModelInferenceTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.artifact = joblib.load(MODEL_PATH)

    def _score(self, row):
        feats = extract_structured_features(row)
        x = np.array([feats[f] for f in self.artifact["features"]], dtype=float)
        return self.artifact["model"].predict_proba(x.reshape(1, -1))[0, 1]

    def test_obvious_phishing_scores_high_risk(self):
        row = {
            "subject": "Urgent - verify your password now",
            "body_text": "Your account will be suspended. Send your password and OTP immediately, act now.",
            "from_address": "security@paypal-secure-verify.com", "reply_to": "recovery@totally-different.ru",
            "url_count": 3, "attachment_count": 0,
            "has_credential_request": True, "has_otp_request": True,
            "has_payment_request": False, "has_urgency": True,
        }
        p = self._score(row)
        self.assertGreaterEqual(p, self.artifact["threshold"], f"expected malicious, got p={p}")

    def test_routine_email_does_not_score_extreme(self):
        row = {
            "subject": "Class rescheduled to Friday",
            "body_text": "Hi all, our Monday class moves to Friday same time.",
            "from_address": "prof.sharma@college.edu", "reply_to": "prof.sharma@college.edu",
            "url_count": 0, "attachment_count": 0,
        }
        p = self._score(row)
        self.assertLess(p, 0.9, f"expected a routine email to not score near-certain malicious, got p={p}")

    def test_missing_optional_fields_does_not_crash(self):
        p = self._score({})
        self.assertTrue(0.0 <= p <= 1.0)

    def test_short_benign_email_not_flagged_high_risk(self):
        """Regression test: v3 (pre-fix) scored this at P(malicious)=0.91
        purely because it was short, driven by subject_len/body_len --
        a corpus-genre artifact (EPVME's malicious set skews short, "
        SpamAssassin ham skews long/quoted), not a phishing signal."""
        row = {"subject": "Class rescheduled to Friday",
               "body_text": "Hi all, our Monday class moves to Friday same time.",
               "from_address": "prof.sharma@college.edu", "reply_to": "prof.sharma@college.edu",
               "url_count": 0, "attachment_count": 0}
        p = self._score(row)
        self.assertLess(p, 0.5, f"short benign email incorrectly flagged, got p={p}")

    def test_feature_order_matches_training(self):
        self.assertEqual(self.artifact["model"].n_features_in_, len(self.artifact["features"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
