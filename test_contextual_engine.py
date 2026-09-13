"""
HALO 2.0 — contextual layer tests.

Covers the 9 required scenarios end to end through the real pipeline
(parser -> features -> ML model -> contextual trust engine), plus the
campaign correlation layer and the Flask API surface.

Run: cd halo_ml && python3 -m unittest tests.test_contextual_engine -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import contextual_trust_engine as engine
import campaign_correlation
import behavioral_baseline


class NineScenarioTests(unittest.TestCase):
    """The 9 scenarios required by the product review."""

    def test_1_obvious_malicious_email(self):
        r = engine.analyze(raw_source=(
            'From: "PayPal Security" <alerts@paypal-secure-verify.com>\n'
            "Reply-To: recovery-support@paypal-secure-verify.com\n"
            "Subject: Action Required: unusual sign-in detected\n"
            "Received: from mx by y; Mon, 07 Sep 2026 11:42:09 +0000\n"
            "Received: from mail-relay.net (185.220.101.47) by mx; Mon, 07 Sep 2026 11:42:07 +0000\n"
            "Authentication-Results: mx; spf=fail; dkim=fail; dmarc=fail (p=reject)\n"
            "\n"
            "Dear valued customer, please verify now by confirming your account password within 24 hours."
        ))
        self.assertEqual(r["bucket"], "high-risk")
        self.assertTrue(r["signals"]["identity"]["flagged"])

    def test_2_legitimate_email(self):
        r = engine.analyze(raw_source=(
            'From: "Tech Insider Daily" <newsletter@tech-insider-daily.com>\n'
            "Subject: Your Monday briefing\n"
            "Received: from mx by y; Mon, 07 Sep 2026 07:00:05 +0000\n"
            "Received: from relay (142.250.80.46) by mx; Mon, 07 Sep 2026 07:00:03 +0000\n"
            "Authentication-Results: mx; spf=pass; dkim=pass; dmarc=pass\n"
            "\n"
            "Hi there, here are five stories in AI infrastructure this week."
        ))
        self.assertEqual(r["bucket"], "safe")
        self.assertGreaterEqual(r["trust_score"], 70)

    def test_3_credential_phishing(self):
        r = engine.analyze(
            from_address="security@accounts-google.co.in", reply_to="security@accounts-google.co.in",
            subject="Your account was accessed from a new device",
            body_text="Verify now by entering your login credentials and security code to confirm your account.",
        )
        self.assertIn(r["bucket"], ("attention", "high-risk"))
        self.assertTrue(r["signals"]["identity"]["flagged"], "homograph domain should flag identity")
        self.assertTrue(any("password" in x.lower() or "login" in x.lower() or "credential" in x.lower()
                              for x in r["signals"]["content_intent"]["evidence"]))

    def test_4_payment_phishing(self):
        r = engine.analyze(
            from_address="ceo.office@corp-mail-secure.net", reply_to="ceo.office@corp-mail-secure.net",
            subject="Confidential wire transfer needed today",
            body_text="Urgent wire transfer, send the bank details and account number right away.",
        )
        self.assertIn(r["bucket"], ("attention", "high-risk"))
        self.assertTrue(r["signals"]["content_intent"]["flagged"])

    def test_5_suspicious_url(self):
        r = engine.analyze(
            from_address="support@some-shop.com", reply_to="support@some-shop.com",
            subject="Confirm your order",
            body_text="Please confirm your order by visiting http://192.168.44.2/confirm now.",
        )
        self.assertTrue(r["signals"]["url_domain"]["flagged"])
        self.assertIn(r["bucket"], ("attention", "high-risk"))

    def test_6_reply_to_mismatch(self):
        r = engine.analyze(
            from_address="billing@legit-bank.com", reply_to="reply@totally-different-domain.ru",
            subject="Update your billing info",
            body_text="Please update your billing information at your convenience.",
        )
        self.assertTrue(r["signals"]["identity"]["reply_to_mismatch"])

    def test_7_behavioural_anomaly(self):
        behavioral_baseline.STORE.__init__()  # fresh seeded baseline
        r = engine.analyze(
            from_address="it-helpdesk@amityonline.edu", reply_to="it-helpdesk@amityonline.edu",
            subject="Immediate payment required to keep your account active",
            body_text="Please send payment immediately or your account access will be revoked today.",
        )
        beh = r["signals"]["behavior"]
        self.assertEqual(beh["status"], "ok")
        self.assertTrue(beh["flagged"])
        self.assertEqual(beh["deviation"], "HIGH")

    def test_7b_behavioural_insufficient_history_is_explicit(self):
        r = engine.analyze(
            from_address="totally-new-sender@nowhere.example", reply_to="totally-new-sender@nowhere.example",
            subject="Hello", body_text="Just checking in.",
        )
        self.assertEqual(r["signals"]["behavior"]["status"], "insufficient_history")

    def test_8_related_campaign(self):
        campaign_correlation.STORE.reset()
        shared = "pay a refundable verification fee and send your college password now"
        for eid, sender, subj in [
            ("m1", "recruiter@talent-portal.example", "Urgent - Final Internship Selection"),
            ("m2", "hr@talent-portal.example", "Urgent - Final Internship Selection round 2"),
        ]:
            r = engine.analyze(from_address=sender, reply_to=sender, subject=subj, body_text=shared)
            campaign_correlation.STORE.add_case({
                "id": eid, "student": eid, "department": "CSE",
                "sender_addr": r["parsed"]["sender_addr"], "from_domain": r["parsed"]["from_domain"],
                "origin_ip": r["parsed"]["origin_ip"], "subject": subj,
                "verdict_bucket": r["bucket"], "urls": [], "ts": "2026-09-10T10:00:00Z",
            })
        campaigns = campaign_correlation.STORE.list_campaigns()
        self.assertEqual(len(campaigns), 1)
        self.assertEqual(campaigns[0]["message_count"], 2)
        self.assertIn("same sending domain", campaigns[0]["reasons"])

    def test_9_malformed_input_does_not_crash(self):
        r = engine.analyze(from_address="not-an-email", reply_to="also not an email", subject="", body_text="")
        self.assertIn(r["bucket"], ("safe", "attention", "high-risk"))
        self.assertIsInstance(r["trust_score"], int)

    def test_regression_short_benign_email(self):
        """The exact false-positive the ML review caught must stay fixed
        when routed through the full contextual engine, not just the
        bare model."""
        r = engine.analyze(
            from_address="prof.sharma@college.edu", reply_to="prof.sharma@college.edu",
            subject="Class rescheduled to Friday",
            body_text="Hi all, our Monday class moves to Friday same time.",
        )
        self.assertEqual(r["bucket"], "safe")


class CampaignCorrelationUnitTests(unittest.TestCase):

    def test_unrelated_safe_email_excluded_from_campaign(self):
        campaign_correlation.STORE.reset()
        r1 = engine.analyze(from_address="a@bad-domain.example", reply_to="a@bad-domain.example",
                              subject="Verify now", body_text="Verify now urgent password required")
        r2 = engine.analyze(from_address="b@bad-domain.example", reply_to="b@bad-domain.example",
                              subject="Verify now please", body_text="Verify now urgent password required")
        r3 = engine.analyze(from_address="registrar@amityonline.edu", reply_to="registrar@amityonline.edu",
                              subject="Receipt", body_text="Thanks, here is your receipt.")
        for eid, r in [("a", r1), ("b", r2), ("c", r3)]:
            campaign_correlation.STORE.add_case({
                "id": eid, "student": eid, "department": "CSE",
                "sender_addr": r["parsed"]["sender_addr"], "from_domain": r["parsed"]["from_domain"],
                "origin_ip": r["parsed"]["origin_ip"], "subject": r["parsed"]["subject"],
                "verdict_bucket": r["bucket"], "urls": [], "ts": None,
            })
        campaigns = campaign_correlation.STORE.list_campaigns()
        all_members = {mid for c in campaigns for mid in c["related_message_ids"]}
        self.assertNotIn("c", all_members)


class FlaskAPITests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import app as flask_app
        cls.app = flask_app.app
        cls.flask_module = flask_app

    def setUp(self):
        self.client = self.app.test_client()
        self.client.post("/reset")

    def test_health(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["status"], "ok")

    def test_analyze_endpoint(self):
        resp = self.client.post("/analyze", json={
            "from_address": "alerts@paypal-secure-verify.com", "reply_to": "recovery@x.ru",
            "subject": "Verify now", "body_text": "verify now urgent password required",
        })
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertIn("trust_score", body)
        self.assertIn("bucket", body)

    def test_scan_endpoint_batch_and_campaign(self):
        resp = self.client.post("/scan", json={"emails": [
            {"id": "m1", "student": "Priya", "department": "CSE",
             "from_address": "x@bad.example", "reply_to": "x@bad.example",
             "subject": "Urgent internship selection", "body_text": "pay a fee and send your password now"},
            {"id": "m2", "student": "Rohan", "department": "ECE",
             "from_address": "y@bad.example", "reply_to": "y@bad.example",
             "subject": "Urgent internship selection final", "body_text": "pay a fee and send your password now"},
        ]})
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body["checked"], 2)
        self.assertGreaterEqual(body["campaigns_detected"], 1)

    def test_scan_endpoint_malformed_item_does_not_break_batch(self):
        resp = self.client.post("/scan", json={"emails": [
            {"id": "ok1", "from_address": "a@a.com", "subject": "hi", "body_text": "hello"},
            {"id": "bad1"},  # missing everything but id -- should still be handled gracefully
        ]})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["checked"], 2)

    def test_campaigns_endpoint(self):
        self.client.post("/scan", json={"emails": [
            {"id": "m1", "student": "Priya", "department": "CSE",
             "from_address": "x@bad.example", "reply_to": "x@bad.example",
             "subject": "Urgent internship selection", "body_text": "pay a fee and send your password now"},
            {"id": "m2", "student": "Rohan", "department": "ECE",
             "from_address": "y@bad.example", "reply_to": "y@bad.example",
             "subject": "Urgent internship selection final", "body_text": "pay a fee and send your password now"},
        ]})
        resp = self.client.get("/campaigns")
        self.assertEqual(resp.status_code, 200)
        camps = resp.get_json()["campaigns"]
        self.assertEqual(len(camps), 1)
        detail = self.client.get(f"/campaigns/{camps[0]['campaign_id']}")
        self.assertEqual(detail.status_code, 200)

    def test_campaign_not_found(self):
        resp = self.client.get("/campaigns/does-not-exist")
        self.assertEqual(resp.status_code, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
