"""
HALO contextual layer — behavioral_baseline.py

Tracks what "normal" looks like per sender, so a request that departs
from a sender's own history can be flagged even when every technical
check passes. This is in-memory for the demo/local-run (a real
deployment would back this with a database keyed by institution +
sender). If a sender has no history, this returns an explicit
"insufficient_history" state -- it never invents a baseline.

Seed data below is clearly synthetic demo data for one fictional
campus (matches the scenario already used in the frontend prototype),
NOT real observed behavior.
"""
import threading
from datetime import datetime, timezone

CREDENTIAL_WORDS = ["password", "otp", "verification code", "login", "credentials", "security code"]
PAYMENT_WORDS = ["wire transfer", "bank details", "processing fee", "gift card", "account number", "payment", "refundable fee"]


class BehaviorStore:
    """Thread-safe in-memory sender history. Demo/local-run only."""

    def __init__(self):
        self._lock = threading.Lock()
        self._history = {}  # sender_addr -> list of {subject, requested_credential, requested_payment, ts}
        self._seed_demo_data()

    def _seed_demo_data(self):
        """SYNTHETIC DEMO DATA for one fictional campus, clearly labeled
        as such wherever it's surfaced. Represents a plausible prior
        relationship so the demo's account-takeover scenario has a
        baseline to deviate from."""
        demo_seed = {
            "placements@amityonline.edu": [
                {"subject": "Interview schedule for CSE batch", "requested_credential": False, "requested_payment": False},
                {"subject": "Document submission reminder", "requested_credential": False, "requested_payment": False},
                {"subject": "Offer letter released", "requested_credential": False, "requested_payment": False},
            ],
            "it-helpdesk@amityonline.edu": [
                {"subject": "Password reset window this weekend", "requested_credential": False, "requested_payment": False},
                {"subject": "Device registration required for wifi", "requested_credential": False, "requested_payment": False},
            ],
            "registrar@amityonline.edu": [
                {"subject": "Fee receipt attached", "requested_credential": False, "requested_payment": False},
                {"subject": "Transcript request processed", "requested_credential": False, "requested_payment": False},
            ],
        }
        for sender, msgs in demo_seed.items():
            self._history[sender] = [{**m, "ts": None, "demo_seed": True} for m in msgs]

    def record(self, sender_addr, subject, requested_credential, requested_payment):
        if not sender_addr:
            return
        with self._lock:
            self._history.setdefault(sender_addr, []).append({
                "subject": subject, "requested_credential": requested_credential,
                "requested_payment": requested_payment,
                "ts": datetime.now(timezone.utc).isoformat(), "demo_seed": False,
            })

    def lookup(self, sender_addr, subject, body):
        if not sender_addr or sender_addr not in self._history or not self._history[sender_addr]:
            return {
                "status": "insufficient_history",
                "seen_before": False,
                "evidence": ["Sender has no prior communication history in this baseline"],
                "flagged": False,
                "score": 20,  # unfamiliar sender is a mild, not damning, signal on its own
            }

        prior = self._history[sender_addr]
        text = f"{subject} {body}".lower()
        requests_credential = any(w in text for w in CREDENTIAL_WORDS)
        requests_payment = any(w in text for w in PAYMENT_WORDS)

        prior_credential_count = sum(1 for m in prior if m["requested_credential"])
        prior_payment_count = sum(1 for m in prior if m["requested_payment"])

        evidence = [f"Sender has {len(prior)} prior message(s) in this baseline" +
                    (" (demo seed data)" if all(m.get("demo_seed") for m in prior) else "")]
        score = 0
        deviation = "NONE"

        if requests_credential and prior_credential_count == 0:
            score += 45
            deviation = "HIGH"
            evidence.append("This sender has never previously requested credentials -- this message does")
        if requests_payment and prior_payment_count == 0:
            score += 45
            deviation = "HIGH"
            evidence.append("This sender has never previously requested payment -- this message does")
        if not evidence[1:]:
            evidence.append("Request type is consistent with this sender's prior messages")

        return {
            "status": "ok",
            "seen_before": True,
            "prior_message_count": len(prior),
            "prior_credential_requests": prior_credential_count,
            "prior_payment_requests": prior_payment_count,
            "current_requests_credential": requests_credential,
            "current_requests_payment": requests_payment,
            "deviation": deviation,
            "evidence": evidence,
            "flagged": score >= 40,
            "score": min(100, score),
        }


STORE = BehaviorStore()
