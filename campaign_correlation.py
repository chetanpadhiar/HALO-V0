"""
HALO contextual layer — campaign_correlation.py

Groups analyzed messages into "campaigns" when they share sending
domain, origin IP/infrastructure, or near-duplicate subject/content.
Deterministic set-based correlation -- no ML, no external service --
because this is a graph-connectivity problem, not a classification
problem, and a simple deterministic method is easier to audit and
explain to a jury or an institution than a black-box one.
"""
import hashlib
import threading
from datetime import datetime, timezone


def _word_set(text):
    return set(w for w in (text or "").lower().split() if len(w) > 2)


def _jaccard(a, b):
    A, B = _word_set(a), _word_set(b)
    if not A or not B:
        return 0.0
    inter = len(A & B)
    return inter / (len(A) + len(B) - inter)


class CampaignStore:
    """In-memory case log + campaign correlation. Demo/local-run only;
    a production deployment would back this with a database scoped per
    institution, with retention limits (see product privacy notes)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.cases = []  # list of case dicts, one per analyzed email
        self.campaigns = {}  # campaign_id -> campaign dict

    def reset(self):
        with self._lock:
            self.cases = []
            self.campaigns = {}

    def add_case(self, case):
        """case: dict with at minimum {id, sender_addr, from_domain,
        origin_ip, subject, student, department, verdict_bucket, ts}."""
        with self._lock:
            self.cases.append(case)
            self._recompute_campaigns()
        return case

    def add_cases(self, cases):
        with self._lock:
            self.cases.extend(cases)
            self._recompute_campaigns()

    def _recompute_campaigns(self):
        """Union-find style grouping: two cases are linked if they share
        a domain, share an origin IP, or have near-duplicate subjects.
        Only messages HALO flagged as attention/high-risk participate --
        a coordinated campaign of entirely safe mail isn't a threat."""
        risky = [c for c in self.cases if c.get("verdict_bucket") in ("attention", "high-risk")]
        n = len(risky)
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        edge_reasons = {}
        for i in range(n):
            for j in range(i + 1, n):
                a, b = risky[i], risky[j]
                reason = None
                if a.get("from_domain") and a["from_domain"] == b.get("from_domain"):
                    reason = "same sending domain"
                elif a.get("origin_ip") and a["origin_ip"] == b.get("origin_ip"):
                    reason = "shared infrastructure (IP)"
                elif _jaccard(a.get("subject"), b.get("subject")) > 0.4:
                    reason = "near-duplicate subject"
                if reason:
                    union(i, j)
                    edge_reasons[(i, j)] = reason

        groups = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)

        self.campaigns = {}
        for root, idxs in groups.items():
            if len(idxs) < 2:
                continue  # a campaign needs at least 2 correlated messages
            members = [risky[i] for i in idxs]
            campaign_id = "camp_" + hashlib.sha1(
                "|".join(sorted(m["id"] for m in members)).encode()
            ).hexdigest()[:10]

            reasons = sorted({r for (i, j), r in edge_reasons.items() if i in idxs and j in idxs})
            domains = sorted({m["from_domain"] for m in members if m.get("from_domain")})
            ips = sorted({m["origin_ip"] for m in members if m.get("origin_ip")})
            students = sorted({m["student"] for m in members if m.get("student")})
            departments = sorted({m["department"] for m in members if m.get("department")})

            self.campaigns[campaign_id] = {
                "campaign_id": campaign_id,
                "name": self._name_for(members),
                "affected_users": students,
                "affected_departments": departments,
                "related_domains": domains,
                "related_urls": sorted({u for m in members for u in m.get("urls", [])}),
                "shared_infrastructure": ips,
                "related_message_ids": [m["id"] for m in members],
                "message_count": len(members),
                "reasons": reasons,
                "timeline": sorted(
                    [{"ts": m.get("ts"), "message_id": m["id"], "student": m.get("student")} for m in members],
                    key=lambda e: e["ts"] or "",
                ),
            }

    @staticmethod
    def _name_for(members):
        subjects = [m.get("subject", "") for m in members]
        if any("internship" in s.lower() or "placement" in s.lower() for s in subjects):
            return "Fake Internship / Placement Campaign"
        if any("scholarship" in s.lower() for s in subjects):
            return "Scholarship Impersonation Campaign"
        if any("password" in s.lower() or "verify" in s.lower() or "account" in s.lower() for s in subjects):
            return "Account Verification Phishing Campaign"
        return "Correlated Suspicious Message Campaign"

    def list_campaigns(self):
        return list(self.campaigns.values())

    def get_campaign(self, campaign_id):
        return self.campaigns.get(campaign_id)


STORE = CampaignStore()
