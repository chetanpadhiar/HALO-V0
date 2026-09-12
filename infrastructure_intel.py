"""
HALO contextual layer — infrastructure_intel.py

Looks up the observed origin IP against a small local reference table.
This stands in for a live threat-intel/WHOIS API call (VirusTotal,
AbuseIPDB, IPinfo) -- clearly labeled as such. Every result is framed
as "observed infrastructure", never as attacker location: a VPN, proxy,
or compromised relay changes this without changing who is really
sending the mail.
"""

INFRA_TABLE = {
    "185.220.101.47": {"asn": "AS208843", "org": "Offshore Hosting Solutions Ltd.", "country": "DE", "region": "Frankfurt", "category": "bulk-anonymous-hosting", "risk": 78},
    "191.101.203.19": {"asn": "AS262954", "org": "DataCenter Brasil Networks", "country": "BR", "region": "Sao Paulo", "category": "shared-vps", "risk": 62},
    "103.87.44.212": {"asn": "AS142804", "org": "QuickServe Cloud Hosting Pvt Ltd", "country": "IN", "region": "Mumbai", "category": "shared-vps", "risk": 66},
    "142.250.80.46": {"asn": "AS15169", "org": "Google Cloud Platform (verified mail infra)", "country": "US", "region": "Mountain View", "category": "established-mail-provider", "risk": 4},
    "103.21.58.10": {"asn": "AS45528", "org": "Amity Online Education Campus Network", "country": "IN", "region": "Noida", "category": "institutional-network", "risk": 6},
    "45.140.16.33": {"asn": "AS210644", "org": "AnonRelay Networks", "country": "NL", "region": "Amsterdam", "category": "bulk-anonymous-hosting", "risk": 74},
}


def lookup(ip):
    if not ip:
        return {
            "ip": None, "score": 0, "confidence": "none",
            "evidence": ["No routable origin IP found in the message"],
            "flagged": False,
        }
    entry = INFRA_TABLE.get(ip)
    if not entry:
        return {
            "ip": ip, "score": 15, "confidence": "low",
            "evidence": [f"IP {ip} is not in the local reference set (production build queries live threat-intel APIs)"],
            "flagged": False,
        }
    return {
        "ip": ip, "score": entry["risk"], "confidence": "low" if entry["risk"] < 40 else "medium",
        "asn": entry["asn"], "org": entry["org"], "country": entry["country"], "region": entry["region"],
        "category": entry["category"],
        "evidence": [
            f"Observed infrastructure: {entry['org']} ({entry['asn']}), {entry['region']}, {entry['country']}",
            f"Infrastructure category: {entry['category']}",
        ],
        "flagged": entry["risk"] >= 40,
    }
