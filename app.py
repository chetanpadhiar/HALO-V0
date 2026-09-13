"""
HALO 2.0 backend service.

This is the authoritative pipeline: email -> parser -> feature
extraction -> trained ML model -> contextual trust engine -> verdict
-> campaign correlation. The frontend calls this over HTTP; no scoring
logic of any kind runs in the browser.

Run (local dev):
    cd halo_ml/service
    python3 app.py
    # serves http://localhost:8787 (or $PORT if set)

Run (production, e.g. Render):
    gunicorn app:app --bind 0.0.0.0:$PORT
    # PORT is provided by the platform; ALLOWED_ORIGINS should be set
    # to the deployed frontend's origin(s), comma-separated.

Endpoints:
    GET  /health                    service + model status
    POST /analyze                   analyze one email -> full verdict
    POST /scan                      analyze a batch of emails (the
                                      demo "inbox scan") -> verdicts +
                                      updates campaign correlation
    GET  /campaigns                  list currently correlated campaigns
    GET  /campaigns/<id>              one campaign's full detail
    POST /reset                      clear the in-memory case log /
                                      campaigns / behavioural history
                                      (demo reset)
"""
import sys
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from flask import Flask, request, jsonify

import contextual_trust_engine as engine
import campaign_correlation
import behavioral_baseline
import auth as halo_auth

META_PATH = os.path.join(BASE_DIR, "halo_trust_model_v3.metadata.json")
with open(META_PATH) as f:
    METADATA = json.load(f)

app = Flask(__name__)

# CORS: origins are restricted via an env var in production, not a blanket
# wildcard. ALLOWED_ORIGINS is a comma-separated list (e.g. the deployed
# Vercel frontend's origin). If unset (local dev), any origin is allowed
# for convenience -- there is no cookie/session auth on this API, so a
# permissive default in dev carries no credential-leak risk; production
# should always set ALLOWED_ORIGINS explicitly.
_allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "").strip()
ALLOWED_ORIGINS = [o.strip() for o in _allowed_origins_env.split(",") if o.strip()]


@app.after_request
def add_cors_headers(resp):
    origin = request.headers.get("Origin")
    if ALLOWED_ORIGINS:
        if origin in ALLOWED_ORIGINS:
            resp.headers["Access-Control-Allow-Origin"] = origin
            resp.headers["Vary"] = "Origin"
    else:
        # No ALLOWED_ORIGINS configured (local dev) -- reflect any origin,
        # or "*" for non-browser callers (e.g. curl) with no Origin header.
        resp.headers["Access-Control-Allow-Origin"] = origin or "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.route("/analyze", methods=["OPTIONS"])
@app.route("/scan", methods=["OPTIONS"])
@app.route("/reset", methods=["OPTIONS"])
@app.route("/auth/<provider>/callback", methods=["OPTIONS"])
def _preflight(provider=None):
    return "", 204


def _case_from_result(email_id, meta, result):
    """Build the compact record campaign_correlation needs from one
    engine.analyze() result plus the caller-supplied demo metadata
    (student/department -- these come from the frontend's demo
    dataset, never from the ML pipeline itself)."""
    return {
        "id": email_id,
        "student": meta.get("student"),
        "department": meta.get("department"),
        "sender_addr": result["parsed"]["sender_addr"],
        "from_domain": result["parsed"]["from_domain"],
        "origin_ip": result["parsed"]["origin_ip"],
        "subject": result["parsed"]["subject"],
        "verdict_bucket": result["bucket"],
        "urls": [],
        "ts": meta.get("ts"),
    }


def _analyze_payload(payload):
    """Shared logic for /analyze and /scan: run one email through the
    real engine given a JSON payload (raw_source OR individual fields)."""
    if payload.get("raw_source"):
        result = engine.analyze(raw_source=payload["raw_source"])
    else:
        result = engine.analyze(
            from_address=payload.get("from_address", ""),
            reply_to=payload.get("reply_to", ""),
            subject=payload.get("subject", ""),
            body_text=payload.get("body_text", ""),
        )
    return result


@app.route("/auth/<provider>/status", methods=["GET"])
def auth_status(provider):
    return jsonify({"provider": provider, "configured": halo_auth.provider_configured(provider)})


@app.route("/auth/<provider>/callback", methods=["POST"])
def auth_callback(provider):
    payload = request.get_json(force=True, silent=True) or {}
    code = payload.get("code")
    redirect_uri = payload.get("redirect_uri")
    if not code or not redirect_uri:
        return jsonify({"error": "missing_params", "message": "code and redirect_uri are required"}), 400
    ok, result = halo_auth.exchange_code_for_token(provider, code, redirect_uri)
    if not ok:
        # Honest failure -- e.g. oauth_not_configured. Never a fake session.
        return jsonify(result), 501 if result.get("error") == "oauth_not_configured" else 400
    # A real token response would land here; session/profile creation
    # (mapping to a Supabase `student` or `institution` row per the
    # role captured at role-selection time) is the next real step once
    # a provider is actually configured.
    return jsonify({"status": "token_exchange_ok", "note": "session creation not yet implemented"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_name": METADATA["model_name"],
        "feature_set": METADATA.get("feature_set"),
        "training_date": METADATA["training_date"],
        "test_metrics": METADATA["final_test_metrics"],
        "majority_class_baseline_test": METADATA.get("majority_class_baseline_test"),
        "known_limitations": METADATA.get("known_limitations", []),
        "cases_this_session": len(campaign_correlation.STORE.cases),
        "campaigns_detected": len(campaign_correlation.STORE.campaigns),
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    payload = request.get_json(force=True, silent=True) or {}
    result = _analyze_payload(payload)

    # A single ad-hoc /analyze call still joins the session's case log
    # (with whatever student/department metadata the caller supplies,
    # if any) so it can participate in campaign correlation too.
    email_id = payload.get("id") or f"adhoc_{len(campaign_correlation.STORE.cases) + 1}"
    case = _case_from_result(email_id, payload, result)
    campaign_correlation.STORE.add_case(case)

    result["id"] = email_id
    return jsonify(result)


@app.route("/scan", methods=["POST"])
def scan():
    """Batch endpoint: the frontend's demo 'inbox scan' posts its whole
    demo email set here in one call and gets real per-email verdicts
    back, computed by the SAME engine as a single /analyze call -- this
    is what makes the scan real instead of a client-side simulation."""
    payload = request.get_json(force=True, silent=True) or {}
    emails = payload.get("emails", [])
    if not isinstance(emails, list):
        return jsonify({"error": "expected {'emails': [...]}"}), 400

    results = []
    cases = []
    for item in emails:
        try:
            result = _analyze_payload(item)
        except Exception as exc:  # a malformed single email must not fail the whole batch
            results.append({"id": item.get("id"), "error": str(exc)})
            continue
        email_id = item.get("id") or f"scan_{len(results) + 1}"
        result["id"] = email_id
        results.append(result)
        cases.append(_case_from_result(email_id, item, result))

    campaign_correlation.STORE.add_cases(cases)

    counts = {"safe": 0, "attention": 0, "high-risk": 0}
    for r in results:
        if r.get("bucket") in counts:
            counts[r["bucket"]] += 1

    return jsonify({
        "checked": len(results),
        "counts": counts,
        "results": results,
        "campaigns_detected": len(campaign_correlation.STORE.campaigns),
    })


@app.route("/campaigns", methods=["GET"])
def list_campaigns():
    return jsonify({"campaigns": campaign_correlation.STORE.list_campaigns()})


@app.route("/campaigns/<campaign_id>", methods=["GET"])
def get_campaign(campaign_id):
    camp = campaign_correlation.STORE.get_campaign(campaign_id)
    if not camp:
        return jsonify({"error": "not found"}), 404
    return jsonify(camp)


@app.route("/reset", methods=["POST"])
def reset():
    campaign_correlation.STORE.reset()
    behavioral_baseline.STORE.__init__()  # reseed demo baseline, drop session-recorded history
    return jsonify({"status": "reset"})


if __name__ == "__main__":
    print(f"Loaded {METADATA['model_name']} (trained {METADATA['training_date']})")
    print(f"Test-set metrics: {METADATA['final_test_metrics']}")
    # PORT is set automatically by Render (and most PaaS providers) in
    # production; 8787 is only the local-dev fallback. Binding 0.0.0.0
    # (not 127.0.0.1) is required for the process to be reachable from
    # outside its own container. Note: this app.run() path is the local
    # dev server only -- production uses gunicorn via the Start Command
    # (see render.yaml / deployment notes), which imports `app` directly
    # and never executes this block.
    port = int(os.environ.get("PORT", 8787))
    app.run(host="0.0.0.0", port=port, debug=False)
