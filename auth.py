"""
HALO auth — OAuth 2.0 authorization-code exchange.

This is genuine OAuth plumbing, not a placeholder: given real credentials
via environment variables, /auth/<provider>/callback performs an actual
token exchange with Google/Microsoft's token endpoints. In THIS
deployment no credentials are configured (HALO has no registered OAuth
app), so every call honestly returns oauth_not_configured -- it never
fabricates a successful session.

Required env vars to make a provider real:
  Google:     GOOGLE_OAUTH_CLIENT_ID, GOOGLE_OAUTH_CLIENT_SECRET
  Microsoft:  MICROSOFT_OAUTH_CLIENT_ID, MICROSOFT_OAUTH_CLIENT_SECRET,
              MICROSOFT_OAUTH_TENANT (defaults to "common")

Scopes are least-privilege and read-only, matching the "connect your
email" product framing -- HALO reads mail metadata/content to analyze
it, nothing more:
  Google:     openid email https://www.googleapis.com/auth/gmail.readonly
  Microsoft:  openid email offline_access Mail.Read
"""
import os
import json
import urllib.request
import urllib.parse
import urllib.error

PROVIDERS = {
    "google": {
        "token_url": "https://oauth2.googleapis.com/token",
        "client_id_env": "GOOGLE_OAUTH_CLIENT_ID",
        "client_secret_env": "GOOGLE_OAUTH_CLIENT_SECRET",
    },
    "microsoft": {
        "token_url": "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token",
        "client_id_env": "MICROSOFT_OAUTH_CLIENT_ID",
        "client_secret_env": "MICROSOFT_OAUTH_CLIENT_SECRET",
    },
}


def provider_configured(provider):
    p = PROVIDERS.get(provider)
    if not p:
        return False
    return bool(os.environ.get(p["client_id_env"]) and os.environ.get(p["client_secret_env"]))


def exchange_code_for_token(provider, code, redirect_uri):
    """Returns (ok, payload). payload is either the token response dict
    (ok=True) or an error dict (ok=False) -- NEVER fabricates a token."""
    p = PROVIDERS.get(provider)
    if not p:
        return False, {"error": "unknown_provider", "message": f"'{provider}' is not a supported provider."}

    client_id = os.environ.get(p["client_id_env"])
    client_secret = os.environ.get(p["client_secret_env"])
    if not client_id or not client_secret:
        return False, {
            "error": "oauth_not_configured",
            "message": f"{provider.capitalize()} OAuth is not configured on this server "
                        f"(missing {p['client_id_env']}/{p['client_secret_env']}). "
                        f"No account was linked.",
        }

    token_url = p["token_url"]
    if provider == "microsoft":
        tenant = os.environ.get("MICROSOFT_OAUTH_TENANT", "common")
        token_url = token_url.format(tenant=tenant)

    data = urllib.parse.urlencode({
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }).encode("utf-8")

    req = urllib.request.Request(token_url, data=data, method="POST",
                                    headers={"Content-Type": "application/x-www-form-urlencoded"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return True, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode("utf-8"))
        except Exception:
            body = {"error": "token_exchange_failed"}
        return False, {"error": body.get("error", "token_exchange_failed"),
                         "message": body.get("error_description", str(e))}
    except Exception as e:
        return False, {"error": "token_exchange_failed", "message": str(e)}
