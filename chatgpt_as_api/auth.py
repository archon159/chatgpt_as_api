"""Sign in with ChatGPT: OAuth login, token storage and refresh.

Implements the open-source app flow documented at
developers.openai.com/siwc/token-sharing-open-source/sign-in: authorization code + PKCE,
first-time registration with client_id "dynamic_agent_client" (the callback returns the issued
client ID), loopback redirect on 127.0.0.1:1455, and the scope chatgpt.tokens.use.direct that
lets Responses API requests run on the user's ChatGPT plan.
"""
from __future__ import annotations

import base64
import hashlib
import http.server
import json
import os
import secrets
import threading
import time
import urllib.parse
import uuid
from pathlib import Path

import httpx

AUTHORIZE_URL = "https://auth.openai.com/api/accounts/authorize"
TOKEN_URL = "https://auth.openai.com/api/accounts/oauth/token"
RESOURCE = "https://api.openai.com/v1"
SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
PLAN_SCOPE = "chatgpt.tokens.use.direct"
DEFAULT_AUTH_PATH = Path(os.environ.get("CHATGPT_AS_API_AUTH") or Path.home() / ".config" / "chatgpt_as_api" / "auth.json")
REFRESH_MARGIN = 120  # seconds before expiry at which the access token is refreshed


class AuthError(RuntimeError):
    pass


class AuthStore:
    """Token file holding the issued client ID, the host ID and the current tokens."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else DEFAULT_AUTH_PATH
        self.data: dict = json.loads(self.path.read_text()) if self.path.exists() else {}
        self._lock = threading.Lock()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1))
        self.path.chmod(0o600)

    @property
    def logged_in(self) -> bool:
        return bool(self.data.get("tokens", {}).get("access_token"))

    @property
    def scopes(self) -> list[str]:
        return (self.data.get("tokens", {}).get("scope") or "").split()

    @property
    def plan_usage(self) -> bool:
        return PLAN_SCOPE in self.scopes

    def expires_at(self) -> float:
        tokens = self.data.get("tokens", {})
        return self.data.get("obtained_at", 0) + tokens.get("expires_in", 3600)

    def access_token(self) -> str:
        """A valid access token, refreshed when close to expiry."""
        if not self.logged_in:
            raise AuthError(f"not logged in: run `chatgpt_as_api login` (token file {self.path})")
        with self._lock:
            if time.time() > self.expires_at() - REFRESH_MARGIN:
                self.refresh()
            return self.data["tokens"]["access_token"]

    def id_claims(self) -> dict:
        """Claims of the ID token (email, etc.), without signature verification."""
        token = self.data.get("tokens", {}).get("id_token")
        if not token:
            return {}
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload))

    def refresh(self) -> None:
        refresh_token = self.data.get("tokens", {}).get("refresh_token")
        if not refresh_token:
            raise AuthError("no refresh token saved: run `chatgpt_as_api login`")
        r = httpx.post(TOKEN_URL, data={"grant_type": "refresh_token", "client_id": self.data["client_id"],
                                        "refresh_token": refresh_token, "resource": RESOURCE}, timeout=60)
        if r.status_code != 200:
            raise AuthError(f"token refresh failed ({r.status_code}): {r.text[:500]}; run `chatgpt_as_api login` again")
        self.data["tokens"] = r.json()
        self.data["obtained_at"] = time.time()
        self.save()

    def login(self, agent_name: str = "chatgpt_as_api", port: int = 1455, timeout: float = 600, echo=print) -> None:
        """Interactive OAuth login: prints the URL to open, waits for the loopback callback."""
        redirect_uri = f"http://127.0.0.1:{port}/auth/callback"
        host_id = self.data.get("ext_agent_host_id") or f"urn:uuid:{uuid.uuid4()}"
        client_id = self.data.get("client_id") or "dynamic_agent_client"
        verifier = base64.urlsafe_b64encode(secrets.token_bytes(64)).rstrip(b"=").decode()
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        state, nonce = secrets.token_urlsafe(24), secrets.token_urlsafe(24)
        params = {"response_type": "code", "client_id": client_id, "ext_agent_host_id": host_id,
                  "redirect_uri": redirect_uri, "scope": SCOPES, "resource": RESOURCE, "state": state, "nonce": nonce,
                  "code_challenge": challenge, "code_challenge_method": "S256"}
        if client_id == "dynamic_agent_client":
            params["agent_name_hint"] = agent_name
        url = AUTHORIZE_URL + "?" + urllib.parse.urlencode(params)
        echo(f"Open this URL in a browser that can reach 127.0.0.1:{port} of this machine:\n\n{url}\n")
        echo(f"waiting for the callback on {redirect_uri} ...")
        callback = _wait_for_callback(port, state, timeout)
        issued = callback.get("client_id") or (client_id if client_id != "dynamic_agent_client" else None)
        if not issued:
            raise AuthError("callback carried no client_id: registration incomplete")
        r = httpx.post(TOKEN_URL, data={"grant_type": "authorization_code", "client_id": issued, "code": callback["code"],
                                        "code_verifier": verifier, "redirect_uri": redirect_uri, "resource": RESOURCE},
                       timeout=60)
        if r.status_code != 200:
            raise AuthError(f"token exchange failed ({r.status_code}): {r.text[:500]}")
        self.data = {"client_id": issued, "ext_agent_host_id": host_id, "tokens": r.json(), "obtained_at": time.time()}
        self.save()
        echo(f"saved {self.path}")
        echo(f"signed in as {self.id_claims().get('email', '?')}; plan usage: {self.plan_usage}")
        if not self.plan_usage:
            echo(f"warning: scope {PLAN_SCOPE} was not granted; requests will not run on the ChatGPT plan")


def _wait_for_callback(port: int, expected_state: str, timeout: float) -> dict:
    result: dict = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            url = urllib.parse.urlparse(self.path)
            if url.path != "/auth/callback":
                self.send_response(404)
                self.end_headers()
                return
            result.update({k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()})
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Signed in. You can close this tab.")

        def log_message(self, *_):
            pass

    server = http.server.HTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    deadline = time.time() + timeout
    while not result and time.time() < deadline:
        time.sleep(0.5)
    server.shutdown()
    if not result:
        raise AuthError("no callback received within the timeout")
    if result.get("error"):
        raise AuthError(f"authorization error: {result['error']} {result.get('error_description', '')}")
    if result.get("state") != expected_state:
        raise AuthError("state mismatch in the callback")
    return result
