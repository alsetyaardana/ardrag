"""OAuth 2.1 authorization server for the MCP endpoint, so it can be added as a Claude.ai
Custom Connector (which requires OAuth — plain unauthenticated SSE, used by Claude Code, is
not accepted there).

Unlike FastMCP's built-in `InMemoryOAuthProvider` (explicitly for testing — its `authorize()`
auto-approves any registered client with no login check at all, and state is lost on restart),
this actually gates access behind Ardrag's own ADMIN_USER/ADMIN_PASSWORD and persists clients/
tokens in SQLite so a container redeploy doesn't silently disconnect every previously-approved
client.

Flow:
  1. Claude.ai POSTs to /register (handled by FastMCP's OAuthProvider routing) -> register_client()
  2. Claude.ai GETs /authorize -> our authorize() redirects to our own /oauth-login page instead
     of auto-approving, carrying the OAuth params along as a query string.
  3. /oauth-login (custom_route, registered in mcp_server.py) shows a login form; on successful
     POST it creates the authorization code itself and 302s to the client's redirect_uri.
  4. Claude.ai POSTs to /token (handled by FastMCP routing) -> exchange_authorization_code() /
     exchange_refresh_token().
"""

import secrets
import time
from urllib.parse import urlencode

from fastmcp.server.auth.auth import OAuthProvider
from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    RefreshToken,
    TokenError,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

from ardrag import db

DEFAULT_ACCESS_TOKEN_EXPIRY_SECONDS = 60 * 60
DEFAULT_REFRESH_TOKEN_EXPIRY_SECONDS = 30 * 24 * 3600
OAUTH_LOGIN_PATH = "/oauth-login"


def _client_dict_to_model(c: dict) -> OAuthClientInformationFull:
    return OAuthClientInformationFull(
        client_id=c["client_id"],
        client_secret=c["client_secret"],
        redirect_uris=c["redirect_uris"],
        client_name=c["client_name"],
        grant_types=c["grant_types"] or ["authorization_code", "refresh_token"],
        response_types=c["response_types"] or ["code"],
        token_endpoint_auth_method=c["token_endpoint_auth_method"] or "none",
        scope=c["scope"],
        client_id_issued_at=c["client_id_issued_at"],
    )


class ArdragOAuthProvider(OAuthProvider):
    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        c = db.oauth_get_client(client_id)
        return _client_dict_to_model(c) if c else None

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        if not client_info.client_id:
            raise ValueError("client_id is required for client registration")
        db.oauth_save_client(
            client_id=client_info.client_id,
            client_secret=client_info.client_secret,
            redirect_uris=[str(u) for u in client_info.redirect_uris],
            client_name=client_info.client_name,
            grant_types=list(client_info.grant_types or []),
            response_types=list(client_info.response_types or []),
            token_endpoint_auth_method=client_info.token_endpoint_auth_method,
            scope=client_info.scope,
            client_id_issued_at=client_info.client_id_issued_at,
        )

    async def authorize(self, client: OAuthClientInformationFull, params) -> str:
        # Deliberately does NOT auto-approve. Send the browser to our real login page —
        # /oauth-login validates against ADMIN_USER/ADMIN_PASSWORD and issues the code itself.
        # `resource` (RFC 8707) must be carried through: MCP clients bind the issued token to
        # this resource URL and reject it client-side if the token isn't scoped to it.
        query = urlencode(
            {
                "client_id": client.client_id,
                "redirect_uri": str(params.redirect_uri),
                "redirect_uri_provided_explicitly": str(bool(params.redirect_uri_provided_explicitly)),
                "state": params.state or "",
                "code_challenge": params.code_challenge,
                "scope": " ".join(params.scopes or []),
                "resource": params.resource or "",
            }
        )
        return f"{OAUTH_LOGIN_PATH}?{query}"

    async def load_authorization_code(self, client: OAuthClientInformationFull, authorization_code: str) -> AuthorizationCode | None:
        code = db.oauth_load_auth_code(authorization_code)
        if not code or code["client_id"] != client.client_id:
            return None
        if code["expires_at"] < time.time():
            db.oauth_delete_auth_code(authorization_code)
            return None
        return AuthorizationCode(
            code=code["code"],
            client_id=code["client_id"],
            redirect_uri=code["redirect_uri"],
            redirect_uri_provided_explicitly=code["redirect_uri_provided_explicitly"],
            scopes=code["scopes"],
            expires_at=code["expires_at"],
            code_challenge=code["code_challenge"],
            resource=code.get("resource"),
        )

    async def exchange_authorization_code(self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode) -> OAuthToken:
        stored = db.oauth_load_auth_code(authorization_code.code)
        if not stored:
            raise TokenError("invalid_grant", "Authorization code not found or already used.")
        db.oauth_delete_auth_code(authorization_code.code)

        access_token_value = secrets.token_urlsafe(32)
        refresh_token_value = secrets.token_urlsafe(32)
        access_expires = int(time.time() + DEFAULT_ACCESS_TOKEN_EXPIRY_SECONDS)
        refresh_expires = int(time.time() + DEFAULT_REFRESH_TOKEN_EXPIRY_SECONDS)
        subject = stored["subject"]
        resource = stored.get("resource")

        db.oauth_save_access_token(
            access_token_value, client.client_id, stored["scopes"], access_expires, subject, resource
        )
        db.oauth_save_refresh_token(
            refresh_token_value, client.client_id, stored["scopes"], refresh_expires, subject, access_token_value
        )

        return OAuthToken(
            access_token=access_token_value,
            token_type="Bearer",
            expires_in=DEFAULT_ACCESS_TOKEN_EXPIRY_SECONDS,
            refresh_token=refresh_token_value,
            scope=" ".join(stored["scopes"]),
        )

    async def load_refresh_token(self, client: OAuthClientInformationFull, refresh_token: str) -> RefreshToken | None:
        token = db.oauth_load_refresh_token(refresh_token)
        if not token or token["client_id"] != client.client_id:
            return None
        if token["expires_at"] is not None and token["expires_at"] < time.time():
            db.oauth_delete_refresh_token(refresh_token)
            return None
        return RefreshToken(token=token["token"], client_id=token["client_id"], scopes=token["scopes"], expires_at=token["expires_at"])

    async def exchange_refresh_token(self, client: OAuthClientInformationFull, refresh_token: RefreshToken, scopes: list[str]) -> OAuthToken:
        stored = db.oauth_load_refresh_token(refresh_token.token)
        if not stored:
            raise TokenError("invalid_grant", "Refresh token not found.")
        original_scopes = set(stored["scopes"])
        use_scopes = scopes or stored["scopes"]
        if not set(use_scopes).issubset(original_scopes):
            raise TokenError("invalid_scope", "Requested scopes exceed those authorized by the refresh token.")

        # Rotate: invalidate old pair, issue new pair. Preserve the resource binding from the
        # old access token, if any, so a refreshed token stays valid for the same MCP resource.
        old_resource = None
        if stored.get("access_token"):
            old_access = db.oauth_load_access_token(stored["access_token"])
            old_resource = old_access.get("resource") if old_access else None
            db.oauth_delete_access_token(stored["access_token"])
        db.oauth_delete_refresh_token(refresh_token.token)

        new_access = secrets.token_urlsafe(32)
        new_refresh = secrets.token_urlsafe(32)
        access_expires = int(time.time() + DEFAULT_ACCESS_TOKEN_EXPIRY_SECONDS)
        refresh_expires = int(time.time() + DEFAULT_REFRESH_TOKEN_EXPIRY_SECONDS)
        subject = stored["subject"]

        db.oauth_save_access_token(new_access, client.client_id, use_scopes, access_expires, subject, old_resource)
        db.oauth_save_refresh_token(new_refresh, client.client_id, use_scopes, refresh_expires, subject, new_access)

        return OAuthToken(
            access_token=new_access,
            token_type="Bearer",
            expires_in=DEFAULT_ACCESS_TOKEN_EXPIRY_SECONDS,
            refresh_token=new_refresh,
            scope=" ".join(use_scopes),
        )

    async def load_access_token(self, token: str) -> AccessToken | None:
        stored = db.oauth_load_access_token(token)
        if not stored:
            return None
        if stored["expires_at"] is not None and stored["expires_at"] < time.time():
            db.oauth_delete_access_token(token)
            return None
        return AccessToken(
            token=stored["token"],
            client_id=stored["client_id"],
            scopes=stored["scopes"],
            expires_at=stored["expires_at"],
            resource=stored.get("resource"),
        )

    async def verify_token(self, token: str) -> AccessToken | None:
        return await self.load_access_token(token)

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        if isinstance(token, AccessToken):
            db.oauth_delete_access_token(token.token)
        elif isinstance(token, RefreshToken):
            db.oauth_delete_refresh_token(token.token)


def issue_code_and_redirect(
    client_id: str,
    redirect_uri: str,
    redirect_uri_provided_explicitly: bool,
    state: str | None,
    code_challenge: str,
    scope: str,
    subject: str,
    resource: str | None = None,
) -> str:
    """Called by the /oauth-login route after a successful username/password check. Validates
    the redirect_uri against the registered client (prevents open-redirect via a forged query
    string), stores the authorization code, and returns the final redirect URL."""
    client = db.oauth_get_client(client_id)
    if not client or redirect_uri not in client["redirect_uris"]:
        raise ValueError("Unknown client or redirect_uri not registered for this client.")

    code_value = secrets.token_urlsafe(32)
    expires_at = int(time.time() + 300)  # 5 minutes, matches standard auth-code lifetime
    scopes = scope.split() if scope else []
    db.oauth_save_auth_code(
        code_value,
        client_id,
        redirect_uri,
        redirect_uri_provided_explicitly,
        code_challenge,
        scopes,
        expires_at,
        subject,
        resource,
    )
    return construct_redirect_uri(redirect_uri, code=code_value, state=state)
