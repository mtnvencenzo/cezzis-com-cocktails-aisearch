import logging
from typing import Any

import httpx
from jose import JWTError, jwt

from cezzis_com_cocktails_aisearch.domain.config.auth0_options import Auth0Options, get_auth0_options

_logger = logging.getLogger("oauth_verification")


class TokenVerificationError(Exception):
    """Exception raised when token verification fails."""

    pass


class OAuth2TokenVerifier:
    """Verifies OAuth2 JWT tokens from Auth0."""

    def __init__(self):
        self.auth0_options: Auth0Options = get_auth0_options()
        self._jwks_cache: dict[str, Any] | None = None

    async def get_jwks(self) -> dict[str, Any]:
        """Fetch JWKS (JSON Web Key Set) from Auth0.

        Returns:
            dict: The JWKS data containing public keys.
        """
        if self._jwks_cache:
            return self._jwks_cache

        jwks_url = f"https://{self.auth0_options.domain}/.well-known/jwks.json"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(jwks_url, timeout=10.0)
                response.raise_for_status()
                self._jwks_cache = response.json()
                assert self._jwks_cache is not None
                return self._jwks_cache
        except Exception as e:
            _logger.error(f"Failed to fetch JWKS from Auth0: {e}")
            raise TokenVerificationError("Unable to fetch JWKS for token verification")

    def _get_signing_key(self, token: str, jwks: dict[str, Any]) -> str:
        """Extract the signing key from JWKS based on the token's kid (key ID).

        Args:
            token: The JWT token
            jwks: The JWKS data

        Returns:
            str: The RSA public key for verification

        Raises:
            TokenVerificationError: If the signing key is not found
        """
        try:
            unverified_header = jwt.get_unverified_header(token)
            kid = unverified_header.get("kid")

            if not kid:
                raise TokenVerificationError("Token header missing 'kid' field")

            # Find the key with matching kid
            for key in jwks.get("keys", []):
                if key.get("kid") == kid:
                    # Construct RSA public key from JWK
                    from jose.backends.cryptography_backend import CryptographyRSAKey

                    rsa_key = CryptographyRSAKey(key, algorithm="RS256")
                    return rsa_key.to_pem().decode()

            raise TokenVerificationError(f"Unable to find signing key with kid: {kid}")
        except JWTError as e:
            _logger.error(f"Error extracting signing key: {e}")
            raise TokenVerificationError("Invalid token header")

    async def verify_token(self, token: str) -> dict[str, Any]:
        """Verify an Auth0 JWT token and return its payload.

        Args:
            token: The JWT token to verify

        Returns:
            dict: The decoded token payload containing claims

        Raises:
            TokenVerificationError: If verification fails
        """
        if not self.auth0_options.domain or not self.auth0_options.api_audience:
            raise TokenVerificationError("Auth0 configuration is incomplete")

        try:
            # Get JWKS
            jwks = await self.get_jwks()

            # Get the signing key
            rsa_key = self._get_signing_key(token, jwks)

            # Decode and verify the token
            payload = jwt.decode(
                token,
                rsa_key,
                algorithms=self.auth0_options.algorithms,
                audience=self.auth0_options.api_audience,
                issuer=self.auth0_options.issuer,
            )

            return payload

        except JWTError as e:
            _logger.warning(f"JWT verification failed: {e}")
            raise TokenVerificationError(f"Token verification failed: {str(e)}")
        except Exception as e:
            _logger.error(f"Unexpected error during token verification: {e}")
            raise TokenVerificationError("Token verification failed")

    def verify_scopes(self, payload: dict[str, Any], required_scopes: list[str]) -> bool:
        """Verify that the token has all required scopes.

        Args:
            payload: The decoded JWT payload
            required_scopes: List of scope strings that must be present

        Returns:
            bool: True if all required scopes are present

        Raises:
            TokenVerificationError: If required scopes are missing
        """
        if not required_scopes:
            return True

        # Auth0 typically puts scopes in the 'scope' claim as a space-separated string
        token_scopes_str = payload.get("scope", "")
        token_scopes = token_scopes_str.split() if token_scopes_str else []

        # Alternatively, scopes might be in 'permissions' array
        if not token_scopes and "permissions" in payload:
            token_scopes = payload.get("permissions", [])

        missing_scopes = [scope for scope in required_scopes if scope not in token_scopes]

        if missing_scopes:
            _logger.warning(f"Token missing required scopes: {missing_scopes}")
            raise TokenVerificationError(f"Missing required scopes: {', '.join(missing_scopes)}")

        return True


# Singleton instance
_verifier: OAuth2TokenVerifier | None = None


def get_token_verifier() -> OAuth2TokenVerifier:
    """Get the singleton OAuth2TokenVerifier instance.

    Returns:
        OAuth2TokenVerifier: The token verifier instance
    """
    global _verifier
    if _verifier is None:
        _verifier = OAuth2TokenVerifier()
    return _verifier
