"""Single-attempt Responses HTTP transport; no SDK retries or redirects."""
from .gemini_transport import GeminiTransport
from .types import ConfigurationError


class OpenAITransport(GeminiTransport):
    host = "api.openai.com"
    provider, api_family, path = "openai", "responses_v1", "/v1/responses"

    @classmethod
    def from_env(cls, credential_env="OPENAI_API_KEY"):
        return super().from_env(credential_env)

    def _headers(self):
        return {"Content-Type": "application/json", "Accept-Encoding": "identity",
                "Authorization": "Bearer " + self._api_key}

    def _validate_request(self, request, timeout):
        if (request.route.provider != self.provider or request.route.api_family != self.api_family
                or request.method != "POST" or request.path != self.path
                or not isinstance(request.body, bytes) or timeout <= 0):
            raise ConfigurationError("invalid native provider request")
