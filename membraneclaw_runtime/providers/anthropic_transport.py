"""Messages REST transport; fixed version header and no automatic attempts."""
from .openai_transport import OpenAITransport


class AnthropicTransport(OpenAITransport):
    host = "api.anthropic.com"
    provider, api_family, path = "anthropic", "messages_v1", "/v1/messages"

    @classmethod
    def from_env(cls, credential_env="ANTHROPIC_API_KEY"):
        return super().from_env(credential_env)

    def _headers(self):
        return {"Content-Type": "application/json", "Accept-Encoding": "identity",
                "x-api-key": self._api_key, "anthropic-version": "2023-06-01"}
