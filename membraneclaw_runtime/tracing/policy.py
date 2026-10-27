"""Reject known authentication data before persistence; never silently redact evidence."""
import json

from .schema import TraceError, canonical_json


class CredentialRejected(TraceError):
    pass


class CapturePolicy:
    """Transport owners must register actual credential VALUES, excluding signatures.

    This is defense in depth, not a general sensitive-data classifier. Prompts and
    provider signatures remain private business data. Never pass auth headers to
    the recorder. Unknown credential strings cannot be reliably detected.
    """
    AUTH_KEYS = {"authorization", "proxyauthorization", "apikey", "accesstoken",
                 "refreshtoken", "clientsecret", "password", "cookie", "setcookie"}

    def __init__(self, authentication_values=()):
        self._values = tuple(v.encode() if isinstance(v, str) else v for v in authentication_values if v)
        if any(not isinstance(v, bytes) for v in self._values):
            raise ValueError("authentication values must be strings/bytes")

    def register_authentication_values(self, values):
        extra = CapturePolicy(values)
        self._values = tuple(dict.fromkeys(self._values + extra._values))

    def check(self, value):
        raw = value if isinstance(value, bytes) else canonical_json(value)
        if any(secret in raw for secret in self._values):
            raise CredentialRejected("registered authentication value rejected")
        def text_check(text):
            encoded = text.encode("utf-8", errors="surrogatepass")
            if any(secret in encoded for secret in self._values):
                raise CredentialRejected("registered authentication value rejected")

        def walk(item):
            if isinstance(item, str):
                text_check(item)
            elif isinstance(item, dict):
                pairs_check(item.items())
            elif isinstance(item, list):
                for child in item:
                    walk(child)

        def pairs_check(pairs):
            # Inspect before dict construction, including overwritten duplicate keys.
            for key, child in pairs:
                text_check(key)
                normalized = "".join(c for c in key.lower() if c.isalnum())
                if normalized in self.AUTH_KEYS and child is not None:
                    raise CredentialRejected("authentication field rejected")
                walk(child)
            return dict(pairs)

        def inspect_json(data):
            try:
                obj = json.loads(data, object_pairs_hook=pairs_check)
            except (ValueError, UnicodeError):
                return
            walk(obj)

        inspect_json(raw)
        # A capture may contain several SSE frames (including cleanup captures).
        try:
            lines = raw.decode("utf-8").splitlines()
        except UnicodeError:
            return
        data = []
        for line in lines + [""]:
            if line.startswith("data:"):
                data.append(line[5:].removeprefix(" "))
            elif not line and data:
                inspect_json("\n".join(data))
                data = []
