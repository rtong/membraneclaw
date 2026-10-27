"""Single-attempt HTTP transport. Captures bytes before an API parser can see them."""
import asyncio
import gzip
import http.client
import os
import re
import socket
import ssl
import time
from dataclasses import replace

from .types import ConfigurationError, NativeResponse, TransportError

SAFE_RESPONSE_HEADERS = {"content-type", "content-encoding", "retry-after", "date", "x-request-id"}


class GeminiTransport:
    host = "generativelanguage.googleapis.com"

    def __init__(self, api_key):
        if not isinstance(api_key, str) or not api_key or any(c.isspace() for c in api_key):
            raise ConfigurationError("Gemini API credential is missing or invalid")
        self._api_key = api_key
        self.authentication_values = (api_key.encode(),)

    @classmethod
    def from_env(cls, credential_env="GEMINI_API_KEY"):
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", credential_env):
            raise ConfigurationError("invalid credential environment name")
        return cls(os.environ.get(credential_env))

    def _connection(self, timeout):
        return http.client.HTTPSConnection(self.host, timeout=timeout, context=ssl.create_default_context())

    def _exchange(self, request, timeout, holder):
        connection = self._connection(timeout)
        holder["connection"] = connection
        start = time.monotonic()
        chunks = []
        try:
            connection.request("POST", request.path, body=request.body, headers=self._headers())
            holder["socket"] = connection.sock
            response = connection.getresponse()
            headers = {key.lower(): value for key, value in response.getheaders()
                       if key.lower() in SAFE_RESPONSE_HEADERS}
            complete, code = True, None
            try:
                while True:
                    remaining = timeout - (time.monotonic() - start)
                    if remaining <= 0:
                        raise TimeoutError
                    if holder["socket"] is not None:
                        holder["socket"].settimeout(remaining)
                    chunk = response.read1(65536)
                    if not chunk:
                        if response.length not in (None, 0):
                            complete, code = False, "truncated_body"
                        break
                    chunks.append(chunk)
                    if response.length == 0:
                        break
            except (OSError, http.client.HTTPException):
                complete, code = False, "response_read_failed"
            body, encoded = b"".join(chunks), None
            encoding = headers.get("content-encoding", "identity").lower()
            if encoding == "gzip":
                encoded = body
                try:
                    body = gzip.decompress(body)
                except (OSError, EOFError):
                    complete, code = False, "gzip_decode_failed"
            elif encoding != "identity":
                complete, code = False, "unsupported_content_encoding"
            return NativeResponse(response.status, headers, body, complete, code, encoded)
        except (OSError, http.client.HTTPException):
            raise TransportError("request_transport_failed") from None
        finally:
            connection.close()

    @staticmethod
    def _abort(holder):
        sock = holder.get("socket")
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        connection = holder.get("connection")
        if connection is not None:
            connection.close()

    def _headers(self):
        return {"Content-Type": "application/json", "Accept-Encoding": "identity", "x-goog-api-key": self._api_key}

    def _validate_request(self, request, timeout):
        if (request.route.provider != "gemini" or request.route.api_family != "generate_content_v1beta"
                or request.method != "POST" or not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", request.route.model)
                or request.path != f"/v1beta/models/{request.route.model}:generateContent"
                or not isinstance(request.body, bytes) or timeout <= 0):
            raise ConfigurationError("invalid Gemini native request")
    async def exchange(self, request, *, before_send, after_receive, timeout):
        self._validate_request(request, timeout)
        start = time.monotonic()
        await before_send(request)  # Failure here sends ZERO HTTP requests.
        timeout -= time.monotonic() - start
        if timeout <= 0:
            raise TransportError("deadline_before_send")
        holder = {}
        task = asyncio.create_task(asyncio.to_thread(self._exchange, request, timeout, holder))
        try:
            response = await asyncio.wait_for(asyncio.shield(task), timeout)
        except (asyncio.CancelledError, TimeoutError) as error:
            self._abort(holder)
            try:
                response = await task  # Settle the worker before releasing execution_guard.
            except TransportError:
                pass
            else:
                await after_receive(replace(response, complete=False, error_code="attempt_aborted"))
            if isinstance(error, asyncio.CancelledError):
                raise
            raise TransportError("request_timeout") from None
        await after_receive(response)  # No JSON parsing has occurred.
        return response

    async def exchange_stream(self, request, **kwargs):
        from ..streaming import exchange_stream
        return await exchange_stream(self, request, **kwargs)
