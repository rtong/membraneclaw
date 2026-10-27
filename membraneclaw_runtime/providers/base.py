"""Adapters own protocol conversion; the gateway owns attempts and recording."""
from typing import Protocol

from .types import (Capabilities, ModelOutput, ModelRequest, NativeRequest,
                    NativeResponse, ProviderState, ToolResult)


class ProviderAdapter(Protocol):
    provider: str
    api_family: str
    adapter_version: str
    def capabilities(self, model: str) -> Capabilities: ...
    def initial_state(self, model: str) -> ProviderState: ...
    def prepare(self, request: ModelRequest, state: ProviderState) -> NativeRequest: ...
    def normalize(self, response: NativeResponse, request: NativeRequest, *,
                  response_ref: str, model_call_id: str) -> ModelOutput: ...
    def advance(self, state: ProviderState, output: ModelOutput,
                tool_results: tuple[ToolResult, ...] = ()) -> ProviderState: ...
    def add_tool_results(self, state: ProviderState,
                         tool_results: tuple[ToolResult, ...]) -> ProviderState: ...


class RawTransport(Protocol):
    authentication_values: tuple[bytes, ...]
    async def exchange(self, request: NativeRequest, *, before_send, after_receive,
                       timeout: float) -> NativeResponse:
        """Await before_send(FINAL request) before I/O and after_receive(raw
        response) before parsing/returning. No retries, redirects or tools.
        Callback failures must propagate, never become network errors.
        """
        ...
