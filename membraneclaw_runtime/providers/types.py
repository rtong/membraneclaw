"""Provider-neutral values; native state is authoritative, normalized fields are views."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


class ProviderError(Exception):
    pass


class ConfigurationError(ProviderError):
    pass


class ProtocolError(ProviderError):
    pass


class BudgetExceeded(ProviderError):
    pass


class ProviderHTTPError(ProviderError):
    def __init__(self, status):
        self.status = status
        super().__init__(f"provider HTTP status {status}")


class TransportError(ProviderError):
    """Fixed messages only; exception URLs/credentials never enter traces."""
    def __init__(self, code="request_transport_failed"):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class ModelRoute:
    provider: str
    model: str
    api_family: str


@dataclass(frozen=True)
class Capabilities:
    tool_calling: bool = True
    parallel_calls: bool = True
    streaming: bool = False  # Remote capability alone does not implement PR 6.
    structured_output: bool = False
    vision: bool = False
    reasoning_options: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContentBlock:
    type: str
    text: str

    @classmethod
    def text_block(cls, text):
        return cls("text", text)


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict


@dataclass(frozen=True)
class ModelRequest:
    route: ModelRoute
    content: tuple[ContentBlock, ...] = ()  # New input ONLY, not full history.
    system_instruction: str = ""
    tools: tuple[ToolDefinition, ...] = ()
    temperature: float | None = None
    max_output_tokens: int | None = None
    provider_options: dict = field(default_factory=dict)
    input_token_upper_bound: int | None = None

    def to_document(self):
        value = asdict(self)
        value["content"] = [asdict(b) for b in self.content]
        value["tools"] = [asdict(t) for t in self.tools]
        return value


@dataclass(frozen=True)
class ToolCall:
    id: str
    provider_call_id: str | None
    name: str
    arguments_raw: dict
    arguments: dict
    native_position: int


@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    result: dict
    is_error: bool = False


@dataclass(frozen=True)
class ProviderState:
    route: ModelRoute
    adapter_version: str
    native_history: tuple[dict, ...] = ()
    pending_calls: tuple[ToolCall, ...] = ()
    config_hash: str | None = None
    schema_version: int = 1

    def to_document(self):
        document = asdict(self)
        document["native_history"] = list(document["native_history"])
        document["pending_calls"] = list(document["pending_calls"])
        return document


@dataclass(frozen=True)
class NativeRequest:
    route: ModelRoute
    method: str
    path: str
    body: bytes  # Final bytes actually sent, not a pre-serialization object.
    config_hash: str


@dataclass(frozen=True)
class NativeResponse:
    status: int
    headers: dict[str, str]  # Allowlisted metadata only.
    body: bytes
    complete: bool = True
    error_code: str | None = None
    encoded_body: bytes | None = None


@dataclass(frozen=True)
class Usage:
    input_tokens: int | None = None  # Includes cached input.
    output_tokens: int | None = None  # Provider candidatesTokenCount.
    total_tokens: int | None = None  # Provider total, never recomputed.
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    raw: dict | None = None
    unavailable_reason: str | None = "not_provided"


@dataclass(frozen=True)
class ModelOutput:
    text_blocks: tuple[ContentBlock, ...]
    thought_summaries: tuple[str, ...]
    tool_calls: tuple[ToolCall, ...]
    finish_status: Literal["completed", "tool_calls", "blocked", "incomplete", "unknown"]
    stop_reason: str | None
    usage: Usage
    native_response: dict
    native_request: NativeRequest
    native_response_ref: str
    selected_candidate: int | None
    model_call_id: str

    @property
    def text(self):
        return "".join(block.text for block in self.text_blocks)
