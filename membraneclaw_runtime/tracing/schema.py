"""canonical_json_v1 and the v1 trace envelope (stdlib only)."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

SCHEMA_VERSION = 1
SERIALIZATION = "canonical_json_v1"
STATUSES = {"running", "awaiting_client_tool", "completed", "failed", "cancelled", "interrupted"}
TERMINAL = STATUSES - {"running", "awaiting_client_tool"}
COMPLETENESS = {"capture": {"complete", "partial", "unknown", "redacted"},
                "replay": {"ready", "unsupported", "missing_payload"},
                "external_outcome": {"known", "unknown"}}


class TraceError(Exception):
    """Base error; callers must stop external actions on storage failure."""


class Conflict(TraceError):
    pass


class Busy(TraceError):
    pass


class Corruption(TraceError):
    pass


class UnsupportedVersion(Corruption):
    pass


def canonical_json(value: Any) -> bytes:
    # Restrict to JSON types: do not silently stringify keys or serialize tuples.
    def check(v):
        if isinstance(v, dict):
            if not all(isinstance(k, str) for k in v):
                raise ValueError("JSON object keys must be strings")
            for x in v.values():
                check(x)
        elif isinstance(v, list):
            for x in v:
                check(x)
        elif v is not None and type(v) not in (str, bool, int, float):
            raise ValueError(f"not a JSON value: {type(v).__name__}")
    check(value)
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def identifier(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value):
        raise ValueError("invalid internal identifier")
    return value


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class EpisodeRef:
    tenant: str
    run_id: str
    episode_id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self):
        for value in asdict(self).values():
            identifier(value)

    @property
    def key(self) -> str:
        return f"{self.tenant}/{self.run_id}/{self.episode_id}"


@dataclass(frozen=True)
class Payload:
    data: bytes

    def __post_init__(self):
        if not isinstance(self.data, bytes):
            raise ValueError("payload requires bytes; use Payload.json for JSON")

    @classmethod
    def json(cls, value: Any) -> Payload:
        return cls(canonical_json(value))

    def ref(self, episode: EpisodeRef) -> str:
        return f"{episode.key}/sha256:{digest(self.data)}"


def payload_hash(ref: str, episode: EpisodeRef) -> str:
    prefix = episode.key + "/sha256:"
    if not isinstance(ref, str) or not ref.startswith(prefix):
        raise Corruption("cross-episode or invalid payload reference")
    h = ref[len(prefix):]
    if not re.fullmatch(r"[a-f0-9]{64}", h):
        raise Corruption("invalid payload hash")
    return h


@dataclass(frozen=True)
class Event:
    type: str
    event_id: str = field(default_factory=lambda: uuid4().hex)
    wall_time: str = field(default_factory=now)
    parent_event_id: str | None = None
    request_id: str | None = None
    attempt_id: str | None = None
    model_call_id: str | None = None
    tool_call_id: str | None = None
    payload_ref: str | None = None
    state_before_ref: str | None = None
    state_after_ref: str | None = None
    data: dict = field(default_factory=dict)

    def envelope(self, episode: EpisodeRef, sequence: int) -> dict:
        identifier(self.event_id)
        if not self.type or not isinstance(self.type, str):
            raise ValueError("event type required")
        return {"schema_version": SCHEMA_VERSION, "episode_id": episode.episode_id,
                "sequence": sequence, **asdict(self)}


@dataclass(frozen=True)
class CommitReceipt:
    revision: int
    first_sequence: int
    last_sequence: int
    commit_hash: str


@dataclass
class Episode:
    ref: EpisodeRef
    metadata: dict
    revision: int = 0
    events: list[dict] = field(default_factory=list)
    commits: list[dict] = field(default_factory=list)
    status: str = "running"
    completeness: dict = field(default_factory=lambda: {
        "capture": "unknown", "replay": "unsupported", "external_outcome": "unknown"})
    checkpoint_ref: str | None = None


def version(document: dict):
    if document.get("schema_version") != SCHEMA_VERSION:
        raise UnsupportedVersion("unsupported trace schema version")


def receipt(commit: dict) -> CommitReceipt:
    events = commit["events"]
    return CommitReceipt(commit["revision"], events[0]["sequence"] if events else 0,
                         events[-1]["sequence"] if events else 0, commit["hash"])
