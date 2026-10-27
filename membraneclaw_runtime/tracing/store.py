"""Storage contract and backend-independent commit validation."""
from __future__ import annotations

import copy
from contextlib import AbstractContextManager, contextmanager
from dataclasses import asdict, dataclass
from typing import Callable, Iterator, Protocol

from .schema import (COMPLETENESS, SCHEMA_VERSION, SERIALIZATION, STATUSES, TERMINAL,
                     CommitReceipt, Conflict, Corruption, Episode, EpisodeRef, Event,
                     Payload, canonical_json, digest, identifier, payload_hash, receipt, version)


@dataclass(frozen=True)
class StoreCapabilities:
    durable: bool
    single_host: bool = True
    multi_host_writes: bool = False


@dataclass(frozen=True)
class EpisodePage:
    items: list[EpisodeRef]
    next_cursor: str | None


class TraceStore(Protocol):
    def capabilities(self) -> StoreCapabilities: ...
    def create_episode(self, metadata: dict, *, tenant: str, run_id: str,
                       episode_id: str | None = None) -> EpisodeRef: ...
    def execution_guard(self, episode: EpisodeRef) -> AbstractContextManager: ...
    def snapshot_guard(self, episode: EpisodeRef) -> AbstractContextManager: ...
    def commit(self, episode: EpisodeRef, commit_id: str, expected_revision: int,
               events: list[Event], payloads: list[Payload], checkpoint: Payload | None = None,
               *, status: str | None = None, completeness: dict | None = None) -> CommitReceipt: ...
    def read_episode(self, episode: EpisodeRef) -> Episode: ...
    def iter_events(self, episode: EpisodeRef, after_sequence: int = 0) -> Iterator[dict]: ...
    def read_payload(self, ref: str) -> bytes: ...
    def list_episodes(self, filters: dict | None = None, cursor: str | None = None,
                      limit: int = 100) -> EpisodePage: ...


def genesis(ref: EpisodeRef, metadata: dict) -> dict:
    value = {"schema_version": SCHEMA_VERSION, "serialization": SERIALIZATION,
             "episode": asdict(ref), "metadata": copy.deepcopy(metadata)}
    return {**value, "hash": digest(canonical_json(value))}


def verify_hash(value: dict):
    version(value)
    if value.get("serialization") != SERIALIZATION:
        raise Corruption("unsupported serialization")
    body = {k: v for k, v in value.items() if k != "hash"}
    if value.get("hash") != digest(canonical_json(body)):
        raise Corruption("document hash mismatch")


def referenced(commit: dict) -> set[str]:
    refs = set()
    for event in commit["events"]:
        for key in ("payload_ref", "state_before_ref", "state_after_ref"):
            if event.get(key) is not None:
                refs.add(event[key])
    if commit["checkpoint_ref"] is not None:
        refs.add(commit["checkpoint_ref"])
    return refs


def apply_commit(ep: Episode, c: dict, read_payload: Callable[[str], bytes]):
    verify_hash(c)
    if c["revision"] != ep.revision + 1 or c["previous_hash"] != (
            ep.commits[-1]["hash"] if ep.commits else genesis(ep.ref, ep.metadata)["hash"]):
        raise Corruption("commit chain gap or previous hash mismatch")
    identifier(c["commit_id"])
    if any(x["commit_id"] == c["commit_id"] for x in ep.commits):
        raise Corruption("duplicate commit ID")
    if c["status"] not in STATUSES:
        raise Corruption("invalid execution status")
    if ep.status in TERMINAL:
        raise Corruption("commit after terminal state")
    if set(c["completeness"]) != set(COMPLETENESS) or any(
            c["completeness"][k] not in allowed for k, allowed in COMPLETENESS.items()):
        raise Corruption("invalid completeness")
    seen = {e["event_id"] for e in ep.events}
    for n, event in enumerate(c["events"], len(ep.events) + 1):
        version(event)
        identifier(event["event_id"])
        if event["event_id"] in seen or event["sequence"] != n or event["episode_id"] != ep.ref.episode_id:
            raise Corruption("duplicate event ID or invalid event sequence/episode")
        if event.get("parent_event_id") is not None and event["parent_event_id"] not in seen:
            raise Corruption("parent event must precede child")
        seen.add(event["event_id"])
    expected_end = {"completed": "episode.completed", "failed": "episode.failed",
                    "cancelled": "episode.cancelled", "interrupted": "episode.interrupted"}
    end_events = [e for e in c["events"] if e["type"] in expected_end.values()]
    if c["status"] in TERMINAL:
        if len(end_events) != 1 or c["events"][-1]["type"] != expected_end[c["status"]]:
            raise Corruption("terminal state must match final event")
    elif end_events:
        raise Corruption("terminal event without terminal state")
    if sorted(referenced(c)) != c["references"]:
        raise Corruption("reference inventory mismatch")
    for ref in c["references"]:
        if digest(read_payload(ref)) != payload_hash(ref, ep.ref):
            raise Corruption("payload hash mismatch")
    ep.commits.append(c)
    ep.events.extend(c["events"])
    ep.revision = c["revision"]
    ep.status, ep.completeness = c["status"], copy.deepcopy(c["completeness"])
    ep.checkpoint_ref = c["checkpoint_ref"]


class StoreBase:
    """Backends implement locks and byte persistence; semantic checks stay identical."""

    def commit(self, episode, commit_id, expected_revision, events, payloads, checkpoint=None,
               *, status=None, completeness=None):
        identifier(commit_id)
        payloads = list(payloads) + ([checkpoint] if checkpoint is not None else [])
        # expected_revision is deliberately NOT part of idempotency identity.
        request = {"events": [asdict(e) for e in events],
                   "payloads": sorted({p.ref(episode) for p in payloads}),
                   "checkpoint": checkpoint.ref(episode) if checkpoint else None,
                   "status": status, "completeness": completeness}
        self.policy.check(request)
        for payload in payloads:
            self.policy.check(payload.data)
        request_hash = digest(canonical_json(request))
        with self._commit_guard(episode):
            ep = self.read_episode(episode)
            for c in ep.commits:
                if c["commit_id"] == commit_id:
                    if c["request_hash"] != request_hash:
                        raise Conflict("commit ID reused with different content")
                    self._confirm_commit(episode, c)
                    return receipt(c)
            if expected_revision != ep.revision:
                raise Conflict("stale revision")
            if {e.event_id for e in events} & {e["event_id"] for e in ep.events}:
                raise Conflict("event ID already committed")
            body = {"schema_version": SCHEMA_VERSION, "serialization": SERIALIZATION,
                    "commit_id": commit_id, "request_hash": request_hash,
                    "revision": ep.revision + 1,
                    "previous_hash": ep.commits[-1]["hash"] if ep.commits else genesis(ep.ref, ep.metadata)["hash"],
                    "events": [e.envelope(episode, len(ep.events) + i + 1) for i, e in enumerate(events)],
                    "checkpoint_ref": checkpoint.ref(episode) if checkpoint else ep.checkpoint_ref,
                    "status": status if status is not None else ep.status,
                    "completeness": completeness if completeness is not None else ep.completeness}
            body["references"] = sorted(referenced(body))
            c = {**body, "hash": digest(canonical_json(body))}
            incoming = {p.ref(episode): p.data for p in payloads}
            def read(ref):
                return incoming[ref] if ref in incoming else self.read_payload(ref)
            # Validate everything before persisting anything.
            apply_commit(ep, c, read)
            for p in payloads:
                self._write_payload(episode, p)
            self._write_commit(episode, c)
            self._update_manifest(ep)
            return receipt(c)

    @contextmanager
    def snapshot_guard(self, episode):
        with self.execution_guard(episode):
            with self._commit_guard(episode):
                yield

    def _confirm_commit(self, episode, commit):
        pass

    def iter_events(self, episode, after_sequence=0):
        return iter([e for e in self.read_episode(episode).events if e["sequence"] > after_sequence])

    def list_episodes(self, filters=None, cursor=None, limit=100):
        if not 1 <= limit <= 200:
            raise ValueError("limit must be 1..200")
        filters = filters or {}
        if set(filters) - {"tenant", "run_id", "status"}:
            raise ValueError("unsupported episode filter")
        matches = []
        for ref in sorted(self._episode_refs(), key=lambda r: r.key):
            if cursor and ref.key <= cursor:
                continue
            if any(getattr(ref, k) != v for k, v in filters.items() if k != "status"):
                continue
            if "status" in filters and self.read_episode(ref).status != filters["status"]:
                continue
            matches.append(ref)
            if len(matches) > limit:
                break
        return EpisodePage(matches[:limit], matches[limit - 1].key if len(matches) > limit else None)

    def recover(self, episode):
        # Nonblocking guard: an active network/tool wait is never interrupted.
        with self.execution_guard(episode):
            ep = self.read_episode(episode)
            if ep.status == "running":
                self.commit(episode, f"recovery_{ep.revision}", ep.revision,
                            [Event("episode.interrupted")], [], status="interrupted",
                            completeness={"capture": "partial", "replay": "unsupported",
                                          "external_outcome": "unknown"})
            return self.read_episode(episode)
