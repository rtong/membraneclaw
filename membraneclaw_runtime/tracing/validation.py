"""Integrity validation and immutable portable snapshots; no external execution."""
import re
from dataclasses import dataclass
from pathlib import Path

from .file_store import FileTraceStore
from .schema import Corruption, EpisodeRef, canonical_json, digest
from .store import genesis


@dataclass(frozen=True)
class ValidationReport:
    revision: int
    event_count: int
    payload_count: int
    status: str
    completeness: dict


def validate(store, episode):
    ep = store.read_episode(episode)
    refs = {r for c in ep.commits for r in c["references"]}
    return ValidationReport(ep.revision, len(ep.events), len(refs), ep.status, ep.completeness)


@dataclass(frozen=True)
class SealedSnapshot:
    episode: EpisodeRef
    revision: int
    files: tuple[tuple[str, bytes], ...]
    snapshot_hash: str

    def inventory(self):
        return {"schema_version": 1, "serialization": "canonical_json_v1",
                "files": [{"path": p, "sha256": digest(b), "size": len(b)} for p, b in self.files]}

    def verify(self):
        paths = [p for p, _ in self.files]
        if paths != sorted(set(paths)):
            raise Corruption("snapshot paths must be sorted and unique")
        for p in paths:
            if not re.fullmatch(r"genesis\.json|commits/[0-9]{6,}\.json|payloads/[a-f0-9]{64}\.bin", p):
                raise Corruption("unsafe snapshot path")
        if digest(canonical_json(self.inventory())) != self.snapshot_hash:
            raise Corruption("snapshot hash mismatch")


def seal(store, episode):
    with store.snapshot_guard(episode):
        ep = store.read_episode(episode)
        files = {"genesis.json": canonical_json(genesis(ep.ref, ep.metadata))}
        for c in ep.commits:
            files[f'commits/{c["revision"]:06d}.json'] = canonical_json(c)
            for ref in c["references"]:
                files[f'payloads/{ref.rsplit(":", 1)[1]}.bin'] = store.read_payload(ref)
        draft = SealedSnapshot(episode, ep.revision, tuple(sorted(files.items())), "")
        snapshot = SealedSnapshot(episode, ep.revision, draft.files, digest(canonical_json(draft.inventory())))
        snapshot.verify()
        return snapshot


def restore(snapshot, destination):
    """Restore into a new isolated directory, never overwrite live traces."""
    snapshot.verify()
    destination = Path(destination)
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    store = FileTraceStore(destination)
    directory = store._directory(snapshot.episode)
    for relative, data in snapshot.files:
        path = store._safe(snapshot.episode.key + "/" + relative)
        store._mkdir(path.parent)
        store._atomic(path, data, immutable=True)
    for relative, data in snapshot.files:
        if (directory / relative).read_bytes() != data:
            raise Corruption("snapshot copy mismatch")
    report = validate(store, snapshot.episode)
    if report.revision != snapshot.revision:
        raise Corruption("snapshot revision mismatch")
    store.rebuild_manifest(snapshot.episode)
    return store
