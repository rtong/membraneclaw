"""Single-host POSIX file store. Commit files are authoritative, not manifest.json."""
from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from .policy import CapturePolicy
from .schema import (Busy, Conflict, Corruption, Episode, EpisodeRef, Payload,
                     canonical_json, digest, payload_hash)
from .store import StoreBase, StoreCapabilities, apply_commit, genesis, verify_hash


class FileTraceStore(StoreBase):
    def __init__(self, root, *, multi_host=False, fault_hook=None, policy=None):
        if multi_host:
            raise ValueError("file traces require single-host local storage")
        self.policy = policy or CapturePolicy()
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._fault_hook = fault_hook or (lambda stage: None)

    def capabilities(self):
        return StoreCapabilities(durable=True)

    def _safe(self, relative):
        path = self.root
        for part in Path(relative).parts:
            if part in ("..", ".") or Path(part).is_absolute():
                raise Corruption("unsafe trace path")
            path = path / part
            if path.is_symlink():
                raise Corruption("symlinks are forbidden in trace data")
        return path

    def _directory(self, episode):
        # EpisodeRef validates each path component even for reads/imported refs.
        return self._safe(episode.key)

    @staticmethod
    def _fsync_dir(path):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _mkdir(self, path):
        if not path.exists():
            self._mkdir(path.parent)
            path.mkdir(mode=0o700, exist_ok=True)
            self._fsync_dir(path.parent)

    def _atomic(self, path, data, *, immutable=False, stage=None):
        if immutable and path.exists():
            if path.read_bytes() != data:
                raise Corruption("immutable object content differs")
            return
        tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with tmp.open("xb") as f:
                os.chmod(tmp, 0o600)
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            if stage:
                self._fault_hook(stage + "_fsync")
                self._fault_hook(stage + "_before_rename")
            os.replace(tmp, path)
            self._fsync_dir(path.parent)
            if stage:
                self._fault_hook(stage + "_committed")
        finally:
            tmp.unlink(missing_ok=True)

    @contextmanager
    def _lock(self, path):
        # Never blocks an async event loop; caller may retry Busy in a worker.
        with path.open("a+b") as f:
            os.chmod(path, 0o600)
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as e:
                raise Busy("episode/store is busy") from e
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def _commit_guard(self, episode):
        return self._lock(self._safe(episode.key + "/.commit.lock"))

    def execution_guard(self, episode):
        return self._lock(self._safe(episode.key + "/.execution.lock"))

    def create_episode(self, metadata, *, tenant, run_id, episode_id=None):
        ref = EpisodeRef(tenant, run_id, episode_id or uuid4().hex)
        self.policy.check(metadata)
        g = genesis(ref, metadata)
        with self._lock(self._safe(".create.lock")):
            directory = self._directory(ref)
            if directory.exists():
                raise Conflict("episode already exists (or incomplete creation requires inspection)")
            for suffix in ("commits", "payloads"):
                self._mkdir(directory / suffix)
            self._atomic(directory / "genesis.json", canonical_json(g), immutable=True)
            self._update_manifest(Episode(ref, g["metadata"]))
        return ref

    def _read_json(self, path):
        try:
            return json.loads(path.read_bytes())
        except (OSError, ValueError) as e:
            raise Corruption(f"cannot read trace document: {path.name}") from e

    def read_episode(self, episode):
        directory = self._directory(episode)
        g = self._read_json(self._safe(episode.key + "/genesis.json"))
        verify_hash(g)
        if g["episode"] != asdict(episode):
            raise Corruption("genesis episode mismatch")
        ep = Episode(episode, g["metadata"])
        files = sorted(self._safe(episode.key + "/commits").iterdir())
        for path in files:
            if path.name.startswith(".") and path.name.endswith(".tmp"):
                continue
            expected = f"{ep.revision + 1:06d}.json"
            if path.name != expected or path.is_symlink():
                raise Corruption("unexpected commit entry or sequence gap")
            c = self._read_json(path)
            try:
                apply_commit(ep, c, self.read_payload)
            except (KeyError, TypeError, ValueError) as e:
                raise Corruption("invalid commit structure") from e
        return ep

    def read_payload(self, ref):
        parts = ref.split("/")
        if len(parts) != 4:
            raise Corruption("invalid payload reference")
        try:
            episode = EpisodeRef(*parts[:3])
        except ValueError as e:
            raise Corruption("invalid payload namespace") from e
        h = payload_hash(ref, episode)
        path = self._safe(episode.key + f"/payloads/{h}.bin")
        try:
            data = path.read_bytes()
        except OSError as e:
            raise Corruption("missing/unreadable payload") from e
        if digest(data) != h:
            raise Corruption("payload hash mismatch")
        return data

    def _write_payload(self, episode, payload):
        h = digest(payload.data)
        path = self._safe(episode.key + f"/payloads/{h}.bin")
        self._atomic(path, payload.data, immutable=True, stage="payload")

    def _confirm_commit(self, episode, commit):
        # A previous response may have been lost after rename but before fsync.
        paths = [self._safe(episode.key + f'/commits/{commit["revision"]:06d}.json')]
        paths.extend(self._safe(episode.key + f'/payloads/{ref.rsplit(":", 1)[1]}.bin')
                     for ref in commit["references"])
        for path in paths:
            with path.open("rb") as f:
                os.fsync(f.fileno())
        for directory in {p.parent for p in paths}:
            self._fsync_dir(directory)

    def _write_commit(self, episode, c):
        if c["revision"] > 999999:
            raise ValueError("file backend revision capacity exceeded; start a new episode")
        path = self._safe(episode.key + f'/commits/{c["revision"]:06d}.json')
        self._atomic(path, canonical_json(c), immutable=True, stage="commit")

    def _update_manifest(self, ep):
        self._fault_hook("manifest_before_update")
        data = {"schema_version": 1, "episode": asdict(ep.ref), "revision": ep.revision,
                "status": ep.status, "completeness": ep.completeness,
                "checkpoint_ref": ep.checkpoint_ref,
                "head_hash": ep.commits[-1]["hash"] if ep.commits else genesis(ep.ref, ep.metadata)["hash"]}
        self._atomic(self._safe(ep.ref.key + "/manifest.json"), canonical_json(data))

    def rebuild_manifest(self, episode):
        with self._commit_guard(episode):
            ep = self.read_episode(episode)
            self._update_manifest(ep)
            return ep

    def orphan_payloads(self, episode):
        ep = self.read_episode(episode)
        used = {payload_hash(ref, episode) for c in ep.commits for ref in c["references"]}
        return sorted(p.name for p in self._safe(episode.key + "/payloads").iterdir()
                      if p.suffix == ".bin" and p.stem not in used)

    def _episode_refs(self):
        for path in self.root.glob("*/*/*/genesis.json"):
            relative = path.relative_to(self.root)
            self._safe(relative)
            yield EpisodeRef(*relative.parts[:3])
