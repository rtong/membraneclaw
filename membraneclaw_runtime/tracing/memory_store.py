"""Non-durable backend for shared storage-contract tests."""
import copy
import threading
from contextlib import contextmanager
from uuid import uuid4

from .policy import CapturePolicy
from .schema import Busy, Conflict, Corruption, Episode, EpisodeRef, digest
from .store import StoreBase, StoreCapabilities, apply_commit, genesis, verify_hash


class MemoryTraceStore(StoreBase):
    def __init__(self, *, policy=None):
        self.policy = policy or CapturePolicy()
        self._episodes = {}
        self._payloads = {}
        self._locks = {}
        self._mutex = threading.Lock()

    def capabilities(self):
        return StoreCapabilities(durable=False)

    def create_episode(self, metadata, *, tenant, run_id, episode_id=None):
        ref = EpisodeRef(tenant, run_id, episode_id or uuid4().hex)
        self.policy.check(metadata)
        g = genesis(ref, metadata)
        with self._mutex:
            if ref.key in self._episodes:
                raise Conflict("episode already exists")
            self._episodes[ref.key] = (g, [])
        return ref

    @contextmanager
    def _guard(self, episode, kind):
        with self._mutex:
            if episode.key not in self._episodes:
                raise Corruption("episode does not exist")
            lock = self._locks.setdefault((episode.key, kind), threading.Lock())
        if not lock.acquire(blocking=False):
            raise Busy("episode is busy")
        try:
            yield
        finally:
            lock.release()

    def _commit_guard(self, episode):
        return self._guard(episode, "commit")

    def execution_guard(self, episode):
        return self._guard(episode, "execution")

    def read_episode(self, episode):
        g, commits = copy.deepcopy(self._episodes[episode.key])
        verify_hash(g)
        ep = Episode(episode, g["metadata"])
        for c in commits:
            apply_commit(ep, c, self.read_payload)
        return ep

    def read_payload(self, ref):
        try:
            return self._payloads[ref]
        except KeyError as e:
            raise Corruption("missing payload") from e

    def _write_payload(self, episode, payload):
        ref = payload.ref(episode)
        if ref in self._payloads and self._payloads[ref] != payload.data:
            raise Corruption("payload collision")
        self._payloads[ref] = payload.data

    def _write_commit(self, episode, c):
        self._episodes[episode.key][1].append(copy.deepcopy(c))

    def _update_manifest(self, ep):
        pass

    def _episode_refs(self):
        return [EpisodeRef(**g["episode"]) for g, _ in self._episodes.values()]
