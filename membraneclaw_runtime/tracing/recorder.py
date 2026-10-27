"""Recorder facade. Callers hold execution_guard across model/tool waits."""
import copy
from uuid import uuid4

from .schema import Conflict, Event, Payload, TERMINAL
from .store import TraceStore


class TraceRecorder:
    def __init__(self, store: TraceStore, episode):
        self.store, self.episode = store, episode
        self.revision = store.read_episode(episode).revision
        self.pending_commit = None

    @classmethod
    def begin_episode(cls, store: TraceStore, metadata, *, tenant="local", run_id=None):
        episode = store.create_episode(metadata, tenant=tenant, run_id=run_id or uuid4().hex)
        recorder = cls(store, episode)
        recorder.append_event(Event("episode.started"))
        return recorder

    def append_event(self, event, *, payload=None, checkpoint=None, commit_id=None,
                     status=None, completeness=None):
        if self.pending_commit is not None:
            raise Conflict("unresolved storage failure: call retry_pending before new work")
        # Retain stable IDs and exact data even when the storage response is lost.
        self.pending_commit = copy.deepcopy({
            "commit_id": commit_id or uuid4().hex, "expected_revision": self.revision,
            "events": [event], "payloads": [payload] if payload is not None else [],
            "checkpoint": checkpoint, "status": status, "completeness": completeness})
        return self.retry_pending()

    def retry_pending(self):
        """Retry storage only. Never reissue model requests or tool actions."""
        if self.pending_commit is None:
            raise ValueError("no pending storage commit")
        receipt = self.store.commit(self.episode, **self.pending_commit)
        self.revision = receipt.revision
        self.pending_commit = None
        return receipt

    def checkpoint(self, state, *, status=None, commit_id=None, event=None):
        payload = Payload.json(state)
        event = event or Event("state.checkpoint", state_after_ref=payload.ref(self.episode))
        return self.append_event(event, checkpoint=payload, status=status, commit_id=commit_id)

    def finish_episode(self, status="completed", *, capture="unknown", replay="unsupported",
                       external_outcome="unknown", commit_id=None, event=None):
        if status not in TERMINAL:
            raise ValueError("finish requires terminal execution status")
        return self.append_event(event or Event(f"episode.{status}"), status=status,
                                 completeness={"capture": capture, "replay": replay,
                                               "external_outcome": external_outcome},
                                 commit_id=commit_id)
