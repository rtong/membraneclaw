"""Shared backend contracts, using synthetic data only."""
import concurrent.futures
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from membraneclaw_runtime.tracing import FileTraceStore, MemoryTraceStore, Event, Payload, TraceRecorder
from membraneclaw_runtime.tracing.schema import (Busy, Conflict, Corruption, EpisodeRef,
                                               UnsupportedVersion, canonical_json, digest)
from membraneclaw_runtime.tracing.validation import restore, seal, validate


class Contracts:
    def make(self):
        raise NotImplementedError

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = self.make()
        self.ep = self.store.create_episode({"fixture": "synthetic"}, tenant="t", run_id="r", episode_id="e")

    def test_canonical_golden(self):
        self.assertEqual(canonical_json({"z": [1, True, None], "a": "水"}),
                         b'{"a":"\xe6\xb0\xb4","z":[1,true,null]}')
        for value in [float("nan"), float("inf"), {1: "bad"}, b"raw"]:
            with self.assertRaises(ValueError):
                canonical_json(value)

    def test_round_trip_large_bytes_and_checkpoint(self):
        r = TraceRecorder(self.store, self.ep)
        raw = Payload(b'\x00\xff' + "水".encode() * 400000)
        with self.store.execution_guard(self.ep):
            for kind in ["attempt.started", "attempt.response", "tool.started", "tool.completed"]:
                r.append_event(Event(kind, payload_ref=raw.ref(self.ep)), payload=raw)
            r.checkpoint({"native_history": [{"opaque_signature": "synthetic"}], "next": "finish"})
            r.finish_episode(capture="complete", replay="ready", external_outcome="known")
        ep = self.store.read_episode(self.ep)
        self.assertEqual(ep.status, "completed")
        self.assertEqual(self.store.read_payload(raw.ref(self.ep)), raw.data)
        self.assertEqual([e["sequence"] for e in ep.events], list(range(1, 7)))
        self.assertEqual(len(list(self.store.iter_events(self.ep, 4))), 2)
        self.assertEqual(validate(self.store, self.ep).payload_count, 2)
        with self.assertRaises(Corruption):
            r.append_event(Event("too.late"))

    def test_idempotency_before_revision_check(self):
        e = Event("attempt.started", event_id="stable")
        first = self.store.commit(self.ep, "stable_commit", 0, [e], [])
        self.assertEqual(first, self.store.commit(self.ep, "stable_commit", 0, [e], []))
        with self.assertRaises(Conflict):
            self.store.commit(self.ep, "stable_commit", 0, [Event("different")], [])
        with self.assertRaises(Conflict):
            self.store.commit(self.ep, "new_commit", 0, [Event("new")], [])
        with self.assertRaises(Conflict):
            self.store.commit(self.ep, "duplicate_event", 1, [e], [])
        self.assertEqual(self.store.read_episode(self.ep).revision, 1)

    def test_duplicate_within_commit_and_missing_refs(self):
        e = Event("same")
        with self.assertRaises(Corruption):
            self.store.commit(self.ep, "c", 0, [e, e], [])
        p = Payload(b"not persisted")
        with self.assertRaises(Corruption):
            self.store.commit(self.ep, "missing", 0, [Event("missing", payload_ref=p.ref(self.ep))], [])
        self.assertEqual(self.store.read_episode(self.ep).revision, 0)

    def test_terminal_and_completeness_are_independent(self):
        r = TraceRecorder(self.store, self.ep)
        with self.assertRaises(Corruption):
            r.append_event(Event("episode.completed"))
        r = TraceRecorder(self.store, self.ep)
        r.finish_episode("failed", capture="complete", replay="ready", external_outcome="known")
        self.assertEqual(validate(self.store, self.ep).completeness["capture"], "complete")

    def test_cross_episode_reference_and_path_validation(self):
        other = self.store.create_episode({}, tenant="t", run_id="r")
        p = Payload(b"secret")
        self.store.commit(other, "c", 0, [Event("x", payload_ref=p.ref(other))], [p])
        with self.assertRaises(Corruption):
            self.store.commit(self.ep, "cross", 0, [Event("x", payload_ref=p.ref(other))], [])
        for invalid in ["../escape", "/absolute", "a/b", ".", "", "a.b"]:
            with self.assertRaises(ValueError):
                EpisodeRef(invalid, "r", "e")

    def test_guard_recovery_during_slow_request(self):
        r = TraceRecorder(self.store, self.ep)
        r.append_event(Event("attempt.started"))
        with self.store.execution_guard(self.ep):
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(self.store.recover, self.ep)
                with self.assertRaises(Busy):
                    future.result(timeout=5)
            self.assertEqual(self.store.read_episode(self.ep).status, "running")
        recovered = self.store.recover(self.ep)
        self.assertEqual(recovered.status, "interrupted")
        self.assertEqual(recovered.completeness["external_outcome"], "unknown")
        self.assertEqual(self.store.recover(self.ep).revision, recovered.revision)

    def test_awaiting_client_not_interrupted(self):
        r = TraceRecorder(self.store, self.ep)
        r.checkpoint({"pending": ["call1"]}, status="awaiting_client_tool")
        self.assertEqual(self.store.recover(self.ep).status, "awaiting_client_tool")
        self.assertEqual(validate(self.store, self.ep).revision, 1)

    def test_snapshot_fixed_revision_and_restore(self):
        r = TraceRecorder(self.store, self.ep)
        r.checkpoint({"pending": ["call1"]}, status="awaiting_client_tool")
        first = seal(self.store, self.ep)
        self.assertEqual(first.snapshot_hash, seal(self.store, self.ep).snapshot_hash)
        r.append_event(Event("client_tool.result_received"), status="running")
        restored = restore(first, Path(self.tmp.name) / "restored")
        self.assertEqual(restored.read_episode(self.ep).revision, 1)
        self.assertEqual(seal(restored, self.ep).snapshot_hash, first.snapshot_hash)
        with self.assertRaises(FileExistsError):
            restore(first, Path(self.tmp.name) / "restored")
        with self.assertRaises(Corruption):
            restore(replace(first, files=first.files + (("../escape", b"bad"),)), Path(self.tmp.name) / "bad")

    def test_episode_isolation_and_pagination(self):
        other = self.store.create_episode({}, tenant="other", run_id="r", episode_id="e")
        def write(ep):
            for i in range(8):
                self.store.commit(ep, f"c{i}", i, [Event("x", data={"owner": ep.tenant})], [])
        with concurrent.futures.ThreadPoolExecutor() as pool:
            list(pool.map(write, [self.ep, other]))
        for ep in [self.ep, other]:
            self.assertEqual({e["data"]["owner"] for e in self.store.iter_events(ep)}, {ep.tenant})
        page = self.store.list_episodes(limit=1)
        next_page = self.store.list_episodes(cursor=page.next_cursor, limit=1)
        self.assertEqual(len(page.items) + len(next_page.items), 2)
        self.assertEqual(self.store.list_episodes({"tenant": "t"}).items, [self.ep])

    def test_write_failure_stops_external_action(self):
        r = TraceRecorder(self.store, self.ep)
        actions = []
        with patch.object(self.store, "_write_commit", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                r.append_event(Event("attempt.started"))
                actions.append("model invoked")
        self.assertEqual(actions, [])
        self.assertEqual(self.store.read_episode(self.ep).revision, 0)


class MemoryContracts(Contracts, unittest.TestCase):
    def make(self):
        return MemoryTraceStore()


class FileContracts(Contracts, unittest.TestCase):
    def make(self):
        return FileTraceStore(Path(self.tmp.name) / "traces")

    def test_manifest_is_not_authoritative(self):
        r = TraceRecorder(self.store, self.ep)
        with patch.object(self.store, "_update_manifest", side_effect=OSError("crash after commit")):
            e = Event("attempt.started")
            with self.assertRaises(OSError):
                r.append_event(e, commit_id="stable")
        self.assertEqual(self.store.read_episode(self.ep).revision, 1)
        self.assertEqual(r.retry_pending().revision, 1)
        self.store.rebuild_manifest(self.ep)
        path = self.store.root / self.ep.key / "manifest.json"
        self.assertEqual(json.loads(path.read_bytes())["revision"], 1)

    def test_unknown_schema_missing_payload_and_corruption(self):
        r = TraceRecorder(self.store, self.ep)
        p = Payload(b"payload")
        r.append_event(Event("x", payload_ref=p.ref(self.ep)), payload=p)
        path = self.store.root / self.ep.key / "commits/000001.json"
        original = path.read_bytes()
        obj = json.loads(original)
        obj["schema_version"] = 999
        path.write_bytes(canonical_json(obj))
        with self.assertRaises(UnsupportedVersion):
            validate(self.store, self.ep)
        path.write_bytes(original)
        payload = self.store.root / self.ep.key / f"payloads/{digest(p.data)}.bin"
        payload.write_bytes(b"tampered")
        with self.assertRaises(Corruption):
            validate(self.store, self.ep)
        payload.unlink()
        with self.assertRaises(Corruption):
            validate(self.store, self.ep)

    def test_orphan_and_commit_gap(self):
        p = Payload(b"orphan")
        with patch.object(self.store, "_write_commit", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.store.commit(self.ep, "c", 0, [Event("x", payload_ref=p.ref(self.ep))], [p])
        self.assertEqual(self.store.orphan_payloads(self.ep), [digest(p.data) + ".bin"])
        self.store.commit(self.ep, "c", 0, [Event("x")], [])
        directory = self.store.root / self.ep.key / "commits"
        (directory / "000001.json").rename(directory / "000002.json")
        with self.assertRaises(Corruption):
            validate(self.store, self.ep)

    def test_symlink_and_multi_host_rejected(self):
        with self.assertRaises(ValueError):
            FileTraceStore(self.store.root, multi_host=True)
        directory = self.store.root / self.ep.key
        (directory / "payloads").rmdir()
        (directory / "payloads").symlink_to(Path(self.tmp.name), target_is_directory=True)
        with self.assertRaises(Corruption):
            self.store._write_payload(self.ep, Payload(b"x"))

class PolicyAndRetry(unittest.TestCase):
    def test_credentials_rejected_before_disk_write(self):
        from membraneclaw_runtime.tracing.policy import CapturePolicy, CredentialRejected
        for backend in (MemoryTraceStore, FileTraceStore):
            with self.subTest(backend=backend.__name__), tempfile.TemporaryDirectory() as root:
                kwargs = {"policy": CapturePolicy(["synthetic-key-value"])}
                store = backend(root, **kwargs) if backend is FileTraceStore else backend(**kwargs)
                with self.assertRaises(CredentialRejected):
                    store.create_episode({"Authorization": "Bearer synthetic"}, tenant="t", run_id="r")
                ep = store.create_episode({"credential_env": "GEMINI_API_KEY"}, tenant="t", run_id="r")
                for payload in [Payload(b"raw synthetic-key-value response"),
                                Payload.json({"api_key": "other"})]:
                    with self.assertRaises(CredentialRejected):
                        store.commit(ep, "c", 0, [Event("x", payload_ref=payload.ref(ep))], [payload])
                self.assertEqual(store.read_episode(ep).revision, 0)
                if backend is FileTraceStore:
                    self.assertEqual(list((store.root / ep.key / "payloads").iterdir()), [])
                # Native continuation signatures are business data, not auth.
                p = Payload.json({"thoughtSignature": "synthetic-opaque"})
                store.commit(ep, "ok", 0, [Event("x", payload_ref=p.ref(ep))], [p])

    def test_recorder_retains_generated_id_after_uncertain_commit(self):
        with tempfile.TemporaryDirectory() as root:
            store = FileTraceStore(root)
            r = TraceRecorder.begin_episode(store, {})
            with patch.object(store, "_update_manifest", side_effect=OSError("lost response")):
                with self.assertRaises(OSError):
                    r.append_event(Event("attempt.started"))
            pending_id = r.pending_commit["commit_id"]
            with self.assertRaises(Conflict):
                r.append_event(Event("must.not.proceed"))
            self.assertEqual(r.retry_pending().revision, 2)
            ep = store.read_episode(r.episode)
            self.assertEqual(ep.commits[-1]["commit_id"], pending_id)
            self.assertEqual(len(ep.events), 2)

    def test_symlink_commit_directory_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            store = FileTraceStore(Path(root) / "store")
            ep = store.create_episode({}, tenant="t", run_id="r")
            commits = store.root / ep.key / "commits"
            commits.rmdir()
            commits.symlink_to(Path(root), target_is_directory=True)
            with self.assertRaises(Corruption):
                store.read_episode(ep)
