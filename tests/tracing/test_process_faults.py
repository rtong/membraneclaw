"""Real process death and POSIX lock tests (no API keys/services)."""
import json
import multiprocessing
import os
import signal
import tempfile
import unittest
from pathlib import Path

from membraneclaw_runtime.tracing import FileTraceStore, Event, Payload
from membraneclaw_runtime.tracing.schema import Busy, EpisodeRef, digest
from membraneclaw_runtime.tracing.validation import validate


def killed_writer(root, stage):
    def fault(current):
        if current == stage:
            os.kill(os.getpid(), signal.SIGKILL)
    store = FileTraceStore(root, fault_hook=fault)
    ep = EpisodeRef("t", "r", "e")
    p = Payload(b"synthetic raw response")
    with store.execution_guard(ep):
        store.commit(ep, "stable_commit", 0,
                     [Event("attempt.response", event_id="stable_event", wall_time="fixed",
                            payload_ref=p.ref(ep))], [p])


def hold_execution(root, ready, release):
    store = FileTraceStore(root)
    with store.execution_guard(EpisodeRef("t", "r", "e")):
        ready.set()
        release.wait(10)


def racing_writer(root, start, results):
    store = FileTraceStore(root)
    start.wait(10)
    try:
        store.commit(EpisodeRef("t", "r", "e"), "same", 0,
                     [Event("attempt.started", event_id="stable", wall_time="fixed")], [])
        results.put("committed")
    except Busy:
        results.put("busy")


class ProcessFaults(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ctx = multiprocessing.get_context("spawn")

    def test_sigkill_at_durability_boundaries(self):
        for stage, committed in [("payload_fsync", False), ("payload_committed", False),
                                 ("commit_before_rename", False), ("commit_committed", True),
                                 ("manifest_before_update", True)]:
            with self.subTest(stage=stage):
                root = Path(self.tmp.name) / stage
                store = FileTraceStore(root)
                ep = store.create_episode({}, tenant="t", run_id="r", episode_id="e")
                process = self.ctx.Process(target=killed_writer, args=(str(root), stage))
                process.start()
                process.join(10)
                self.assertFalse(process.is_alive())
                self.assertEqual(process.exitcode, -signal.SIGKILL)
                report = validate(store, ep)
                self.assertEqual(report.revision, int(committed))
                p = Payload(b"synthetic raw response")
                if stage in ("payload_committed", "commit_before_rename"):
                    self.assertEqual(store.orphan_payloads(ep), [digest(p.data) + ".bin"])
                # OS released both locks; stable commit retry never duplicates.
                with store.execution_guard(ep):
                    result = store.commit(ep, "stable_commit", 0,
                                          [Event("attempt.response", event_id="stable_event", wall_time="fixed",
                                                 payload_ref=p.ref(ep))], [p])
                self.assertEqual(result.revision, 1)
                self.assertEqual(len(list(store.iter_events(ep))), 1)
                store.rebuild_manifest(ep)
                manifest = json.loads((root / ep.key / "manifest.json").read_bytes())
                self.assertEqual(manifest["revision"], 1)

    def test_cross_process_recovery_and_lock_release(self):
        store = FileTraceStore(Path(self.tmp.name) / "locks")
        ep = store.create_episode({}, tenant="t", run_id="r", episode_id="e")
        ready, release = self.ctx.Event(), self.ctx.Event()
        process = self.ctx.Process(target=hold_execution, args=(str(store.root), ready, release))
        process.start()
        try:
            self.assertTrue(ready.wait(5))
            with self.assertRaises(Busy):
                store.recover(ep)
            self.assertEqual(store.read_episode(ep).status, "running")
        finally:
            release.set()
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join()
        self.assertEqual(process.exitcode, 0)
        self.assertEqual(store.recover(ep).status, "interrupted")

    def test_two_processes_same_commit(self):
        store = FileTraceStore(Path(self.tmp.name) / "race")
        ep = store.create_episode({}, tenant="t", run_id="r", episode_id="e")
        start, results = self.ctx.Event(), self.ctx.Queue()
        processes = [self.ctx.Process(target=racing_writer, args=(str(store.root), start, results)) for _ in range(2)]
        for process in processes:
            process.start()
        start.set()
        for process in processes:
            process.join(10)
            self.assertFalse(process.is_alive())
            self.assertEqual(process.exitcode, 0)
        outcomes = [results.get(timeout=5) for _ in processes]
        self.assertIn("committed", outcomes)
        self.assertTrue(all(x in {"committed", "busy"} for x in outcomes))
        self.assertEqual(validate(store, ep).revision, 1)
