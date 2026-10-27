"""Synthetic request/result trace. Does not contact a provider or execute tools."""
from . import Event, FileTraceStore, Payload, TraceRecorder
from .validation import validate


def main():
    store = FileTraceStore("var/traces")
    recorder = TraceRecorder.begin_episode(store, {"fixture": "synthetic", "provider": None})
    with store.execution_guard(recorder.episode):
        for kind, data in [
            ("attempt.started", {"input": "synthetic request"}),
            ("attempt.response", {"tool_call": {"id": "call1", "name": "demo_read"}}),
            ("tool.started", {"id": "call1", "arguments": {"value": 2}}),
            ("tool.completed", {"id": "call1", "result": 4}),
            ("model.completed", {"text": "synthetic final result: 4"}),
        ]:
            payload = Payload.json(data)
            recorder.append_event(Event(kind, payload_ref=payload.ref(recorder.episode)), payload=payload)
        recorder.checkpoint({"history": "synthetic", "pending": [], "next": "stop"})
        recorder.finish_episode(capture="complete", replay="unsupported", external_outcome="known")
    print(f"episode={recorder.episode.key} status={validate(store, recorder.episode).status}")


if __name__ == "__main__":
    main()
