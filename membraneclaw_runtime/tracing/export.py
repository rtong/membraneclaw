"""Explicit private replay bundles and default content-free analysis exports."""
import json
import os
import zipfile
from dataclasses import asdict
from pathlib import Path

from .schema import Corruption, EpisodeRef, canonical_json, digest
from .validation import SealedSnapshot, restore, seal, validate


def analysis_view(store, episode):
    ep=store.read_episode(episode)
    return {"trace_schema_version":1,"derived":True,"redacted":True,"replay":"unsupported",
            "episode_id":episode.episode_id,"status":ep.status,
            "completeness":{"capture":"redacted","replay":"unsupported","external_outcome":ep.completeness["external_outcome"]},
            "events":[{"event_id":e["event_id"],"sequence":e["sequence"],"type":e["type"]} for e in ep.events]}


def export_bundle(store, episode, output, *, mode="analysis"):
    if mode not in {"analysis","private-replay"}:raise ValueError("unsupported export mode")
    if mode=="analysis":
        files={"analysis.json":canonical_json(analysis_view(store,episode))}
        envelope={"format":"membraneclaw_bundle_v1","mode":mode,"sensitive":False,"replay":"unsupported"}
    else:
        snapshot=seal(store,episode)
        files={"trace/"+p:b for p,b in snapshot.files}
        envelope={"format":"membraneclaw_bundle_v1","mode":mode,"sensitive":True,
                  "episode":asdict(snapshot.episode),"revision":snapshot.revision,"snapshot_hash":snapshot.snapshot_hash}
    envelope["files"]=[{"path":p,"sha256":digest(b),"size":len(b)} for p,b in sorted(files.items())]
    output=Path(output)
    # O_EXCL prevents replacing an unrelated artifact; mode excludes other users before bytes are written.
    fd=os.open(output,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    try:
        with os.fdopen(fd,"wb") as target:
            with zipfile.ZipFile(target,"w",compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("bundle.json",canonical_json(envelope))
                for path,data in sorted(files.items()):archive.writestr(path,data)
            target.flush();os.fsync(target.fileno())
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return envelope


def load_bundle(path, *, max_bytes=256*1024*1024):
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)) or sum(i.file_size for i in archive.infolist())>max_bytes:raise Corruption("duplicate/oversized bundle")
        if "bundle.json" not in names:raise Corruption("bundle manifest missing")
        envelope=json.loads(archive.read("bundle.json"))
        if envelope.get("format")!="membraneclaw_bundle_v1" or envelope.get("mode")!="private-replay" or envelope.get("sensitive") is not True:
            raise Corruption("private replay bundle required")
        expected={"bundle.json"}|{x["path"] for x in envelope["files"]}
        if set(names)!=expected or len(expected)!=len(envelope["files"])+1:raise Corruption("bundle inventory mismatch")
        files=[]
        for item in envelope["files"]:
            name=item["path"]
            if not name.startswith("trace/"):raise Corruption("invalid bundle path")
            data=archive.read(name)
            if len(data)!=item["size"] or digest(data)!=item["sha256"]:raise Corruption("bundle content mismatch")
            files.append((name[6:],data))
    snapshot=SealedSnapshot(EpisodeRef(**envelope["episode"]),envelope["revision"],tuple(sorted(files)),envelope["snapshot_hash"])
    snapshot.verify()
    return snapshot


def restore_bundle(path,destination):
    snapshot=load_bundle(path)
    return restore(snapshot,destination),snapshot.episode
