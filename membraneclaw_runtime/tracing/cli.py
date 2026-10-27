"""Local operator trace inspection; exports are redacted unless explicitly private."""
import argparse
import asyncio
import base64
import json
import tempfile
from dataclasses import asdict

from .file_store import FileTraceStore
from .schema import EpisodeRef
from .validation import validate
from .export import export_bundle, restore_bundle
from .replay import playback, execute_replay


def resolve(store,value):
    if "/" in value:return EpisodeRef(*value.split("/"))
    found=[];cursor=None
    while True:
        page=store.list_episodes(cursor=cursor,limit=200)
        found.extend(r for r in page.items if r.episode_id==value)
        cursor=page.next_cursor
        if cursor is None:break
    if len(found)!=1:raise ValueError("episode ID must resolve uniquely")
    return found[0]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("inspect","validate","export","replay"))
    p.add_argument("--trace-dir",default="./var/traces");p.add_argument("--episode");p.add_argument("--bundle")
    p.add_argument("--output");p.add_argument("--mode",choices=("analysis","private-replay"),default="analysis")
    p.add_argument("--offline",action="store_true",help="Replay is always offline")
    p.add_argument("--execute",action="store_true",help="Drive the current runtime with recorded providers/tools")
    a=p.parse_args(argv)
    try:
        with tempfile.TemporaryDirectory() as temp:
            if a.bundle:store,episode=restore_bundle(a.bundle,temp+"/trace")
            else:
                if not a.episode:p.error("--episode or --bundle is required")
                store=FileTraceStore(a.trace_dir);episode=resolve(store,a.episode)
            if a.command=="validate":result=asdict(validate(store,episode))
            elif a.command=="export":
                if not a.output:p.error("--output required")
                result=export_bundle(store,episode,a.output,mode=a.mode)
            elif a.command=="replay":result=asyncio.run(execute_replay(store,episode)) if a.execute else playback(store,episode)
            else:
                result=playback(store,episode)
                result["payloads"]={ref:{"encoding":"base64","data":base64.b64encode(store.read_payload(ref)).decode()}
                    for c in store.read_episode(episode).commits for ref in c["references"]}
            print(json.dumps(result,ensure_ascii=False,indent=2));return 0
    except Exception as error:
        print(json.dumps({"error":type(error).__name__}));return 1

if __name__=="__main__":raise SystemExit(main())
