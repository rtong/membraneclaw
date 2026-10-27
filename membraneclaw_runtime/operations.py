"""Local read-only gauges derived from authoritative commits and archive receipts."""
import json
from pathlib import Path


def health(store, *, archive_state=None):
    counts={'episodes':0,'incomplete_episodes':0,'unknown_tool_outcomes':0,'provider_attempts':0,
            'retry_count':0,'pending_continuations':0,'archive_pending':0,'archive_published':0}
    cursor=None
    while True:
        page=store.list_episodes(cursor=cursor,limit=200)
        for ref in page.items:
            ep=store.read_episode(ref);counts['episodes']+=1
            counts['incomplete_episodes']+=ep.completeness['capture']!='complete'
            counts['pending_continuations']+=ep.status=='awaiting_client_tool'
            counts['unknown_tool_outcomes']+=sum(e['type']=='tool.outcome_unknown' for e in ep.events)
            counts['provider_attempts']+=sum(e['type']=='attempt.started' for e in ep.events)
            counts['retry_count']+=sum(e['type']=='attempt.retry_scheduled' for e in ep.events)
        cursor=page.next_cursor
        if cursor is None:break
    if hasattr(store,'root'):
        counts['storage_bytes']=sum(p.stat().st_size for p in store.root.rglob('*') if p.is_file() and not p.is_symlink())
    if archive_state:
        for p in Path(archive_state).glob('*.receipt.json'):
            receipt=json.loads(p.read_bytes())
            counts['archive_pending']+=receipt['status']=='pending'
            counts['archive_published']+=receipt['status']=='published'
    return counts

if __name__=='__main__':
    import argparse
    from .tracing.file_store import FileTraceStore
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-dir',default='./var/traces');parser.add_argument('--archive-state')
    args=parser.parse_args()
    print(json.dumps(health(FileTraceStore(args.trace_dir),archive_state=args.archive_state),indent=2))
