"""Full observable evidence and explicitly trimmed judge views."""
import copy
import base64
from .providers.json_utils import strict_json


def canonical_trajectory(store, episode):
    ep = store.read_episode(episode)
    pending, events = {}, []
    for event in ep.events:
        kind, cid = event['type'], event.get('tool_call_id')
        row = {'event_id':event['event_id'], 'sequence':event['sequence'], 'canonical_type':kind,
               'event_type':'execution_event', 'evidence_source':'canonical_recorder', 'payload_ref':event.get('payload_ref')}
        if kind == 'tool.requested':
            pending[cid] = strict_json(store.read_payload(event['payload_ref']))
            continue
        if kind in {'tool.completed', 'tool.outcome_unknown'}:
            request = pending.get(cid, {})
            raw = store.read_payload(event['payload_ref'])
            pk = event['data']['payload_kind']
            value = strict_json(raw) if pk == 'json' else {'raw_text':raw.decode()} if pk == 'utf8' else {'encoding':'base64','data':base64.b64encode(raw).decode()}
            row.update(event_type='tool_interaction', tool_name=request.get('name'), arguments=request.get('arguments_raw'),
                       observation=value, tool_call_id=cid,
                       status='unknown' if kind == 'tool.outcome_unknown' else 'error' if event['data']['is_error'] else 'success')
        elif kind == 'model.completed':
            output = strict_json(store.read_payload(event['payload_ref']))
            row.update(event_type='final_response' if output['finish_status']=='completed' else 'model_step',
                       status=output['finish_status'], content=''.join(b['text'] for b in output['text_blocks']))
        events.append(row)
    return {'schema_version':'2.0','source':'canonical_recorder','provenance':'observed','episode_id':episode.episode_id,
            'completeness':copy.deepcopy(ep.completeness),'execution_status':ep.status,'events':events,
            'summary':{'tool_interactions':sum(e['event_type']=='tool_interaction' for e in events),
                       'retrieval_interactions':0,'tool_errors':sum(e.get('status') in {'error','unknown'} for e in events)}}


def judge_view(trajectory, max_events=80):
    if type(max_events) is not int or max_events < 1:
        raise ValueError('positive limit required')
    view = copy.deepcopy(trajectory)
    events = view.get('events', [])
    view['events'] = events[:max_events]
    view['omitted_event_ids'] = [e['event_id'] for e in events[max_events:]]
    view['view_transform'] = {'method':'first_n_events','limit':max_events,'original_event_count':len(events)}
    return view
