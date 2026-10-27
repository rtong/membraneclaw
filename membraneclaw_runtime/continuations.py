"""Single-host, identity-bound client handoffs. Checkpoints are authoritative."""
import copy
import hmac
import json
import secrets
import time
from dataclasses import asdict

from .gateway import disk
from .providers.json_utils import strict_json
from .providers.types import ConfigurationError
from .tracing.schema import canonical_json, digest


class ContinuationError(Exception):
    def __init__(self, code, status=409):
        self.code, self.status = code, status
        super().__init__(code)


def fingerprint(messages):
    return digest(canonical_json(messages))


class Continuations:
    def __init__(self, store, *, ttl=900, clock=time.time):
        if type(ttl) is not int or ttl < 1: raise ConfigurationError("positive continuation TTL required")
        self.store, self.ttl, self.clock = store, ttl, clock

    def find(self, handle):
        if not isinstance(handle, str) or not 40 <= len(handle) <= 64: raise ContinuationError("invalid_handle",404)
        cursor = None
        while True:
            page = self.store.list_episodes(cursor=cursor,limit=200)
            for ref in page.items:
                ep = self.store.read_episode(ref)
                if ep.checkpoint_ref:
                    state = strict_json(self.store.read_payload(ep.checkpoint_ref))
                    for phase in state.get("application_state",{}).get("continuations",[]):
                        if handle in phase["handles"]: return ref
            cursor=page.next_cursor
            if cursor is None: break
        raise ContinuationError("invalid_handle",404)

    async def issue(self, runtime, prepared, outcomes, *, owner, messages):
        if not owner: raise ContinuationError("verified_identity_required",401)
        g=runtime.gateway
        calls=[];handles={}
        for p in prepared:
            handle=secrets.token_urlsafe(32)
            handles[handle]={"call_id":p.call.id,"provider_call_id":p.call.provider_call_id,"name":p.call.name}
            calls.append({"id":handle,"type":"function","function":{"name":p.spec.name,"arguments":canonical_json(p.arguments).decode()}})
        message={"role":"assistant","content":g._last_output.text or None,"tool_calls":calls}
        state=copy.deepcopy(g.application_state)
        phases=state.setdefault("continuations",[])
        for phase in phases:
            if phase["status"]=="resuming":phase.update(status="consumed",reply=message)
        phases.append({"owner":fingerprint(owner),"expires":self.clock()+self.ttl,
                       "prefix":fingerprint(messages+[message]),"handles":handles,"results":{},
                       "local_results":[asdict(o.result) for o in outcomes.values()],
                       "request":json.loads(json.dumps(asdict(runtime.request))),"limits":asdict(runtime.limits),
                       "status":"pending","reply":message})
        g.set_application_state(state)
        await g.pause()
        return message

    def verify(self, ep, state, owner, prefix, incoming):
        if not owner: raise ContinuationError("verified_identity_required",401)
        matches=[p for p in state["application_state"].get("continuations",[]) if set(incoming) <= set(p["handles"])]
        if len(matches)!=1:raise ContinuationError("conflicting_handles")
        phase=matches[0]
        if not hmac.compare_digest(phase["owner"],fingerprint(owner)):raise ContinuationError("owner_mismatch",403)
        if self.clock()>=phase["expires"]:raise ContinuationError("continuation_expired",410)
        if not hmac.compare_digest(phase["prefix"],fingerprint(prefix)):raise ContinuationError("history_mismatch")
        for handle,value in incoming.items():
            if handle in phase["results"] and phase["results"][handle]!=value:raise ContinuationError("result_conflict")
        if phase["status"]=="consumed":
            if not all(h in phase["results"] for h in incoming):raise ContinuationError("result_conflict")
            return phase,True
        if ep.status!="awaiting_client_tool" or phase["status"]!="pending":raise ContinuationError("outcome_unknown")
        return phase,False

    async def complete(self, runtime, status, reason, text):
        g=runtime.gateway;state=copy.deepcopy(g.application_state)
        for phase in state.get("continuations",[]):
            if phase["status"]=="resuming":
                phase.update(status="consumed",reply={"role":"assistant","content":text or None},
                             final_status=status,reason=reason)
        g.set_application_state(state)
