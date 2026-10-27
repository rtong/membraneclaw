"""Opt-in native evaluation, with gateway recording for every model role."""
import asyncio
import time
from pathlib import Path
from dataclasses import asdict
from .openwebui import ChatResult, OpenWebUIAgentError, OpenWebUIError


class TracedEvaluationClient:
    native_tracing = True

    def __init__(self, generation, *, factory=None):
        from agent.factory import runtime_session
        self.defaults = dict(generation)
        self.factory = factory or runtime_session

    def chat(self, *, model, messages, generation):
        return asyncio.run(self._chat(model, messages, generation, False))

    def chat_stream(self, *, model, messages, generation):
        return asyncio.run(self._chat(model, messages, generation, True))

    async def _chat(self, model, messages, generation, stream):
        from membraneclaw_runtime.evaluation import canonical_trajectory
        from membraneclaw_runtime.tracing.export import export_bundle
        from membraneclaw_runtime.tracing.schema import canonical_json
        cfg = {**self.defaults, **generation}
        role = cfg.get('trace_role', 'solver')
        if role not in {'solver','router','teacher','judge','recovery'}:
            raise OpenWebUIError('unsupported native role')
        route = cfg.get('routes', {}).get(role, {})
        provider = route.get('provider', cfg.get('provider'))
        model = route.get('model', model)
        if provider not in {'gemini','openai','anthropic'}:
            raise OpenWebUIError('native provider required')
        if cfg.get('top_p') is not None or cfg.get('enable_thinking') is not None:
            raise OpenWebUIError('top_p/enable_thinking unsupported; use provider_options')
        context = {**cfg.get('trace_context',{}), 'dataset_split':cfg.get('dataset_split','unspecified'), 'model_role':role}
        root = Path(cfg.get('trace_dir','./var/traces/evaluation')).resolve()
        start, runtime, raw = time.perf_counter(), None, {}
        try:
            system = '\n'.join(m['content'] for m in messages if m['role']=='system')
            prompt = canonical_json({'portable_chat_history':[m for m in messages if m['role']!='system']}).decode()
            async with self.factory(provider=provider, model=model, prompt=prompt, system_instruction=system,
                    trace_dir=str(root), role=role, trace_context=context, no_mcp=cfg.get('no_mcp',True),
                    include_builtin_tools=cfg.get('tools_enabled',role=='solver'),
                    generation={'temperature':cfg.get('temperature'), 'max_output_tokens':cfg.get('max_tokens'),
                                'provider_options':route.get('provider_options',cfg.get('provider_options',{}))}) as runtime:
                result = await runtime.run(stream=stream)
                raw = self._raw(runtime, result.text, root)
                latency = round((time.perf_counter()-start)*1000)
                if result.status != 'completed':
                    raise OpenWebUIAgentError('traced execution '+result.reason,response_text=result.text,raw_response=raw,latency_ms=latency)
                return ChatResult(result.text,raw,latency)
        except OpenWebUIAgentError:
            raise
        except Exception as error:
            if runtime is not None and not raw:
                try:
                    g = runtime.gateway
                    if g.recorder.store.read_episode(g.recorder.episode).status == 'running':
                        g.mark_external_unknown()
                        await g.finish('failed')
                    raw = self._raw(runtime,'',root)
                except Exception:
                    pass  # A storage failure cannot promise an export.
            raise OpenWebUIAgentError('traced execution failed: '+type(error).__name__,response_text='',raw_response=raw,
                                     latency_ms=round((time.perf_counter()-start)*1000)) from None

    def _raw(self, runtime, text, root):
        from membraneclaw_runtime.evaluation import canonical_trajectory
        from membraneclaw_runtime.tracing.export import export_bundle
        g = runtime.gateway
        ep = g.recorder.store.read_episode(g.recorder.episode)
        directory = root/'bundles'
        directory.mkdir(parents=True,exist_ok=True,mode=0o700)
        bundle = directory/(ep.ref.episode_id+'.zip')
        export_bundle(g.recorder.store,ep.ref,bundle,mode='private-replay')
        return {'trace':{'schema_version':1,'episode_id':ep.ref.episode_id,'completeness':ep.completeness,'status':ep.status,
                         'locator':asdict(ep.ref),'private_bundle':str(bundle),'sensitive':True},
                'canonical_trajectory':canonical_trajectory(g.recorder.store,ep.ref),
                'usage':None if g.usage_unknown else {'total_tokens':g.tokens_observed},
                'choices':[{'message':{'role':'assistant','content':text},'finish_reason':'stop' if ep.status=='completed' else 'error'}]}
