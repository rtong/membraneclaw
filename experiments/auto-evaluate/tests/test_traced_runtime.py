import json
import socket
import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch
from auto_evaluate.traced import TracedEvaluationClient
from auto_evaluate.trajectory import extract_observable_trajectory
from auto_evaluate.judge import _limit_trajectory
from auto_evaluate.runner import make_client
from membraneclaw_runtime.providers.openai import OpenAIAdapter
from membraneclaw_runtime.providers.types import NativeResponse
from membraneclaw_runtime.tracing import FileTraceStore
from membraneclaw_runtime.tracing.schema import canonical_json,EpisodeRef

class Synthetic:
    authentication_values=()
    def __init__(self):self.requests=[]
    async def exchange(self,native,*,before_send,after_receive,timeout):
        await before_send(native);self.requests.append(native)
        r=NativeResponse(200,{},canonical_json({'status':'completed','output':[{'type':'message','role':'assistant','content':[{'type':'output_text','text':'synthetic answer'}]}]}))
        await after_receive(r);return r

class NativeEvaluation(unittest.TestCase):
    def test_all_roles_gateway_bundle_and_judge_isolation(self):
        from agent.factory import runtime_session
        with tempfile.TemporaryDirectory() as root:
            t=Synthetic()
            @asynccontextmanager
            async def factory(**kwargs):
                async with runtime_session(**kwargs,adapter=OpenAIAdapter(),transport=t) as rt:yield rt
            client=TracedEvaluationClient({'provider':'openai','trace_dir':root},factory=factory)
            with patch.object(socket,'create_connection',side_effect=AssertionError('network forbidden')):
                for role in ('solver','router','teacher','judge','recovery'):
                    r=client.chat(model='gpt-4.1-2025-04-14',messages=[{'role':'user','content':'synthetic'}],generation={
                        'trace_role':role,'dataset_split':'test','trace_context':{'case_id':'case1','parent_episode_id':'parent' if role=='recovery' else None}})
                    trace=r.raw['trace'];self.assertTrue(Path(trace['private_bundle']).exists())
                    ep=FileTraceStore(root).read_episode(EpisodeRef(**trace['locator']))
                    self.assertEqual(ep.metadata['role'],role);self.assertEqual(ep.metadata['context']['dataset_split'],'test')
                    if role!='solver':self.assertNotIn('tools',json.loads(t.requests[-1].body))
                    self.assertEqual(extract_observable_trajectory(r.content,raw_response=r.raw)['source'],'canonical_recorder')
            self.assertEqual(len(t.requests),5)
    def test_omitted_ids_and_legacy_provenance(self):
        events=[{'event_id':'e'+str(i),'event_type':'tool_interaction'} for i in range(100)]
        original={'events':events,'summary':{}}
        view=_limit_trajectory(original)
        self.assertEqual(view['omitted_event_ids'],[e['event_id'] for e in events[80:]])
        self.assertEqual(len(original['events']),100)
        self.assertEqual(extract_observable_trajectory('old')['provenance'],'legacy/inferred')
    def test_native_config_does_not_require_openwebui(self):
        with patch.dict('os.environ',{},clear=True):self.assertIsInstance(make_client({'provider':'gemini'}),TracedEvaluationClient)
    def test_native_judge_dispatch_and_evidence(self):
        from auto_evaluate.codex_worker import run
        from auto_evaluate.openwebui import ChatResult
        r=ChatResult('{"score":5}',{'trace':{'episode_id':'synthetic'},'canonical_trajectory':{}},1)
        with patch('auto_evaluate.traced.TracedEvaluationClient.chat',return_value=r),patch('auto_evaluate.codex_worker.validate_task_output',return_value=[]):
            out=run({'provider':'openai','model':'gpt-4.1-2025-04-14','stage':'judge','task':{'task_id':'j1','evidence':[{'event_id':'e1'}]},'prompt':'synthetic','retries':0})
        self.assertEqual(out['output']['evaluator_provenance']['evidence_event_ids'],['e1'])
