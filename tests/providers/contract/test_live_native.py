"""Explicit opt-in live tests. Missing flags/credentials/models are skipped, never passed."""
import os
import tempfile
import unittest
from membraneclaw_runtime.gateway import ModelGateway,Budget
from membraneclaw_runtime.providers.registry import default_registry
from membraneclaw_runtime.providers.types import ModelRequest,ContentBlock
from membraneclaw_runtime.tracing import FileTraceStore
from membraneclaw_runtime.tracing.validation import validate


def configured(provider):
    prefix=provider.upper()
    return (os.environ.get('RUN_LIVE_PROVIDER_SMOKE')=='1' and bool(os.environ.get(prefix+'_API_KEY'))
            and bool(os.environ.get('LIVE_'+prefix+'_MODEL')))


class LiveNativeSmoke(unittest.IsolatedAsyncioTestCase):
    async def smoke(self,provider,transport_factory):
        model=os.environ['LIVE_'+provider.upper()+'_MODEL']
        adapter,route=default_registry().create_for_model(provider,model)
        with tempfile.TemporaryDirectory() as directory:
            store=FileTraceStore(directory)
            gateway=await ModelGateway.start(store,adapter,transport_factory(),route,
                budget=Budget(max_requests=1,max_elapsed_seconds=60,attempt_timeout_seconds=45))
            output=await gateway.generate(ModelRequest(route,(ContentBlock.text_block('Reply with OK.'),),max_output_tokens=4096))
            self.assertEqual(output.finish_status,'completed')
            self.assertTrue(output.text.strip())
            await gateway.finish()
            report=validate(store,gateway.recorder.episode)
            self.assertEqual(report.status,'completed')
            self.assertEqual(report.completeness['capture'],'complete')

    @unittest.skipUnless(configured('openai'),'requires explicit live flag, OPENAI_API_KEY and LIVE_OPENAI_MODEL')
    async def test_openai(self):
        from membraneclaw_runtime.providers.openai_transport import OpenAITransport
        await self.smoke('openai',OpenAITransport.from_env)

    @unittest.skipUnless(configured('anthropic'),'requires explicit live flag, ANTHROPIC_API_KEY and LIVE_ANTHROPIC_MODEL')
    async def test_anthropic(self):
        from membraneclaw_runtime.providers.anthropic_transport import AnthropicTransport
        await self.smoke('anthropic',AnthropicTransport.from_env)
