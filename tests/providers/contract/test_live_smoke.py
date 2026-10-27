"""Live provider check is explicit opt-in and skipped in offline CI."""
import os
import tempfile
import unittest

from membraneclaw_runtime.providers.smoke import smoke


@unittest.skipUnless(
    os.environ.get("MEMBRANECLAW_LIVE_SMOKE") == "1"
    and os.environ.get("GEMINI_API_KEY") and os.environ.get("GEMINI_MODEL_ID"),
    "requires explicit live opt-in, GEMINI_API_KEY and GEMINI_MODEL_ID")
class LiveGeminiSmoke(unittest.IsolatedAsyncioTestCase):
    async def test_live_text(self):
        with tempfile.TemporaryDirectory() as directory:
            episode, output = await smoke(os.environ["GEMINI_MODEL_ID"], directory)
            self.assertEqual(output.finish_status, "completed")
            self.assertTrue(output.native_response_ref)
