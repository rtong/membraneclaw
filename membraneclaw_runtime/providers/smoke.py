"""Explicit, one-request real Gemini smoke. This is not the agent CLI (PR 5)."""
import argparse
import asyncio
import os

from ..gateway import Budget, ModelGateway, RetryPolicy
from ..tracing import FileTraceStore
from .gemini_transport import GeminiTransport
from .registry import default_registry
from .types import ContentBlock, ModelRequest, ModelRoute


async def smoke(model, trace_dir, credential_env="GEMINI_API_KEY"):
    route = ModelRoute("gemini", model, "generate_content_v1beta")
    adapter = default_registry().create(route)
    transport = GeminiTransport.from_env(credential_env)
    gw = await ModelGateway.start(FileTraceStore(trace_dir), adapter, transport, route,
                                  budget=Budget(max_requests=1), retry=RetryPolicy(max_attempts=1))
    try:
        out = await gw.generate(ModelRequest(route, (ContentBlock.text_block("Reply with the word OK."),),
                                            max_output_tokens=128,
                                            provider_options={"gemini": {"thinkingConfig": {"thinkingBudget": 0}}}))
        await gw.finish("completed" if out.finish_status == "completed" else "failed")
        return gw.recorder.episode, out
    except Exception:
        try:
            await gw.finish("failed")
        except Exception:
            pass
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Explicit reviewed Gemini model ID")
    parser.add_argument("--trace-dir", default=os.environ.get("TRACE_DIR", "./var/traces"))
    parser.add_argument("--credential-env", default="GEMINI_API_KEY")
    args = parser.parse_args()
    episode, out = asyncio.run(smoke(args.model, args.trace_dir, args.credential_env))
    print(f"episode={episode.key} finish_status={out.finish_status} total_tokens={out.usage.total_tokens}")


if __name__ == "__main__":
    main()
