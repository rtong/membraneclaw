"""CLI provider boundary. The legacy Qwen branch remains lazy and keeps its history behavior."""
import argparse
import asyncio
import os
import sys


def render(chunk, printed_upto):
    settled = max(len(chunk) - 1, 0)
    for msg in chunk[printed_upto:settled]:
        role = msg.get("role")
        if role == "assistant" and msg.get("function_call"):
            fc = msg["function_call"]
            print(f"\n  [tool] {fc['name']}({fc['arguments']})", flush=True)
        elif role == "function":
            preview = str(msg.get("content", ""))
            if len(preview) > 300:
                preview = preview[:300] + " ..."
            print(f"  [result] {preview}", flush=True)
    return settled


def run_qwen(args):
    from agent import config, google_oauth
    from agent.factory import build_qwen_agent
    if args.model:
        config.MODEL = args.model
    agent = build_qwen_agent(files=args.file, google_token_path=google_oauth.TOKEN_FILE)
    print(f"model={config.MODEL} @ {config.BASE_URL}  native_tools={config.NATIVE_TOOL_CALLS}"
          + (f"  files={len(args.file)}" if args.file else ""))
    history = []
    def turn(user_text):
        history.append({"role": "user", "content": user_text})
        shown, final = 0, []
        for chunk in agent.run(messages=history):
            shown = render(chunk, shown)
            final = chunk
        render(final + [None], shown)
        for msg in final:
            if msg.get("role") == "assistant" and msg.get("content"):
                print(f"\n{msg['content'].strip()}\n")
        history.extend(final)
    if args.prompt:
        turn(args.prompt)
        return 0
    print("Type your message. Ctrl-D or 'exit' to quit.\n")
    while True:
        try:
            user_text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_text in {"exit", "quit"}:
            break
        if not user_text:
            continue
        try:
            turn(user_text)
        except Exception as exc:
            print(f"[error] {exc}", file=sys.stderr)
    return 0


async def run_traced(args, prompt):
    from agent.factory import runtime_session
    from .gateway import Budget
    from .state import RuntimeLimits
    token_path = None
    if args.google_tools:
        from agent.google_oauth import TOKEN_FILE
        token_path = TOKEN_FILE  # CLI is the local operator, never an HTTP user.
    limits = RuntimeLimits(args.max_steps, args.max_tool_calls, args.tool_timeout,
                           args.max_context_bytes, args.parallel_read_only)
    budget = Budget(max_requests=args.max_requests, max_elapsed_seconds=args.max_seconds)
    async with runtime_session(provider=args.provider, model=args.model,
            trace_backend=args.trace_backend, trace_dir=args.trace_dir, prompt=prompt,
            no_mcp=args.no_mcp, google_token_path=token_path, limits=limits, budget=budget) as runtime:
        print(f"episode={runtime.gateway.recorder.episode.episode_id} provider={args.provider}", flush=True)
        result = await runtime.run(stream=args.stream)
        if result.text:
            print(result.text)
        print(f"episode={result.episode_id} status={result.status} reason={result.reason}")
        return 0 if result.status == "completed" else 1


def main(argv=None):
    from agent import config  # Loads .env without importing Qwen or applying patches.
    parser = argparse.ArgumentParser(description="MembraneClaw agent harness")
    parser.add_argument("-f", "--file", action="append", default=[], help="Attach a document for Qwen RAG")
    parser.add_argument("-p", "--prompt", help="Run a single prompt and exit")
    parser.add_argument("--provider", default=os.environ.get("AGENT_PROVIDER", "qwen"))
    parser.add_argument("--model", default=os.environ.get("AGENT_MODEL"))
    parser.add_argument("--trace-backend", default="files")
    parser.add_argument("--trace-dir", default=os.environ.get("TRACE_DIR", "./var/traces"))
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--max-tool-calls", type=int, default=32)
    parser.add_argument("--tool-timeout", type=float, default=60)
    parser.add_argument("--max-context-bytes", type=int)
    parser.add_argument("--max-requests", type=int, default=36)
    parser.add_argument("--max-seconds", type=float, default=300)
    parser.add_argument("--parallel-read-only", action="store_true")
    parser.add_argument("--stream", action="store_true", help="Use durably captured native SSE")
    parser.add_argument("--no-mcp", action="store_true")
    parser.add_argument("--google-tools", action="store_true", help="Bind Google tools to this CLI operator")
    args = parser.parse_args(argv)
    if args.provider == "gemini" and not args.model:
        args.model = os.environ.get("GEMINI_MODEL_ID")
    if args.provider == "qwen":
        return run_qwen(args)
    if args.file:
        parser.error("document RAG is currently supported only by the Qwen path")
    if not args.model:
        parser.error("an explicit --model or AGENT_MODEL/GEMINI_MODEL_ID is required")
    def turn(prompt):
        try:
            return asyncio.run(run_traced(args, prompt))
        except KeyboardInterrupt:
            print("[error] cancelled", file=sys.stderr)
            return 130
        except Exception as error:
            # Never expose arbitrary exception URLs or credential values in normal logs.
            print(f"[error] {type(error).__name__}", file=sys.stderr)
            return 1
    if args.prompt is not None:
        return turn(args.prompt)
    print("Each prompt starts a new traced episode. Ctrl-D or 'exit' to quit.")
    while True:
        try:
            prompt = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if prompt in {"exit", "quit"}:
            break
        if prompt:
            turn(prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
