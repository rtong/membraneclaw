import argparse
import sys

from agent import config
from agent.core import build_agent


def render(chunk, printed_upto):
    """Echo tool activity. The trailing message is still streaming, so hold it back."""
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


def main():
    parser = argparse.ArgumentParser(description="Qwen3.5 agent harness")
    parser.add_argument("-f", "--file", action="append", default=[],
                        help="Attach a document for RAG (repeatable)")
    parser.add_argument("-p", "--prompt", help="Run a single prompt and exit")
    args = parser.parse_args()

    agent = build_agent(files=args.file)
    print(f"model={config.MODEL} @ {config.BASE_URL}  "
          f"native_tools={config.NATIVE_TOOL_CALLS}"
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
        return

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


if __name__ == "__main__":
    main()
