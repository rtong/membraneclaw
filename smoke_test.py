"""Verify the vLLM backend and the agent harness end to end."""
import json
import urllib.error
import urllib.request

from agent import config
from agent.core import build_agent


def check_raw_tool_calling():
    payload = {
        "model": config.MODEL,
        "messages": [{"role": "user", "content": "What is the weather in Paris? Use the tool."}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }],
        "tool_choice": "auto",
        "max_tokens": 300,
        "temperature": 0,
    }
    req = urllib.request.Request(
        f"{config.BASE_URL}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.API_KEY}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.load(resp)
    except urllib.error.HTTPError as exc:
        return False, exc.read().decode()[:300]

    calls = body["choices"][0]["message"].get("tool_calls")
    if not calls:
        return False, f"no tool_calls; content={body['choices'][0]['message'].get('content')!r}"
    return True, calls[0]["function"]


def run_agent(prompt, files=None):
    agent = build_agent(files=files)
    final, tools_used = [], []
    for chunk in agent.run(messages=[{"role": "user", "content": prompt}]):
        final = chunk
    for msg in final:
        if msg.get("role") == "assistant" and msg.get("function_call"):
            tools_used.append(msg["function_call"]["name"])
    answer = next(
        (m["content"].strip() for m in reversed(final)
         if m.get("role") == "assistant" and m.get("content")),
        "",
    )
    return tools_used, answer


if __name__ == "__main__":
    print("=" * 60)
    print("1. Native server-side tool parsing")
    ok, detail = check_raw_tool_calling()
    print(f"   {'PASS' if ok else 'FAIL'}: {detail}")

    print("=" * 60)
    print("2. Agent + custom tool (calculator)")
    tools, answer = run_agent("What is 4871 * 293? Use the calculator tool.")
    print(f"   tools={tools}")
    print(f"   answer={answer[:200]}")
    print(f"   {'PASS' if '1427203' in answer.replace(',', '') else 'CHECK'}")

    print("=" * 60)
    print("3. Plain chat, no tools")
    _, answer = run_agent("In one sentence, what is a membrane protein?")
    print(f"   answer={answer[:200]}")

    print("=" * 60)
    print("4. RAG over the synthetic fixture")
    tools, answer = run_agent(
        "What port does the service listen on, according to the document?",
        files=[str(config.ROOT / "tests" / "fixtures" / "service_notes.md")],
    )
    print(f"   answer={answer[:300]}")
