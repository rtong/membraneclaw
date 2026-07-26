import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()

BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")
MODEL = os.environ.get("VLLM_MODEL", "qwen3.5-9b")

# Qwen3.5 is trained to emit XML tool calls, which vLLM parses server-side with
# --tool-call-parser qwen3_xml. Passing tools natively keeps the model on that
# format instead of Qwen-Agent's prompt-injected JSON dialect.
NATIVE_TOOL_CALLS = os.environ.get("NATIVE_TOOL_CALLS", "true").lower() == "true"

# With thinking on, the model routinely finishes its whole answer inside the
# <think> block and emits nothing after it, so --reasoning-parser qwen3 hands back
# an empty `content` and the agent loop stalls. Off by default; set
# ENABLE_THINKING=true only if you also handle empty-content turns.
ENABLE_THINKING = os.environ.get("ENABLE_THINKING", "false").lower() == "true"


def llm_config() -> dict:
    return {
        "model": MODEL,
        "model_server": BASE_URL,
        "api_key": API_KEY,
        "generate_cfg": {
            "temperature": 0.7,
            "top_p": 0.8,
            "max_input_tokens": 14000,
            "use_raw_api": NATIVE_TOOL_CALLS,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": ENABLE_THINKING}},
        },
    }
