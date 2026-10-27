import json5
from qwen_agent.tools.base import BaseTool, register_tool

from agent.tools import builtin_impl


@register_tool("calculator")
class Calculator(BaseTool):
    description = "Evaluate an arithmetic expression and return the numeric result."
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Arithmetic expression, e.g. '(12.5 * 3) ** 2 / 7'",
            }
        },
        "required": ["expression"],
    }

    def call(self, params: str, **kwargs) -> str:
        return builtin_impl.calculator(json5.loads(params)["expression"])


@register_tool("http_get")
class HttpGet(BaseTool):
    description = "Fetch the body of an HTTP(S) URL. Use for reading public web pages or JSON APIs."
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Fully-qualified http:// or https:// URL",
            }
        },
        "required": ["url"],
    }

    def call(self, params: str, **kwargs) -> str:
        return builtin_impl.http_get(json5.loads(params)["url"], max_bytes=200_000)


@register_tool("now")
class Now(BaseTool):
    description = "Get the current local date and time. Call this instead of guessing the date."
    parameters = {"type": "object", "properties": {}, "required": []}

    def call(self, params: str, **kwargs) -> str:
        return builtin_impl.now()
