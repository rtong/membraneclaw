import ast
import json
import operator
import urllib.request

import json5
from qwen_agent.tools.base import BaseTool, register_tool

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


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
        expression = json5.loads(params)["expression"]
        try:
            return str(_eval(ast.parse(expression, mode="eval").body))
        except Exception as exc:
            return f"Error: {exc}"


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
        url = json5.loads(params)["url"]
        if not url.startswith(("http://", "https://")):
            return "Error: url must start with http:// or https://"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "MembraneClaw/1.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                body = resp.read(200_000).decode("utf-8", errors="replace")
            return body
        except Exception as exc:
            return f"Error: {exc}"


@register_tool("now")
class Now(BaseTool):
    description = "Get the current local date and time. Call this instead of guessing the date."
    parameters = {"type": "object", "properties": {}, "required": []}

    def call(self, params: str, **kwargs) -> str:
        from datetime import datetime

        return json.dumps({"now": datetime.now().astimezone().isoformat()})
