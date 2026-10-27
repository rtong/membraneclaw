"""Shared builtin logic; no provider imports or global registration."""
import ast
import json
import operator
import urllib.request

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


def calculator(expression):
    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except Exception as exc:
        return f"Error: {exc}"


def http_get(url, *, max_bytes=None):
    if not url.startswith(("http://", "https://")):
        return "Error: url must start with http:// or https://"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MembraneClaw/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read() if max_bytes is None else resp.read(max_bytes)
            return body.decode("utf-8", errors="replace")
    except Exception as exc:
        return f"Error: {exc}"


def now():
    from datetime import datetime
    return json.dumps({"now": datetime.now().astimezone().isoformat()})
