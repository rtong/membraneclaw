"""Qwen wrappers around shared identity-bound Google implementations."""
from qwen_agent.tools.base import BaseTool
from agent.tools import google_impl as impl
from agent.tools.google_impl import (extract_id, _quote_range, _clip, google_oauth,
    DRIVE_API, SHEETS_API, MAX_CHARS, TIMEOUT)


def _qwen_tool(cls):
    class Wrapped(cls, BaseTool):
        def __init__(self, token_path, login_url=None, cfg=None):
            BaseTool.__init__(self, cfg)
            cls.__init__(self, token_path, login_url, cfg)
    Wrapped.__name__ = cls.__name__
    Wrapped.__qualname__ = cls.__name__
    return Wrapped


_GoogleTool = _qwen_tool(impl._GoogleTool)
DriveSearch = _qwen_tool(impl.DriveSearch)
DriveRead = _qwen_tool(impl.DriveRead)
SheetsInfo = _qwen_tool(impl.SheetsInfo)
SheetsGetValues = _qwen_tool(impl.SheetsGetValues)
SheetsUpdateValues = _qwen_tool(impl.SheetsUpdateValues)
GoogleDisconnect = _qwen_tool(impl.GoogleDisconnect)
ALL_TOOL_CLASSES = [DriveSearch, DriveRead, SheetsInfo, SheetsGetValues, SheetsUpdateValues, GoogleDisconnect]
TOOL_CLASSES = [DriveRead, SheetsGetValues, GoogleDisconnect]
ALL_TOOL_NAMES = [c.name for c in ALL_TOOL_CLASSES]
TOOL_NAMES = [c.name for c in TOOL_CLASSES]


def google_tools(token_path, login_url=None, allow=None):
    classes = TOOL_CLASSES if allow is None else [c for c in ALL_TOOL_CLASSES if c.name in allow]
    return [cls(token_path, login_url) for cls in classes]
