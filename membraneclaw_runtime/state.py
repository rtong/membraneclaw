"""Application limits and returned execution status, separate from capture status."""
import math
from dataclasses import dataclass

from .providers.types import ConfigurationError


@dataclass(frozen=True)
class RuntimeLimits:
    max_steps: int = 12
    max_tool_calls: int = 32
    tool_timeout_seconds: float = 60
    max_context_bytes: int | None = None
    parallel_read_only: bool = False

    def validate(self):
        for value in (self.max_steps, self.max_tool_calls):
            if type(value) is not int or value < 1:
                raise ConfigurationError("step and tool budgets must be positive integers")
        if (type(self.tool_timeout_seconds) not in (int, float) or not math.isfinite(self.tool_timeout_seconds)
                or self.tool_timeout_seconds <= 0):
            raise ConfigurationError("tool timeout must be finite and positive")
        if self.max_context_bytes is not None and (type(self.max_context_bytes) is not int or self.max_context_bytes < 1):
            raise ConfigurationError("context byte limit must be a positive integer")
        if type(self.parallel_read_only) is not bool:
            raise ConfigurationError("parallel_read_only must be boolean")


@dataclass(frozen=True)
class RunResult:
    episode_id: str
    status: str
    reason: str
    text: str
    steps: int
    tool_calls: int
