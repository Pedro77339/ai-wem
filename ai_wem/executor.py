"""Tool executor abstract base class.

Each app implements this with its own domain-specific tool handlers.
"""

from abc import ABC, abstractmethod
from .models import ToolCall, ToolResult


class ToolExecutor(ABC):
    """Abstract tool executor. Apps must implement execute()."""

    @abstractmethod
    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call. Must be async.

        Args:
            tool_call: The tool call with name and arguments.

        Returns:
            ToolResult with content string and is_error flag.
        """
        ...
