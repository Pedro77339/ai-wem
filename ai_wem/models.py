"""Data models for WEM AI Chat engine."""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from enum import Enum
import time


class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class ToolCall:
    """A tool call requested by the LLM."""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolResult:
    """Result of executing a tool call."""
    call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass
class ChatMessage:
    """A message in the chat conversation."""
    role: MessageRole
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_results: List[ToolResult] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    status: str = ""
