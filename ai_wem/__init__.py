"""WEM — Worker/Expert/Master AI Chat Engine.

Reusable library for multi-tier LLM chat with tool calling.
Apps provide their own tools, executor, and system prompt.
"""

from .models import ChatMessage, MessageRole, ToolCall, ToolResult
from .providers import LLMProvider, HttpApiProvider, HttpOllamaProvider
from .engine import WEMEngine
from .executor import ToolExecutor
from .config import WEMConfig
from .scripts import ScriptIndex

__version__ = "0.1.0"

__all__ = [
    "ChatMessage", "MessageRole", "ToolCall", "ToolResult",
    "LLMProvider", "HttpApiProvider", "HttpOllamaProvider",
    "WEMEngine", "ToolExecutor", "WEMConfig", "ScriptIndex",
]
