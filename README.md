# ai-wem

**Worker/Expert/Master** — a multi-tier AI chat engine with tool calling.

Route messages through up to 3 LLM tiers to balance cost, speed, and quality:

| Tier | Role | Default Model |
|------|------|---------------|
| **Worker** | Fast classifier + single tool call | Gemini 3 Flash (free) |
| **Expert** | Full tool-calling loop | Gemini 3 Pro |
| **Master** | Evaluates Expert, redoes if wrong | Claude Opus 4.6 |

Worker and Master are optional. If you only configure Expert, it works as a standard tool-calling chat engine.

## Install

```bash
pip install ai-wem
```

## Quick Start

```python
import asyncio
from ai_wem import WEMEngine, WEMConfig, ToolExecutor, ToolCall, ToolResult

# 1. Implement your tool executor
class MyExecutor(ToolExecutor):
    async def execute(self, tc: ToolCall) -> ToolResult:
        if tc.name == "get_weather":
            city = tc.arguments.get("city", "NYC")
            return ToolResult(tc.id, tc.name, f'{{"city": "{city}", "temp": 22}}')
        return ToolResult(tc.id, tc.name, "Unknown tool", is_error=True)

# 2. Define tools (OpenAI function-calling format)
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

# 3. Configure and run
async def main():
    config = WEMConfig(
        expert_api_key="your-gemini-key",
        # fast_api_key="..." to enable Worker tier
        # master_api_key="..." to enable Master tier
    )
    engine = WEMEngine(config)
    result = await engine.send_message(
        user_text="What's the weather in Tokyo?",
        system_prompt="You are a helpful assistant with weather tools.",
        tools=TOOLS,
        executor=MyExecutor(),
    )
    print(result["reply"])   # Natural language response
    print(result["tier"])    # "worker", "expert", or "master"
    print(result["cost"])    # Estimated cost in USD

asyncio.run(main())
```

## How It Works

```
User message
    |
    v
[Worker] — fast/free model classifies intent
    |         If match: single tool call + format → done ($0)
    |         If no match: fall through ↓
    v
[Expert] — full tool-calling loop (multiple rounds)
    |         Calls tools, reads results, generates response
    |         Returns response ↓
    v
[Master] — evaluates Expert's response
              If correct: pass through → done
              If wrong: redo with Master model → done
```

## Configuration

```python
from ai_wem import WEMConfig

config = WEMConfig(
    # Expert (required) — main tool-calling model
    expert_model="gemini-3-pro-preview",
    expert_base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    expert_api_key="your-key",

    # Worker (optional) — fast/free classifier
    fast_model="gemini-3-flash-preview",
    fast_base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    fast_api_key="your-key",

    # Master (optional) — evaluator
    master_model="claude-opus-4-6",
    master_base_url="https://api.anthropic.com",
    master_api_key="your-anthropic-key",

    # Limits
    max_iterations=15,        # Max tool-calling rounds
    max_result_chars=10000,   # Truncate tool results
)
```

### Worker Fast Path

Enable the Worker tier by providing a classify prompt and intent map:

```python
result = await engine.send_message(
    user_text="weather in Paris",
    system_prompt="...",
    tools=TOOLS,
    executor=executor,
    # Worker classification
    classify_prompt=(
        'Classify the intent. Respond with JSON: {"intent": "...", "params": {...}}\n'
        'Categories: weather, products, none\n'
        'Question: {user_text}'
    ),
    intent_map={
        "weather": ("get_weather", {"city": "NYC"}),  # tool_name, default_params
        "products": ("search_products", {"query": "popular"}),
    },
)
```

Or use a custom worker hook:

```python
async def my_worker(user_text, executor):
    if "weather" in user_text.lower():
        tc = ToolCall(id="w1", name="get_weather", arguments={"city": "NYC"})
        result = await executor.execute(tc)
        return f"Weather: {result.content}"  # Return string = handled
    return None  # Return None = fall through to Expert

result = await engine.send_message(
    ...,
    worker_hook=my_worker,
)
```

## Providers

### Cloud APIs (OpenAI, Gemini, Claude)

`HttpApiProvider` handles OpenAI-compatible APIs and auto-detects Anthropic:

```python
from ai_wem import HttpApiProvider

# Gemini (OpenAI-compatible)
gemini = HttpApiProvider(
    "https://generativelanguage.googleapis.com/v1beta/openai",
    "your-key", "gemini-3-pro-preview"
)

# Claude (auto-detected by URL)
claude = HttpApiProvider(
    "https://api.anthropic.com",
    "your-key", "claude-opus-4-6"
)

# OpenAI
openai = HttpApiProvider(
    "https://api.openai.com",
    "your-key", "gpt-4o"
)
```

### Local Models (Ollama)

`HttpOllamaProvider` auto-detects native tool support:

```python
from ai_wem import HttpOllamaProvider

ollama = HttpOllamaProvider("http://localhost:11434", "qwen3:8b")

# Use as Expert provider
engine.set_expert_provider(ollama)
```

Models without native tool calling automatically use prompt-based simulation via `<tool_call>` XML tags.

## Script System (Optional)

For apps that want deterministic task execution (no LLM needed for known patterns):

```python
from ai_wem import ScriptIndex

scripts = ScriptIndex("/path/to/scripts/")

# Scripts are .py files with:
# DESCRIPTION = "Get device status"
# PARAMS = {"device_id": {"type": "str", "required": True}}
# def run(api, **kwargs) -> dict: ...

result = scripts.run("device_status", api=my_api, device_id="DEV-001")
```

## Sync Wrapper

For non-async apps (e.g., Flet UI):

```python
result = engine.send_message_sync(
    user_text="...",
    system_prompt="...",
    tools=TOOLS,
    executor=executor,
)
```

## Demo

Interactive CLI for testing models:

```bash
# Gemini (default)
GEMINI_API_KEY=... python -m ai_wem.demo

# Ollama local
python -m ai_wem.demo --provider ollama --model qwen3:8b

# Claude
ANTHROPIC_API_KEY=... python -m ai_wem.demo --provider claude
```

Commands: `/switch <provider>`, `/model <name>`, `/tier`, `/clear`, `/quit`

## API Reference

### `WEMEngine.send_message()` → `dict`

Returns:
```python
{
    "reply": str,    # Natural language response
    "tokens": int,   # Total tokens used
    "cost": float,   # Estimated cost in USD
    "tier": str,     # "worker", "expert", or "master"
}
```

### `ToolExecutor` (ABC)

```python
class ToolExecutor(ABC):
    async def execute(self, tool_call: ToolCall) -> ToolResult: ...
```

### `ToolCall`

```python
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
```

### `ToolResult`

```python
@dataclass
class ToolResult:
    call_id: str
    name: str
    content: str
    is_error: bool = False
```

## License

MIT
