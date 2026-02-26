# ai-wem — Worker/Expert/Master AI Chat Engine

You are helping a user set up an AI chat application using the `ai-wem` library.

## What is ai-wem?

A Python library that routes chat messages through up to 3 LLM tiers:

- **Worker** (optional) — fast/free model that handles simple, repetitive questions instantly
- **Expert** (required) — main model with full tool-calling loop for complex questions
- **Master** (optional) — evaluator that checks Expert's answer and redoes if wrong

## Your job

Help the user create a working chat by generating these 4 things:

### 1. System Prompt

Ask the user: **"What is your chat assistant supposed to do?"**

From their answer, write a system prompt that tells the LLM:
- What domain it works in (e.g., inventory, customer support, IoT monitoring)
- What tone to use (formal, casual, technical)
- What language to respond in
- What it should NOT do (e.g., don't make up data, don't execute destructive operations)

```python
SYSTEM_PROMPT = """You are an inventory assistant for a retail store.
You have access to tools that query the product database.
Always use tools to get real data — never guess or make up information.
Respond in Spanish. Be concise."""
```

### 2. Tools

Ask the user: **"What data sources or actions should the chat have access to?"**

Common patterns:
- **Database queries** → SQL tool
- **API calls** → HTTP request tool
- **File operations** → read/write file tool
- **Calculations** → math/calculator tool
- **External services** → service-specific tools

Define tools in OpenAI function-calling format:

```python
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_inventory",
            "description": "Search products in the inventory database",
            "parameters": {
                "type": "object",
                "properties": {
                    "search": {"type": "string", "description": "Product name or SKU"},
                    "category": {"type": "string", "description": "Product category (optional)"},
                },
                "required": ["search"],
            },
        },
    },
]
```

### 3. Tool Executor

Create a class that actually runs each tool when the LLM requests it:

```python
from ai_wem import ToolExecutor, ToolCall, ToolResult
import json

class MyExecutor(ToolExecutor):
    async def execute(self, tc: ToolCall) -> ToolResult:
        if tc.name == "query_inventory":
            # Connect to your real data source here
            results = db.query("SELECT * FROM products WHERE name LIKE ?", tc.arguments["search"])
            return ToolResult(tc.id, tc.name, json.dumps(results))

        return ToolResult(tc.id, tc.name, f"Unknown tool: {tc.name}", is_error=True)
```

### 4. Config (models + API keys)

Ask the user: **"What LLM providers do you have access to?"**

Available providers and when to recommend each:

| Provider | Best for | Cost | Setup |
|----------|----------|------|-------|
| Gemini (Google) | Most use cases, free tier available | Free (Flash) / $2-12/M (Pro) | `GEMINI_API_KEY` env var |
| Ollama (local) | Privacy, no API costs, testing | Free | Install Ollama + pull model |
| Claude (Anthropic) | Complex reasoning, Master evaluator | $15-75/M (Opus) | `ANTHROPIC_API_KEY` env var |
| OpenAI | Familiar API, GPT models | $2-60/M | `OPENAI_API_KEY` env var |

**Default recommendation** (best cost/quality balance):
```python
from ai_wem import WEMConfig

config = WEMConfig(
    # Expert: Gemini Pro (good quality, affordable)
    expert_api_key=os.environ["GEMINI_API_KEY"],
    # Worker and Master: leave empty to start simple
)
```

**Budget-conscious** (100% free):
```python
config = WEMConfig(
    expert_model="qwen3:8b",
    expert_base_url="http://localhost:11434",
)
# Then: engine.set_expert_provider(HttpOllamaProvider("http://localhost:11434", "qwen3:8b"))
```

**Maximum quality** (Expert + Master):
```python
config = WEMConfig(
    expert_api_key=os.environ["GEMINI_API_KEY"],
    master_api_key=os.environ["ANTHROPIC_API_KEY"],
)
```

## Putting it all together

Once you have the 4 pieces, generate the complete app:

```python
import os
import asyncio
from ai_wem import WEMEngine, WEMConfig, ToolExecutor, ToolCall, ToolResult

# 1. System prompt
SYSTEM_PROMPT = "..."

# 2. Tools
TOOLS = [...]

# 3. Executor
class MyExecutor(ToolExecutor):
    async def execute(self, tc: ToolCall) -> ToolResult:
        ...

# 4. Config
config = WEMConfig(expert_api_key=os.environ["GEMINI_API_KEY"])

# Create engine — everything configured once
engine = WEMEngine(
    config,
    tools=TOOLS,
    executor=MyExecutor(),
    system_prompt=SYSTEM_PROMPT,
)

# Chat loop
async def main():
    while True:
        user_input = input("You: ").strip()
        if not user_input or user_input == "/quit":
            break
        result = await engine.send_message(user_input)
        print(f"Bot: {result['reply']}")

asyncio.run(main())
```

## Step-by-step workflow

When a user asks you to help them build a chat, follow this order:

1. **Ask what the chat is for** → write system prompt
2. **Ask what data/actions it needs** → define tools + executor
3. **Ask what LLM provider they have** → create config
4. **Generate the complete app** → working code they can run
5. **Test together** → run it, fix issues, iterate

## Important rules

- Always use `ToolResult` to return data from tools — content must be a string (use `json.dumps` for structured data)
- Tool executor must be async (`async def execute`)
- Never hardcode API keys — always use environment variables or config files
- Start simple: Expert only, add Worker/Master later if needed
- If the user doesn't know what model to use, default to Gemini Pro (good balance of cost and quality)

## Worker fast path (optional, add later)

Only recommend Worker tier if the user has repetitive, predictable questions:

```python
CLASSIFY_PROMPT = (
    "Classify the intent into ONE category. "
    "Respond ONLY with JSON: {\"intent\": \"category\", \"params\": {}}\n"
    "Categories:\n"
    "- inventory: product search (params: search=keyword)\n"
    "- order_status: order lookup (params: order_id=ID)\n"
    "- none: doesn't fit\n\n"
    "Question: {user_text}"
)

INTENT_MAP = {
    "inventory": ("query_inventory", {"search": "all"}),
    "order_status": ("get_order", {"order_id": ""}),
}

engine = WEMEngine(
    config,
    tools=TOOLS,
    executor=MyExecutor(),
    system_prompt=SYSTEM_PROMPT,
    classify_prompt=CLASSIFY_PROMPT,
    intent_map=INTENT_MAP,
)
```

## API reference

### WEMEngine

```python
engine = WEMEngine(
    config: WEMConfig,            # Models, API keys, costs
    tools: list = [],             # OpenAI function-calling format
    executor: ToolExecutor = None,# Runs tool calls
    system_prompt: str = "",      # Domain instructions
    classify_prompt: str = "",    # Worker classification template (optional)
    intent_map: dict = None,      # Worker intent → tool mapping (optional)
    worker_hook: callable = None, # Custom Worker logic (optional)
)

# Simple usage
result = await engine.send_message("user question")

# Override per-call (for dynamic context)
result = await engine.send_message("user question",
    tools=different_tools,
    executor=different_executor,
    system_prompt=different_prompt,
)

# Sync wrapper (for non-async apps)
result = engine.send_message_sync("user question")

# Result format
result = {
    "reply": "Natural language response",
    "tokens": 1234,
    "cost": 0.0042,
    "tier": "expert",  # "worker", "expert", or "master"
}
```

### Providers

```python
from ai_wem import HttpApiProvider, HttpOllamaProvider

# Cloud (Gemini, Claude, OpenAI — auto-detected by URL)
provider = HttpApiProvider(base_url, api_key, model)

# Local (Ollama — auto-detects native tool support)
provider = HttpOllamaProvider(base_url, model, api_key="")

# Swap Expert provider
engine.set_expert_provider(provider)
```
