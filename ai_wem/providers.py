"""LLM provider abstraction — HTTP API (Claude/OpenAI/Gemini) and HTTP Ollama."""

import json
import re
import uuid
import logging

log = logging.getLogger("ai_wem")


class LLMProvider:
    """Base class for LLM backends.

    Returns:
        {"content": str, "tool_calls": [{"id", "name", "arguments"}], "usage": {"input", "output"}}
    """

    def chat(self, messages: list, tools: list) -> dict:
        raise NotImplementedError


class HttpApiProvider(LLMProvider):
    """OpenAI-compatible HTTP API provider (also handles Anthropic API)."""

    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def chat(self, messages, tools):
        if "anthropic.com" in self.base_url:
            return self._chat_anthropic(messages, tools)
        return self._chat_openai(messages, tools)

    def _chat_openai(self, messages, tools):
        """Standard OpenAI-compatible chat completions."""
        import requests

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {"model": self.model, "messages": messages}
        if tools:
            body["tools"] = tools
        # Avoid doubling /v1/ if base_url already includes it
        if "/v1" in self.base_url:
            url = f"{self.base_url}/chat/completions"
        else:
            url = f"{self.base_url}/v1/chat/completions"
        resp = requests.post(url, headers=headers, json=body, timeout=300)
        if not resp.ok:
            log.error("API %d: %s", resp.status_code, resp.text[:500])
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]["message"]

        result = {"content": choice.get("content", "") or ""}
        if choice.get("tool_calls"):
            result["tool_calls"] = []
            for tc in choice["tool_calls"]:
                result["tool_calls"].append({
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "arguments": json.loads(tc["function"]["arguments"]),
                })
        usage = data.get("usage", {})
        result["usage"] = {
            "input": usage.get("prompt_tokens", 0),
            "output": usage.get("completion_tokens", 0),
        }
        return result

    def _chat_anthropic(self, messages, tools):
        """Anthropic Messages API."""
        import requests

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        anthropic_tools = [
            {
                "name": t["function"]["name"],
                "description": t["function"]["description"],
                "input_schema": t["function"]["parameters"],
            }
            for t in tools
        ] if tools else []

        system = ""
        api_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            elif msg["role"] == "tool":
                api_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg["tool_call_id"],
                        "content": msg["content"],
                    }],
                })
            elif msg["role"] == "assistant" and msg.get("tool_calls"):
                content_blocks = []
                if msg.get("content"):
                    content_blocks.append({"type": "text", "text": msg["content"]})
                for tc in msg["tool_calls"]:
                    args = tc["function"]["arguments"]
                    if isinstance(args, str):
                        args = json.loads(args)
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "input": args,
                    })
                api_messages.append({"role": "assistant", "content": content_blocks})
            else:
                api_messages.append(msg)

        body = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": api_messages,
        }
        if system:
            body["system"] = system
        if anthropic_tools:
            body["tools"] = anthropic_tools

        resp = requests.post(
            f"{self.base_url}/v1/messages",
            headers=headers, json=body, timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()

        content = ""
        tool_calls = []
        for block in data.get("content", []):
            if block["type"] == "text":
                content += block["text"]
            elif block["type"] == "tool_use":
                tool_calls.append({
                    "id": block["id"],
                    "name": block["name"],
                    "arguments": block["input"],
                })

        result = {"content": content}
        if tool_calls:
            result["tool_calls"] = tool_calls
        usage = data.get("usage", {})
        result["usage"] = {
            "input": usage.get("input_tokens", 0),
            "output": usage.get("output_tokens", 0),
        }
        return result


# ── Ollama HTTP Provider ──────────────────────────────────


class _PromptToolsMixin:
    """Simulates tool calling via <tool_call> tags in the prompt.

    Used for LLM models that don't support native tool calling.
    Injects tool definitions into the system prompt and parses
    <tool_call>{"name": ..., "arguments": {...}}</tool_call> from responses.
    """

    def _format_tools(self, tools):
        """Format tools as text for injection into the system prompt."""
        lines = ["Available tools (call with <tool_call>{...}</tool_call>):"]
        for t in tools:
            fn = t["function"]
            params = fn.get("parameters", {}).get("properties", {})
            required = fn.get("parameters", {}).get("required", [])
            param_parts = []
            for pname, pdef in params.items():
                req = " (required)" if pname in required else ""
                param_parts.append(f"    {pname}: {pdef.get('type', 'any')} — "
                                   f"{pdef.get('description', '')}{req}")
            lines.append(f"\n{fn['name']}: {fn.get('description', '')}")
            if param_parts:
                lines.append("  Parameters:")
                lines.extend(param_parts)
        lines.append(
            '\nTo call a tool, respond with: <tool_call>{"name": "tool_name", '
            '"arguments": {"param": "value"}}</tool_call>'
        )
        return "\n".join(lines)

    def _build_chat_messages(self, messages, tool_text):
        """Inject tool definitions into the system message."""
        enhanced = []
        system_injected = False
        for msg in messages:
            if msg["role"] == "system" and not system_injected:
                enhanced.append({
                    "role": "system",
                    "content": msg["content"] + "\n\n" + tool_text,
                })
                system_injected = True
            elif msg["role"] == "tool":
                enhanced.append({
                    "role": "user",
                    "content": f"[Tool result for {msg.get('tool_call_id', '?')}]:\n{msg['content']}",
                })
            elif msg["role"] == "assistant" and msg.get("tool_calls"):
                parts = [msg.get("content", "")]
                for tc in msg["tool_calls"]:
                    args = tc["function"]["arguments"]
                    if isinstance(args, str):
                        args = json.loads(args)
                    parts.append(
                        f'<tool_call>{json.dumps({"name": tc["function"]["name"], "arguments": args})}</tool_call>'
                    )
                enhanced.append({"role": "assistant", "content": "\n".join(parts)})
            else:
                enhanced.append(msg)
        if not system_injected and tool_text:
            enhanced.insert(0, {"role": "system", "content": tool_text})
        return enhanced

    def _extract_tool_calls(self, text):
        pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        matches = re.findall(pattern, text, re.DOTALL)
        calls = []
        for m in matches:
            try:
                data = json.loads(m)
                calls.append({
                    "id": f"tc_{uuid.uuid4().hex[:8]}",
                    "name": data["name"],
                    "arguments": data.get("arguments", {}),
                })
            except (json.JSONDecodeError, KeyError):
                continue
        return calls

    def _strip_tool_calls(self, text):
        return re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL).strip()

    def _parse_response(self, response_text):
        """Parse response text into {"content": ..., "tool_calls": [...]}."""
        tool_calls = self._extract_tool_calls(response_text)
        content = self._strip_tool_calls(response_text)
        result = {"content": content}
        if tool_calls:
            result["tool_calls"] = tool_calls
        return result


class HttpOllamaProvider(LLMProvider, _PromptToolsMixin):
    """HTTP Ollama provider — calls /api/chat directly with optional Bearer auth.

    Auto-detects native tool support: tries native tools first,
    falls back to prompt-based <tool_call> simulation.
    """

    def __init__(self, base_url: str, model: str, api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self._native_tools = None  # None=unknown, True/False after first call
        self._nothink = "qwen" in model.lower()

    def _inject_nothink(self, messages):
        """Prepend /nothink to system message for qwen3 models."""
        if not self._nothink:
            return messages
        result = []
        for msg in messages:
            if msg["role"] == "system" and not msg["content"].startswith("/nothink"):
                result.append({**msg, "content": "/nothink\n" + msg["content"]})
            else:
                result.append(msg)
        return result

    def chat(self, messages, tools):
        import requests

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Try native tools first (if not already known to be unsupported)
        if self._native_tools is not False and tools:
            try:
                return self._chat_native(messages, tools, headers)
            except Exception as ex:
                err = str(ex)
                if "does not support tools" in err or "400" in err:
                    log.info("Model %s: no native tools, using prompt-based", self.model)
                    self._native_tools = False
                else:
                    raise

        # Prompt-based fallback
        return self._chat_prompt(messages, tools, headers)

    def _chat_native(self, messages, tools, headers):
        """Use Ollama native tool calling (/api/chat with tools param)."""
        import requests

        ollama_messages = []
        for msg in self._inject_nothink(messages):
            if msg["role"] == "tool":
                ollama_messages.append({
                    "role": "tool",
                    "content": msg["content"],
                })
            elif msg["role"] == "assistant" and msg.get("tool_calls"):
                tc_list = []
                for tc in msg["tool_calls"]:
                    args = tc["function"]["arguments"]
                    if isinstance(args, str):
                        args = json.loads(args)
                    tc_list.append({
                        "function": {"name": tc["function"]["name"], "arguments": args},
                    })
                ollama_messages.append({
                    "role": "assistant",
                    "content": msg.get("content", ""),
                    "tool_calls": tc_list,
                })
            else:
                ollama_messages.append(msg)

        body = {
            "model": self.model,
            "messages": ollama_messages,
            "tools": tools,
            "stream": False,
        }

        resp = requests.post(
            f"{self.base_url}/api/chat",
            headers=headers, json=body, timeout=180,
        )
        if resp.status_code == 400:
            data = resp.json()
            err_msg = data.get("error", "")
            if "does not support tools" in err_msg:
                raise ValueError(err_msg)
        resp.raise_for_status()
        data = resp.json()

        self._native_tools = True
        msg = data.get("message", {})
        content = msg.get("content", "")

        tool_calls = []
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            tool_calls.append({
                "id": tc.get("id", f"tc_{uuid.uuid4().hex[:8]}"),
                "name": fn.get("name", ""),
                "arguments": fn.get("arguments", {}),
            })

        result = {"content": content}
        if tool_calls:
            result["tool_calls"] = tool_calls
        return result

    def _chat_prompt(self, messages, tools, headers):
        """Prompt-based tool simulation for models without native support."""
        import requests

        tool_text = self._format_tools(tools) if tools else ""
        chat_messages = self._build_chat_messages(self._inject_nothink(messages), tool_text)

        body = {
            "model": self.model,
            "messages": chat_messages,
            "stream": False,
        }

        resp = requests.post(
            f"{self.base_url}/api/chat",
            headers=headers, json=body, timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()

        response_text = data.get("message", {}).get("content", "")
        return self._parse_response(response_text)
