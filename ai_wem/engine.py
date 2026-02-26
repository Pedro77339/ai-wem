"""WEM Engine — Worker/Expert/Master AI chat orchestration.

Async-first with sync wrapper. Apps configure at init:
- system_prompt (str) — domain context
- tools (list) — OpenAI function-calling defs
- executor (ToolExecutor) — runs tool calls
- optionally: classify_prompt + intent_map for Worker fast path
- optionally: worker_hook for custom Worker logic

Then just call: await engine.send_message("user text")
"""

import asyncio
import json
import time
import uuid
import logging
from typing import Optional, List, Dict, Tuple, Callable

from .models import ChatMessage, MessageRole, ToolCall
from .config import WEMConfig
from .executor import ToolExecutor
from .providers import LLMProvider, HttpApiProvider

log = logging.getLogger("ai_wem")


class WEMEngine:
    """Orchestrates AI chat with Worker/Expert/Master tiers.

    - Worker (fast provider): classify common intents, handle with single tool call
    - Expert (main provider): full tool-calling loop
    - Master (evaluator): evaluate Expert response, redo if needed

    Usage:
        engine = WEMEngine(
            config=WEMConfig(expert_api_key="..."),
            tools=MY_TOOLS,
            executor=MyExecutor(),
            system_prompt="You are a helpful assistant...",
        )
        result = await engine.send_message("hello")
    """

    def __init__(
        self,
        config: WEMConfig,
        tools: Optional[list] = None,
        executor: Optional[ToolExecutor] = None,
        system_prompt: str = "",
        classify_prompt: str = "",
        intent_map: Optional[Dict[str, Tuple[str, dict]]] = None,
        worker_hook: Optional[Callable] = None,
    ):
        self.config = config
        self.messages: List[ChatMessage] = []
        self._total_input = 0
        self._total_output = 0
        self._master_input = 0
        self._master_output = 0

        # Domain config (set once, used every message)
        self.tools: list = tools or []
        self.executor: Optional[ToolExecutor] = executor
        self.system_prompt: str = system_prompt
        self.classify_prompt: str = classify_prompt
        self.intent_map: Optional[Dict[str, Tuple[str, dict]]] = intent_map
        self.worker_hook: Optional[Callable] = worker_hook

        # Expert provider (required)
        self.expert_provider: LLMProvider = HttpApiProvider(
            config.expert_base_url,
            config.expert_api_key,
            config.expert_model,
        )

        # Worker provider (optional — skip if no fast_api_key)
        self._fast_provider: Optional[LLMProvider] = None
        if config.fast_api_key:
            self._fast_provider = HttpApiProvider(
                config.fast_base_url or config.expert_base_url,
                config.fast_api_key,
                config.fast_model or config.expert_model,
            )

        # Master provider (optional — skip if no master_api_key)
        self._master_provider: Optional[LLMProvider] = None
        if config.master_api_key:
            self._master_provider = HttpApiProvider(
                config.master_base_url or "https://api.anthropic.com",
                config.master_api_key,
                config.master_model or "claude-opus-4-6",
            )

        # Optional callbacks
        self.on_status: Optional[Callable[[str], None]] = None

    def set_expert_provider(self, provider: LLMProvider):
        """Replace the expert provider (e.g., with HttpOllamaProvider)."""
        self.expert_provider = provider

    def clear(self):
        """Clear conversation history."""
        self.messages.clear()

    # ── Main entry point ──────────────────────────────────

    async def send_message(self, user_text: str, **overrides) -> dict:
        """Send user message through Worker/Expert/Master pipeline.

        Uses tools/executor/system_prompt from __init__ by default.
        Pass keyword args to override per-call (e.g., dynamic tools based on user role).

        Overridable kwargs:
            system_prompt, tools, executor, classify_prompt, intent_map, worker_hook

        Returns:
            {"reply": str, "tokens": int, "cost": float, "tier": "worker"|"expert"|"master"}
        """
        # Resolve per-call values (override or instance default)
        system_prompt = overrides.get("system_prompt", self.system_prompt)
        tools = overrides.get("tools", self.tools)
        executor = overrides.get("executor", self.executor)
        classify_prompt = overrides.get("classify_prompt", self.classify_prompt)
        intent_map = overrides.get("intent_map", self.intent_map)
        worker_hook = overrides.get("worker_hook", self.worker_hook)

        if not executor:
            raise ValueError("No executor configured. Set executor at init or pass as override.")

        t0 = time.time()
        self._total_input = 0
        self._total_output = 0
        self._master_input = 0
        self._master_output = 0

        self.messages.append(ChatMessage(role=MessageRole.USER, content=user_text))

        response = None
        tier = "expert"

        # 1) Worker fast path
        if self._fast_provider:
            if worker_hook:
                response = await worker_hook(user_text, executor)
            elif classify_prompt and intent_map:
                response = await self._try_worker(
                    user_text, executor, classify_prompt, intent_map
                )
            if response:
                tier = "worker"

        # 2) Expert tool loop
        api_messages = None
        if not response:
            api_messages = [{"role": "system", "content": system_prompt}]
            for msg in self.messages:
                if msg.role in (MessageRole.USER, MessageRole.ASSISTANT):
                    api_messages.append({"role": msg.role.value, "content": msg.content})
            response = await self._run_tool_loop(
                self.expert_provider, api_messages, executor, tools
            )
            tier = "expert"

        # 3) Master evaluation (only after Expert, if configured)
        if tier == "expert" and self._master_provider and response:
            verdict = await self._master_evaluate(user_text, response, api_messages)
            if verdict.get("verdict") == "redo":
                reason = verdict.get("reason", "?")
                log.info("Master: redo — %s", reason)
                master_messages = [{"role": "system", "content": system_prompt}]
                for msg in self.messages:
                    if msg.role in (MessageRole.USER, MessageRole.ASSISTANT):
                        master_messages.append({"role": msg.role.value, "content": msg.content})
                self._total_input = 0
                self._total_output = 0
                response = await self._run_tool_loop(
                    self._master_provider, master_messages, executor, tools
                )
                tier = "master"

        # Calculate cost
        elapsed = time.time() - t0
        tokens = self._total_input + self._total_output + self._master_input + self._master_output

        pin, pout = self.config.cost_expert if tier == "expert" else self.config.cost_fast
        cost = (self._total_input * pin + self._total_output * pout) / 1_000_000
        if self._master_input + self._master_output > 0:
            mpin, mpout = self.config.cost_master
            cost += (self._master_input * mpin + self._master_output * mpout) / 1_000_000

        if tokens > 0:
            response += f"\n\n_(tokens: {tokens:,}, ${cost:.4f}, {elapsed:.0f}s)_"

        self.messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=response))

        return {"reply": response, "tokens": tokens, "cost": cost, "tier": tier}

    def send_message_sync(self, user_text: str, **overrides) -> dict:
        """Synchronous wrapper for send_message. For use in sync apps (e.g., Flet UI)."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self.send_message(user_text, **overrides))
                return future.result()
        else:
            return asyncio.run(self.send_message(user_text, **overrides))

    # ── Worker: classify intent → single tool call ────────

    async def _try_worker(
        self,
        user_text: str,
        executor: ToolExecutor,
        classify_prompt: str,
        intent_map: Dict[str, Tuple[str, dict]],
    ) -> Optional[str]:
        """Classify with fast provider, execute single tool, format response."""
        prompt = classify_prompt.format(user_text=user_text)

        try:
            self._notify("Clasificando...")
            result = await asyncio.to_thread(
                self._fast_provider.chat,
                [{"role": "user", "content": prompt}],
                [],
            )
            usage = result.get("usage", {})
            self._total_input += usage.get("input", 0)
            self._total_output += usage.get("output", 0)

            content = result.get("content", "").strip()
            # Parse JSON from response
            if "{" not in content:
                return None
            json_str = content[content.index("{"):content.rindex("}") + 1]
            data = json.loads(json_str)

            intent = data.get("intent", "none")
            if intent == "none" or intent not in intent_map:
                return None

            tool_name, default_params = intent_map[intent]
            params = {**default_params, **data.get("params", {})}

            # Execute the tool directly
            self._notify(f"Ejecutando: {tool_name}")
            tc = ToolCall(id=f"w_{uuid.uuid4().hex[:6]}", name=tool_name, arguments=params)
            tool_result = await executor.execute(tc)
            if tool_result.is_error:
                return None  # Fall back to Expert

            # Format result with fast provider
            self._notify("Formateando...")
            format_prompt = (
                f"Pregunta: {user_text}\n\n"
                f"Datos:\n{tool_result.content}\n\n"
                "Responde al usuario de forma clara y concisa. "
                "Presenta los datos de forma legible. Usa markdown."
            )
            format_result = await asyncio.to_thread(
                self._fast_provider.chat,
                [{"role": "user", "content": format_prompt}],
                [],
            )
            usage = format_result.get("usage", {})
            self._total_input += usage.get("input", 0)
            self._total_output += usage.get("output", 0)

            formatted = format_result.get("content", "").strip()
            return formatted or tool_result.content

        except Exception as e:
            log.debug("Worker failed: %s", e)
            return None

    # ── Expert: full tool-calling loop ────────────────────

    async def _run_tool_loop(
        self,
        provider: LLMProvider,
        api_messages: list,
        executor: ToolExecutor,
        tools: list,
    ) -> str:
        """Run LLM tool loop until final text response or max iterations."""
        max_iterations = self.config.max_iterations
        max_chars = self.config.max_result_chars

        for iteration in range(max_iterations):
            self._notify("Pensando...")
            log.info("Iteration %d — calling LLM...", iteration)

            # Call LLM (sync provider, run in thread)
            result = await asyncio.to_thread(provider.chat, api_messages, tools)

            usage = result.get("usage", {})
            self._total_input += usage.get("input", 0)
            self._total_output += usage.get("output", 0)

            content = result.get("content", "")
            tool_calls = result.get("tool_calls", [])

            if not tool_calls:
                return content or "No pude generar una respuesta."

            # Append assistant message with tool calls
            assistant_msg = {"role": "assistant", "content": content}
            assistant_msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["arguments"])
                            if isinstance(tc["arguments"], dict)
                            else tc["arguments"],
                    },
                }
                for tc in tool_calls
            ]
            api_messages.append(assistant_msg)

            # Execute each tool call
            for tc_data in tool_calls:
                tc = ToolCall(
                    id=tc_data["id"],
                    name=tc_data["name"],
                    arguments=tc_data["arguments"],
                )
                log.info("Tool: %s(%s)", tc.name, tc.arguments)
                self._notify(f"Ejecutando: {tc.name}")

                tool_result = await executor.execute(tc)
                api_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result.content[:max_chars],
                })

            if iteration > 0:
                await asyncio.sleep(1)

        return "Se alcanzo el limite de iteraciones."

    # ── Master: evaluate Expert response ──────────────────

    async def _master_evaluate(
        self, user_text: str, expert_response: str, api_messages: list
    ) -> dict:
        """Ask Master to evaluate if Expert's response is correct."""
        tool_summary = []
        for msg in api_messages:
            if msg["role"] == "tool":
                tool_summary.append(
                    f"  [{msg.get('tool_call_id', '?')}]: {msg['content'][:200]}"
                )

        eval_prompt = (
            "Evalua si la respuesta del Expert es correcta y completa.\n\n"
            f"Pregunta del usuario: {user_text}\n\n"
            f"Respuesta del Expert:\n{expert_response[:2000]}\n\n"
        )
        if tool_summary:
            eval_prompt += f"Herramientas usadas ({len(tool_summary)}):\n"
            eval_prompt += "\n".join(tool_summary[:5])
            eval_prompt += "\n\n"

        eval_prompt += (
            'Responde SOLO con JSON: {"verdict": "ok"} o {"verdict": "redo", "reason": "..."}\n'
            "Usa 'redo' SOLO si la respuesta tiene errores factuales o no responde la pregunta."
        )

        try:
            result = await asyncio.to_thread(
                self._master_provider.chat,
                [{"role": "user", "content": eval_prompt}],
                [],
            )
            usage = result.get("usage", {})
            self._master_input += usage.get("input", 0)
            self._master_output += usage.get("output", 0)

            content = result.get("content", "").strip()
            if "{" in content:
                json_str = content[content.index("{"):content.rindex("}") + 1]
                return json.loads(json_str)
            return {"verdict": "ok"}
        except Exception as e:
            log.warning("Master eval failed: %s", e)
            return {"verdict": "ok"}

    # ── Helpers ───────────────────────────────────────────

    def _notify(self, status: str):
        if self.on_status:
            try:
                self.on_status(status)
            except Exception:
                pass
