"""WEM Demo Chat — interactive CLI for testing models and the W/E/M pipeline.

Usage:
    python -m ai_wem.demo                          # uses config from env vars
    python -m ai_wem.demo --provider gemini         # Gemini Flash
    python -m ai_wem.demo --provider ollama --model qwen3:8b
    python -m ai_wem.demo --provider claude

Environment variables:
    GEMINI_API_KEY   — for Gemini provider
    ANTHROPIC_API_KEY — for Claude provider

Commands during chat:
    /clear      — clear conversation history
    /provider   — show current provider info
    /switch <p> — switch provider (gemini, claude, ollama)
    /model <m>  — change model name
    /tier       — show which tier handled the last message
    /quit       — exit
"""

import asyncio
import argparse
import os

from ai_wem import WEMEngine, WEMConfig, HttpApiProvider, HttpOllamaProvider
from ai_wem.demo.mock_tools import (
    MOCK_TOOLS, MockToolExecutor, CLASSIFY_PROMPT, INTENT_MAP,
)

SYSTEM_PROMPT = (
    "You are a demo assistant for testing the WEM engine.\n"
    "You have tools for: weather, products, orders, calculator, system info.\n"
    "Respond in the user's language. Be concise.\n"
    "ALWAYS use the available tools to respond — do not make up data.\n"
)

PROVIDERS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-3-pro-preview",
        "env_key": "GEMINI_API_KEY",
    },
    "claude": {
        "base_url": "https://api.anthropic.com",
        "model": "claude-opus-4-6",
        "env_key": "ANTHROPIC_API_KEY",
    },
    "ollama": {
        "base_url": "http://localhost:11434",
        "model": "qwen3:8b",
        "env_key": "",
    },
}


def build_config(provider_name: str, model: str = "") -> WEMConfig:
    prov = PROVIDERS.get(provider_name, PROVIDERS["gemini"])
    api_key = os.environ.get(prov["env_key"], "") if prov["env_key"] else ""
    m = model or prov["model"]

    # For Gemini, use same key for fast (worker) tier
    fast_key = api_key if provider_name == "gemini" else os.environ.get("GEMINI_API_KEY", "")
    fast_url = PROVIDERS["gemini"]["base_url"] if fast_key else ""
    fast_model = "gemini-3-flash-preview" if fast_key else ""

    return WEMConfig(
        expert_model=m,
        expert_base_url=prov["base_url"],
        expert_api_key=api_key,
        fast_model=fast_model,
        fast_base_url=fast_url,
        fast_api_key=fast_key,
    )


def create_engine(provider_name: str, model: str = "") -> tuple:
    """Create engine for the given provider. Returns (engine, provider_name, model)."""
    config = build_config(provider_name, model)
    engine = WEMEngine(config)

    # For Ollama, swap the expert provider
    if provider_name == "ollama":
        prov = PROVIDERS["ollama"]
        m = model or prov["model"]
        engine.set_expert_provider(
            HttpOllamaProvider(prov["base_url"], m)
        )

    actual_model = model or PROVIDERS.get(provider_name, {}).get("model", "?")
    return engine, provider_name, actual_model


async def main():
    parser = argparse.ArgumentParser(description="WEM Demo Chat")
    parser.add_argument("--provider", "-p", default="gemini",
                        choices=["gemini", "claude", "ollama"],
                        help="LLM provider (default: gemini)")
    parser.add_argument("--model", "-m", default="",
                        help="Model name override")
    args = parser.parse_args()

    engine, prov_name, model_name = create_engine(args.provider, args.model)
    executor = MockToolExecutor()
    last_tier = ""

    print(f"\n=== WEM Demo Chat ===")
    print(f"Provider: {prov_name} | Model: {model_name}")
    print(f"Worker: {'ON' if engine._fast_provider else 'OFF'}")
    print(f"Master: {'ON' if engine._master_provider else 'OFF'}")
    print(f"Tools: {', '.join(t['function']['name'] for t in MOCK_TOOLS)}")
    print(f"\nCommands: /clear /provider /switch <p> /model <m> /tier /quit\n")

    def on_status(status):
        print(f"  [{status}]")

    engine.on_status = on_status

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        # Commands
        if user_input.startswith("/"):
            cmd = user_input.lower().split()
            if cmd[0] == "/quit":
                print("Bye!")
                break
            elif cmd[0] == "/clear":
                engine.clear()
                print("  [History cleared]")
            elif cmd[0] == "/provider":
                print(f"  Provider: {prov_name} | Model: {model_name}")
                print(f"  Worker: {'ON' if engine._fast_provider else 'OFF'}")
                print(f"  Master: {'ON' if engine._master_provider else 'OFF'}")
            elif cmd[0] == "/switch" and len(cmd) > 1:
                new_prov = cmd[1]
                if new_prov in PROVIDERS:
                    engine, prov_name, model_name = create_engine(new_prov, "")
                    engine.on_status = on_status
                    print(f"  [Switched to {prov_name} / {model_name}]")
                else:
                    print(f"  [Providers: {', '.join(PROVIDERS.keys())}]")
            elif cmd[0] == "/model" and len(cmd) > 1:
                new_model = cmd[1]
                engine, prov_name, model_name = create_engine(prov_name, new_model)
                engine.on_status = on_status
                print(f"  [Model: {model_name}]")
            elif cmd[0] == "/tier":
                print(f"  [Last tier: {last_tier or 'none'}]")
            else:
                print("  [Commands: /clear /provider /switch <p> /model <m> /tier /quit]")
            continue

        # Send message through WEM
        try:
            result = await engine.send_message(
                user_text=user_input,
                system_prompt=SYSTEM_PROMPT,
                tools=MOCK_TOOLS,
                executor=executor,
                classify_prompt=CLASSIFY_PROMPT,
                intent_map=INTENT_MAP,
            )
            last_tier = result.get("tier", "?")
            print(f"\nBot [{last_tier}]: {result['reply']}\n")
        except Exception as ex:
            print(f"\n  [ERROR: {ex}]\n")


if __name__ == "__main__":
    asyncio.run(main())
