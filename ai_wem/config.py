"""Configuration for WEM AI Chat engine."""

from dataclasses import dataclass, field


@dataclass
class WEMConfig:
    """Configuration for the Worker/Expert/Master engine.

    Only expert_model, expert_base_url, and expert_api_key are required.
    Worker and Master tiers are optional — leave their API keys empty to skip.
    """

    GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
    ANTHROPIC_BASE = "https://api.anthropic.com"

    # Expert (main provider — full tool-calling loop)
    expert_model: str = "gemini-3-pro-preview"
    expert_base_url: str = GEMINI_BASE
    expert_api_key: str = ""

    # Worker (fast/free classifier) — optional
    fast_model: str = "gemini-3-flash-preview"
    fast_base_url: str = GEMINI_BASE
    fast_api_key: str = ""

    # Master (evaluator) — optional
    master_model: str = "claude-opus-4-6"
    master_base_url: str = ANTHROPIC_BASE
    master_api_key: str = ""

    # Limits
    max_iterations: int = 15
    max_result_chars: int = 10000

    # Cost per 1M tokens (input, output) by tier
    cost_expert: tuple = (2.0, 12.0)    # Gemini 3 Pro
    cost_fast: tuple = (0.0, 0.0)       # Gemini 3 Flash Preview (free)
    cost_master: tuple = (15.0, 75.0)   # Claude Opus 4.6
