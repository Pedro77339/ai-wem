"""Script loader and executor for task-specific Python scripts.

Scripts live in a configurable directory and must define:
  - DESCRIPTION: str — one-line description for the classifier index
  - PARAMS: dict — parameter definitions {name: {type, required, description}}
  - def run(api, **kwargs) -> dict — execute the task, return structured result
"""

import importlib.util
import json
import os
import logging

log = logging.getLogger("ai_wem")


class ScriptIndex:
    """Loads and indexes scripts from a directory of .py files."""

    def __init__(self, scripts_dir: str):
        self.scripts_dir = scripts_dir
        self._scripts = {}  # name -> {description, params, format, module}
        self.reload()

    def reload(self):
        """Scan scripts directory and import each .py script."""
        self._scripts.clear()
        if not os.path.isdir(self.scripts_dir):
            return
        for fname in sorted(os.listdir(self.scripts_dir)):
            if not fname.endswith(".py") or fname.startswith("_"):
                continue
            name = fname[:-3]
            path = os.path.join(self.scripts_dir, fname)
            try:
                mod = self._load_module(name, path)
                if not hasattr(mod, "run") or not callable(mod.run):
                    log.warning("Script %s: missing run() function, skipped", name)
                    continue
                self._scripts[name] = {
                    "description": getattr(mod, "DESCRIPTION", ""),
                    "params": getattr(mod, "PARAMS", {}),
                    "format": getattr(mod, "FORMAT", ""),
                    "module": mod,
                }
            except Exception as ex:
                log.error("Failed to load script %s: %s", name, ex)

    def _load_module(self, name, path):
        """Import a .py file as a module."""
        spec = importlib.util.spec_from_file_location(f"wem_script_{name}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def get_index_text(self) -> str:
        """Return formatted index for the classifier prompt."""
        if not self._scripts:
            return ""
        lines = []
        for name, info in self._scripts.items():
            params = info["params"]
            user_params = {k: v for k, v in params.items() if k != "mac"}
            if user_params:
                param_str = ", ".join(
                    f"{k} ({v.get('description', '')})"
                    for k, v in user_params.items()
                )
                lines.append(f"  - {name}: {info['description']} | params: {param_str}")
            else:
                lines.append(f"  - {name}: {info['description']}")
        return "\n".join(lines)

    def get_script_names(self) -> list:
        return list(self._scripts.keys())

    def has_scripts(self) -> bool:
        return bool(self._scripts)

    def run(self, name: str, api, **params) -> dict:
        """Execute a script's run() function. Returns dict result."""
        if name not in self._scripts:
            return {"error": f"Script '{name}' not found. Available: {', '.join(self._scripts)}"}
        try:
            result = self._scripts[name]["module"].run(api, **params)
            if not isinstance(result, dict):
                return {"error": f"Script returned {type(result).__name__}, expected dict"}
            return result
        except Exception as ex:
            log.error("Script %s error: %s", name, ex)
            return {"error": f"Script error: {ex}"}

    def format_result(self, name: str, result: dict) -> str:
        """Format a script result dict into human-readable text."""
        if "error" in result:
            return f"Error: {result['error']}"

        fmt = self._scripts.get(name, {}).get("format", "")
        if fmt:
            try:
                return self._apply_format(fmt, result)
            except Exception:
                pass

        return json.dumps(result, indent=2, ensure_ascii=False)

    def _apply_format(self, fmt: str, data: dict) -> str:
        """Apply a format template with simple {key} substitution."""
        text = fmt
        for key, val in data.items():
            if isinstance(val, dict):
                for k2, v2 in val.items():
                    text = text.replace(f"{{{key}.{k2}}}", str(v2))
            text = text.replace(f"{{{key}}}", str(val))
        return text

    def save_script(self, name: str, description: str, code: str) -> str:
        """Save a new script .py file and reload the index."""
        os.makedirs(self.scripts_dir, exist_ok=True)
        safe_name = "".join(c for c in name if c.isalnum() or c == "_")
        if not safe_name:
            return "Error: invalid script name"
        path = os.path.join(self.scripts_dir, f"{safe_name}.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        self.reload()
        if safe_name in self._scripts:
            return f"Script saved and loaded: {safe_name}.py ({len(code)} chars)"
        return f"Script saved but failed to load: {safe_name}.py"
