#!/usr/bin/env python3
"""
agent.py — Fully local autonomous research agent via Ollama.

Drives the experiment loop from program.md using a local Ollama model —
no API keys, no internet required.

Usage:
  # Default model (qwen2.5-coder:14b)
  python agent.py

  # Pick a different model
  python agent.py --model llama3.1:70b

  # Custom Ollama host
  python agent.py --host http://192.168.1.5:11434

Recommended models (tool/function-calling support required):
  qwen2.5-coder:14b   good balance of speed and quality (default)
  qwen2.5-coder:32b   higher quality, needs ~20 GB VRAM
  llama3.1:8b         fast, lower quality
  llama3.1:70b        high quality, needs ~40 GB VRAM
  qwen2.5:72b         strong general-purpose choice

Environment variables:
  OLLAMA_HOST    Ollama server URL (default: http://localhost:11434)
  OLLAMA_MODEL   Default model name
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).parent

# ---------------------------------------------------------------------------
# Tools exposed to the agent
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the full contents of a file (relative to the repo root).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file (relative to repo root or absolute)."
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write (overwrite) a file with the given content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file (relative to repo root or absolute)."
                    },
                    "content": {
                        "type": "string",
                        "description": "Full content to write."
                    }
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": (
                "Run a bash command in the repository directory and return combined stdout+stderr. "
                "Default timeout is 700 s to comfortably cover a 5-minute training run plus overhead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute."
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 700)."
                    }
                },
                "required": ["command"]
            }
        }
    }
]


def execute_tool(name: str, args: dict) -> str:
    if name == "read_file":
        path = Path(args["path"])
        if not path.is_absolute():
            path = REPO_ROOT / path
        try:
            return path.read_text(encoding="utf-8")
        except Exception as exc:
            return f"Error reading {path}: {exc}"

    elif name == "write_file":
        path = Path(args["path"])
        if not path.is_absolute():
            path = REPO_ROOT / path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(args["content"], encoding="utf-8")
            return f"Wrote {len(args['content'])} bytes to {path}"
        except Exception as exc:
            return f"Error writing {path}: {exc}"

    elif name == "run_bash":
        command = args["command"]
        timeout = int(args.get("timeout", 700))
        print(f"  $ {command}", flush=True)
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(REPO_ROOT),
            )
            output = result.stdout + result.stderr
            # Truncate very long outputs to avoid flooding the context window
            if len(output) > 8000:
                output = output[:4000] + "\n...[output truncated]...\n" + output[-4000:]
            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout}s"
        except Exception as exc:
            return f"Error: {exc}"

    return f"Unknown tool: {name}"


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_agent(model: str, host: str, initial_prompt: str) -> None:
    # Verify Ollama is reachable before starting
    try:
        requests.get(f"{host}/api/tags", timeout=5).raise_for_status()
    except Exception as exc:
        sys.exit(f"Cannot reach Ollama at {host}: {exc}\nIs Ollama running?  Try: ollama serve")

    system_prompt = (REPO_ROOT / "program.md").read_text(encoding="utf-8")
    messages: list = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_prompt},
    ]

    print(f"[agent] Ollama @ {host}  model: {model}")
    print("[agent] Research loop started. Press Ctrl+C to stop.\n")
    step = 0

    while True:
        step += 1
        print(f"\n[agent] Step {step} — calling {model} …", flush=True)

        try:
            resp = requests.post(
                f"{host}/api/chat",
                json={"model": model, "messages": messages, "tools": TOOLS, "stream": False},
                timeout=120,
            )
            resp.raise_for_status()
        except KeyboardInterrupt:
            print("\n[agent] Stopped by user.")
            break

        msg = resp.json()["message"]
        messages.append(msg)  # append assistant turn verbatim

        if msg.get("content", "").strip():
            print(f"\n[assistant] {msg['content']}")

        tool_calls = msg.get("tool_calls") or []

        # Some models (e.g. qwen2.5-coder via Ollama) emit tool calls as JSON
        # in the content field instead of the tool_calls field.  Detect and
        # normalise that so the rest of the loop works unchanged.
        if not tool_calls:
            content = msg.get("content", "").strip()
            if content.startswith("{"):
                try:
                    parsed = json.loads(content)
                    if "name" in parsed and "arguments" in parsed:
                        tool_calls = [{"function": parsed}]
                        # Don't re-print it as assistant text — it's a tool call
                        pass
                except json.JSONDecodeError:
                    pass

        if not tool_calls:
            # Model paused — let the user nudge it or exit
            print("[agent] No tool calls — agent paused.")
            try:
                user_input = input("[you] > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[agent] Stopped.")
                break
            if user_input.lower() in ("exit", "quit", "q", ""):
                break
            messages.append({"role": "user", "content": user_input})
            continue

        for tc in tool_calls:
            fn = tc["function"]
            name = fn["name"]
            args = fn["arguments"]
            if isinstance(args, str):
                args = json.loads(args)

            args_preview = json.dumps(args)
            print(f"\n[tool:{name}] {args_preview[:120]}{'…' if len(args_preview) > 120 else ''}")
            result = execute_tool(name, args)
            preview = result[:300].replace("\n", " ")
            print(f"[result] {preview}{'…' if len(result) > 300 else ''}")

            messages.append({"role": "tool", "content": result})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="agent.py",
        description="Fully local autonomous research agent using Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:14b"),
        help="Ollama model to use (env: OLLAMA_MODEL, default: qwen2.5-coder:14b)",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        help="Ollama server URL (env: OLLAMA_HOST, default: http://localhost:11434)",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Hi, have a look at program.md and let's kick off a new experiment! "
            "Let's do the setup first."
        ),
        help="Initial prompt sent to the agent",
    )
    args = parser.parse_args()
    run_agent(model=args.model, host=args.host.rstrip("/"), initial_prompt=args.prompt)


if __name__ == "__main__":
    main()
