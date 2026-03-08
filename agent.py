#!/usr/bin/env python3
"""
agent.py — Autonomous research agent runner for autoresearch.

Drives the experiment loop from program.md using a configurable LLM backend:
  - ollama:    Fully local models via Ollama (no API key needed) [DEFAULT]
  - anthropic: Claude models via the Anthropic Messages API
  - openai:    GPT models or any OpenAI-compatible API (LM Studio, vLLM, etc.)

The existing workflow (running Claude Code / Codex directly in this repo) is
unchanged — this script is an optional standalone driver for when you want a
fully local or API-driven loop without a separate IDE/CLI.

Usage:
  # Local Ollama — default model is qwen2.5-coder:14b
  python agent.py

  # Local Ollama, specific model
  python agent.py --provider ollama --model llama3.1:70b

  # Anthropic Claude
  python agent.py --provider anthropic --model claude-opus-4-6

  # OpenAI
  python agent.py --provider openai --model gpt-4o

  # OpenAI-compatible local API (LM Studio, vLLM, llama.cpp server …)
  python agent.py --provider openai \\
      --base-url http://localhost:1234/v1 \\
      --model my-model \\
      --api-key none

Configuration via environment variables (see .env.example):
  AGENT_PROVIDER      Provider to use (ollama | anthropic | openai)
  OLLAMA_HOST         Ollama server URL  (default: http://localhost:11434)
  OLLAMA_MODEL        Default Ollama model
  ANTHROPIC_API_KEY   Anthropic API key
  ANTHROPIC_MODEL     Default Anthropic model
  OPENAI_API_KEY      OpenAI API key
  OPENAI_BASE_URL     Base URL for OpenAI-compatible APIs
  OPENAI_MODEL        Default OpenAI model

Recommended Ollama models (tool-calling support required):
  qwen2.5-coder:14b   good balance of speed and quality (default)
  qwen2.5-coder:32b   higher quality, needs ~20 GB VRAM
  llama3.1:8b         fast, lower quality
  llama3.1:70b        high quality, needs ~40 GB VRAM
  qwen2.5:72b         strong general-purpose choice
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
# Tool definitions — OpenAI/Ollama function-calling schema
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
    """Execute a tool call and return the result as a plain string."""
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
            return f"Error executing command: {exc}"

    return f"Unknown tool: {name}"


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

class Provider:
    """Abstract base for LLM providers."""

    def chat(self, messages: list) -> dict:
        """
        Call the LLM and return a normalised response:
          {
            "content":    str,          # assistant text (may be empty)
            "tool_calls": [             # list of tool-call dicts
              {"id": str|None, "name": str, "arguments": dict}
            ],
            "_raw": any,               # provider-specific raw data for make_assistant_message
          }
        """
        raise NotImplementedError

    def make_assistant_message(self, response: dict) -> dict:
        """Build the assistant turn message to append to the history."""
        raise NotImplementedError

    def batch_tool_results(self, calls_and_results: list) -> list:
        """
        Given [(tool_call_dict, result_str), ...] return a list of message
        dicts to append.  Providers differ in how many messages they expect.
        """
        raise NotImplementedError


class OllamaProvider(Provider):
    """Local Ollama backend — no API key required."""

    def __init__(self, model: str, host: str):
        self.model = model
        self.host = host.rstrip("/")

    def chat(self, messages: list) -> dict:
        resp = requests.post(
            f"{self.host}/api/chat",
            json={"model": self.model, "messages": messages, "tools": TOOLS, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        msg = resp.json()["message"]

        tool_calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc["function"]
            args = fn["arguments"]
            if isinstance(args, str):
                args = json.loads(args)
            tool_calls.append({"id": None, "name": fn["name"], "arguments": args})

        return {"content": msg.get("content") or "", "tool_calls": tool_calls, "_raw": msg}

    def make_assistant_message(self, response: dict) -> dict:
        return {"role": "assistant", **response["_raw"]}

    def batch_tool_results(self, calls_and_results: list) -> list:
        # Ollama: one "tool" role message per call (no id needed)
        return [{"role": "tool", "content": result} for _, result in calls_and_results]


class AnthropicProvider(Provider):
    """Anthropic Claude via the Messages API (no SDK dependency)."""

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
        self._base = "https://api.anthropic.com/v1"

    @staticmethod
    def _convert_tools():
        return [
            {
                "name": t["function"]["name"],
                "description": t["function"]["description"],
                "input_schema": t["function"]["parameters"],
            }
            for t in TOOLS
        ]

    def chat(self, messages: list) -> dict:
        # Anthropic uses a top-level "system" param, not a system role message
        system = None
        history = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                history.append(m)

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body: dict = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": history,
            "tools": self._convert_tools(),
        }
        if system:
            body["system"] = system

        resp = requests.post(f"{self._base}/messages", json=body, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        text = ""
        tool_calls = []
        raw_content = data.get("content", [])
        for block in raw_content:
            if block["type"] == "text":
                text += block["text"]
            elif block["type"] == "tool_use":
                tool_calls.append({"id": block["id"], "name": block["name"], "arguments": block["input"]})

        return {"content": text, "tool_calls": tool_calls, "_raw": raw_content, "_stop": data.get("stop_reason")}

    def make_assistant_message(self, response: dict) -> dict:
        return {"role": "assistant", "content": response["_raw"]}

    def batch_tool_results(self, calls_and_results: list) -> list:
        # Anthropic batches all tool results into a single user message
        return [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": tc["id"], "content": result}
                    for tc, result in calls_and_results
                ],
            }
        ]


class OpenAIProvider(Provider):
    """OpenAI or any OpenAI-compatible endpoint (LM Studio, vLLM, llama.cpp …)."""

    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def chat(self, messages: list) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": "auto",
        }
        resp = requests.post(f"{self.base_url}/chat/completions", json=body, headers=headers, timeout=120)
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]

        tool_calls = []
        for tc in msg.get("tool_calls") or []:
            args = tc["function"]["arguments"]
            if isinstance(args, str):
                args = json.loads(args)
            tool_calls.append({"id": tc["id"], "name": tc["function"]["name"], "arguments": args})

        return {"content": msg.get("content") or "", "tool_calls": tool_calls, "_raw": msg}

    def make_assistant_message(self, response: dict) -> dict:
        return {"role": "assistant", **response["_raw"]}

    def batch_tool_results(self, calls_and_results: list) -> list:
        # OpenAI: one "tool" role message per call, referencing tool_call_id
        return [
            {"role": "tool", "tool_call_id": tc["id"], "content": result}
            for tc, result in calls_and_results
        ]


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def build_provider(args) -> Provider:
    provider_name = args.provider

    if provider_name == "ollama":
        host = args.ollama_host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        model = args.model or os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:14b")
        print(f"[agent] Provider: Ollama @ {host}  model: {model}")
        return OllamaProvider(model=model, host=host)

    if provider_name == "anthropic":
        api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            sys.exit("Error: set ANTHROPIC_API_KEY or pass --api-key for the anthropic provider.")
        model = args.model or os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-6")
        print(f"[agent] Provider: Anthropic  model: {model}")
        return AnthropicProvider(model=model, api_key=api_key)

    if provider_name == "openai":
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "none")
        base_url = args.base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = args.model or os.environ.get("OPENAI_MODEL", "gpt-4o")
        print(f"[agent] Provider: OpenAI-compatible @ {base_url}  model: {model}")
        return OpenAIProvider(model=model, api_key=api_key, base_url=base_url)

    sys.exit(f"Unknown provider: {provider_name!r}")


def run_agent_loop(provider: Provider, initial_prompt: str) -> None:
    system_prompt = (REPO_ROOT / "program.md").read_text(encoding="utf-8")

    messages: list = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_prompt},
    ]

    print("[agent] Research loop started. Press Ctrl+C to stop.\n")
    step = 0

    while True:
        step += 1
        print(f"\n[agent] Step {step} — calling LLM …", flush=True)

        try:
            response = provider.chat(messages)
        except requests.HTTPError as exc:
            print(f"[agent] HTTP error: {exc.response.status_code} {exc.response.text}", file=sys.stderr)
            raise
        except KeyboardInterrupt:
            print("\n[agent] Stopped by user.")
            break

        # Append the assistant turn to history
        messages.append(provider.make_assistant_message(response))

        # Print any free-text content
        if response["content"].strip():
            print(f"\n[assistant] {response['content']}")

        tool_calls = response["tool_calls"]

        if not tool_calls:
            # The model produced no tool calls — it's either done or waiting.
            # Allow the user to continue the conversation manually.
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

        # Execute every tool call, then batch-append results
        calls_and_results = []
        for tc in tool_calls:
            name = tc["name"]
            args_str = json.dumps(tc["arguments"])
            preview = args_str[:100] + ("…" if len(args_str) > 100 else "")
            print(f"\n[tool:{name}] {preview}")
            result = execute_tool(name, tc["arguments"])
            result_preview = result[:300].replace("\n", " ")
            print(f"[result] {result_preview}{'…' if len(result) > 300 else ''}")
            calls_and_results.append((tc, result))

        messages.extend(provider.batch_tool_results(calls_and_results))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="agent.py",
        description="Autonomous research agent runner (Ollama / Anthropic / OpenAI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--provider",
        default=os.environ.get("AGENT_PROVIDER", "ollama"),
        choices=["ollama", "anthropic", "openai"],
        help="LLM backend to use (default: ollama, env: AGENT_PROVIDER)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model name — provider defaults: "
            "ollama=qwen2.5-coder:14b, anthropic=claude-opus-4-6, openai=gpt-4o"
        ),
    )
    parser.add_argument(
        "--ollama-host",
        default=None,
        metavar="URL",
        help="Ollama server URL (default: $OLLAMA_HOST or http://localhost:11434)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        metavar="KEY",
        help="API key for cloud providers (env: ANTHROPIC_API_KEY / OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        metavar="URL",
        help="Base URL for OpenAI-compatible APIs (env: OPENAI_BASE_URL)",
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

    provider = build_provider(args)
    run_agent_loop(provider, args.prompt)


if __name__ == "__main__":
    main()
