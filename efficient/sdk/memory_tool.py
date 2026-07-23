"""Anthropic memory-tool backend backed by the efficient daemon.

Drop-in for the Agent SDK's tool runner: memory operations persist in the
daemon (durable across processes, agent-scoped) and every write is embedded,
so memory-tool content is vector-searchable via /memory/retrieve and the
retrieve_memory MCP tool.

    from anthropic import Anthropic
    from efficient.sdk import EfficientMemoryTool

    client = Anthropic()
    runner = client.beta.messages.tool_runner(
        model="claude-sonnet-5", max_tokens=1024,
        tools=[EfficientMemoryTool(agent_id="support-bot")],
        messages=[...],
    )
"""
import os

import httpx
from anthropic.lib.tools import BetaAbstractMemoryTool


class EfficientMemoryTool(BetaAbstractMemoryTool):
    def __init__(self, daemon_url: str | None = None, agent_id: str = "default"):
        super().__init__()
        self._url = (daemon_url or os.getenv("EFFICIENT_DAEMON_URL",
                                             "http://localhost:7432")).rstrip("/")
        self._agent_id = agent_id

    def _headers(self) -> dict:
        token = os.getenv("EFFICIENT_API_TOKEN", "")
        return {"Authorization": f"Bearer {token}"} if token else {}

    def _post(self, command: str, **args) -> str:
        payload = {"agent_id": self._agent_id, "command": command,
                   **{k: v for k, v in args.items() if v is not None}}
        response = httpx.post(f"{self._url}/memory/tool", json=payload,
                              headers=self._headers(), timeout=30.0)
        response.raise_for_status()
        body = response.json()
        # tool errors are content for the model — return them as text
        return body["result"] if body.get("ok") else f"Error: {body.get('error')}"

    def view(self, command):
        return self._post("view", path=command.path,
                          view_range=getattr(command, "view_range", None))

    def create(self, command):
        return self._post("create", path=command.path, file_text=command.file_text)

    def str_replace(self, command):
        return self._post("str_replace", path=command.path,
                          old_str=command.old_str, new_str=command.new_str)

    def insert(self, command):
        return self._post("insert", path=command.path,
                          insert_line=command.insert_line,
                          insert_text=command.insert_text)

    def delete(self, command):
        return self._post("delete", path=command.path)

    def rename(self, command):
        return self._post("rename", old_path=command.old_path,
                          new_path=command.new_path)

    def clear_all_memory(self):
        return self._post("clear_all")
