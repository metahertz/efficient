import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent.parent
PLUGIN = REPO / "plugin"


def _load(path):
    with open(path) as f:
        return json.load(f)


def test_marketplace_manifest():
    m = _load(REPO / ".claude-plugin" / "marketplace.json")
    assert m["name"] == "efficient"
    entry = m["plugins"][0]
    assert entry["name"] == "efficient"
    assert (REPO / entry["source"]).is_dir()


def test_plugin_manifest():
    p = _load(PLUGIN / ".claude-plugin" / "plugin.json")
    assert p["name"] == "efficient"
    assert p["version"]


def _commands(node):
    if isinstance(node, dict):
        if node.get("type") == "command":
            yield node["command"]
        for v in node.values():
            yield from _commands(v)
    elif isinstance(node, list):
        for v in node:
            yield from _commands(v)


def test_hooks_reference_existing_executable_scripts():
    hooks = _load(PLUGIN / "hooks" / "hooks.json")
    commands = list(_commands(hooks))
    assert len(commands) == 4
    for cmd in commands:
        rel = re.sub(r'"?\$\{CLAUDE_PLUGIN_ROOT\}"?/', "", cmd)
        script = PLUGIN / rel
        assert script.is_file(), cmd
        assert os.access(script, os.X_OK), f"{script} not executable"


def test_monitor_manifest():
    monitors = _load(PLUGIN / "monitors" / "monitors.json")
    assert len(monitors) == 1
    mon = monitors[0]
    assert mon["name"] == "efficient-daemon"
    assert mon["description"]
    rel = re.sub(r'"?\$\{CLAUDE_PLUGIN_ROOT\}"?/', "", mon["command"])
    script = PLUGIN / rel
    assert script.is_file() and os.access(script, os.X_OK)


def test_mcp_manifest():
    mcp = _load(PLUGIN / ".mcp.json")
    server = mcp["mcpServers"]["efficient"]
    assert server["command"] == "docker"
    assert "${CLAUDE_PLUGIN_ROOT}/docker-compose.yml" in server["args"]
    assert "-T" in server["args"]


def test_hook_scripts_bash_syntax():
    for script in (PLUGIN / "scripts").glob("*.sh"):
        subprocess.run(["bash", "-n", str(script)], check=True)


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker not installed")
def test_compose_file_valid():
    subprocess.run(
        ["docker", "compose", "-f", str(PLUGIN / "docker-compose.yml"), "config", "-q"],
        check=True,
    )
