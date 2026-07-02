from pathlib import Path
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from finops.cli.main import cli


def test_status_when_daemon_down():
    runner = CliRunner()
    with patch("httpx.get", side_effect=ConnectionRefusedError("refused")):
        result = runner.invoke(cli, ["status"])
    assert result.exit_code == 0
    assert "daemon not running" in result.output


def test_start_writes_pid_file(tmp_path):
    pid_file = tmp_path / "daemon.pid"
    mock_proc = MagicMock()
    mock_proc.pid = 12345
    runner = CliRunner()
    with patch("finops.cli.main.PID_FILE", pid_file), \
         patch("subprocess.Popen", return_value=mock_proc):
        result = runner.invoke(cli, ["start"])
    assert result.exit_code == 0
    assert pid_file.read_text().strip() == "12345"
    assert "12345" in result.output


def test_start_blocks_if_pid_file_exists(tmp_path):
    pid_file = tmp_path / "daemon.pid"
    pid_file.write_text("99999")
    runner = CliRunner()
    with patch("finops.cli.main.PID_FILE", pid_file), \
         patch("os.kill", return_value=None):  # process appears alive
        result = runner.invoke(cli, ["start"])
    assert "already running" in result.output


def test_start_clears_stale_pid_file(tmp_path):
    pid_file = tmp_path / "daemon.pid"
    pid_file.write_text("99999")
    mock_proc = MagicMock()
    mock_proc.pid = 12345
    runner = CliRunner()
    with patch("finops.cli.main.PID_FILE", pid_file), \
         patch("os.kill", side_effect=[ProcessLookupError, None]), \
         patch("subprocess.Popen", return_value=mock_proc):
        result = runner.invoke(cli, ["start"])
    assert "stale" in result.output.lower()
    assert pid_file.read_text().strip() == "12345"


def test_stop_removes_pid_file(tmp_path):
    pid_file = tmp_path / "daemon.pid"
    pid_file.write_text("12345")
    runner = CliRunner()
    with patch("finops.cli.main.PID_FILE", pid_file), \
         patch("os.kill"):
        result = runner.invoke(cli, ["stop"])
    assert not pid_file.exists()
    assert "stopped" in result.output


def test_stop_when_no_daemon_running(tmp_path):
    pid_file = tmp_path / "daemon.pid"
    runner = CliRunner()
    with patch("finops.cli.main.PID_FILE", pid_file):
        result = runner.invoke(cli, ["stop"])
    assert "No daemon running" in result.output


def test_status_shows_module_state():
    health_resp = MagicMock()
    health_resp.json.return_value = {"status": "ok", "version": "0.1.0"}
    config_resp = MagicMock()
    config_resp.json.return_value = {
        "modules": {
            "semantic_cache": {"enabled": True},
            "agent_memory":   {"enabled": False},
        }
    }
    runner = CliRunner()
    with patch("httpx.get", side_effect=[health_resp, config_resp]):
        result = runner.invoke(cli, ["status"])
    assert "daemon running" in result.output
    assert "ON" in result.output
    assert "OFF" in result.output
