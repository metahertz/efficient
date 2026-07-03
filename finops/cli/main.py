import os
import signal
import subprocess
from pathlib import Path

import click
import httpx


def _daemon_url() -> str:
    return os.getenv("FINOPS_DAEMON_URL", "http://localhost:7432")


PID_FILE = Path.home() / ".finops" / "daemon.pid"


@click.group()
def cli():
    """fullFinOps-AI — token optimization toolkit."""
    pass


@cli.command()
def start():
    """Start the finops daemon in the background."""
    if PID_FILE.exists():
        pid = int(PID_FILE.read_text().strip())
        try:
            os.kill(pid, 0)  # liveness check — raises if process is gone
            click.echo("Daemon already running. Run 'finops stop' first.")
            return
        except ProcessLookupError:
            click.echo(f"Removing stale PID file (process {pid} is gone).")
            PID_FILE.unlink(missing_ok=True)
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    port = os.getenv("FINOPS_PORT", "7432")
    proc = subprocess.Popen(
        ["uvicorn", "finops.daemon.app:app", "--host", "0.0.0.0", "--port", port],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    PID_FILE.write_text(str(proc.pid))
    click.echo(f"Daemon started (PID {proc.pid}) at {_daemon_url()}")


@cli.command()
def stop():
    """Stop the finops daemon."""
    if not PID_FILE.exists():
        click.echo("No daemon running.")
        return
    pid = int(PID_FILE.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        click.echo(f"Daemon stopped (PID {pid})")
    except ProcessLookupError:
        click.echo(f"Process {pid} not found — removing stale PID file.")
    PID_FILE.unlink(missing_ok=True)


@cli.command()
def status():
    """Show daemon health and module on/off state."""
    try:
        health = httpx.get(f"{_daemon_url()}/health", timeout=2.0).json()
        click.echo(f"● daemon running  version={health['version']}")
        modules = httpx.get(f"{_daemon_url()}/config", timeout=2.0).json().get("modules", {})
        for name, cfg in modules.items():
            state = "ON " if cfg.get("enabled") else "OFF"
            click.echo(f"  [{state}] {name}")
    except Exception:
        click.echo("○ daemon not running")


@cli.command()
def warmup():
    """Download and cache local models (embeddings + compressor)."""
    from finops.modules.embeddings import _get_model
    click.echo("Loading embedding model (voyageai/voyage-4-nano)...")
    _get_model()
    click.echo("  embedding model ready.")
    from finops.modules.context_compressor import _get_compressor
    click.echo("Loading compressor model (LLMLingua-2)...")
    _get_compressor()
    click.echo("  compressor model ready.")
    click.echo("Warmup complete.")


if __name__ == "__main__":
    cli()
