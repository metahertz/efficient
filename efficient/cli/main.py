import os
import signal
import subprocess
from pathlib import Path

import click
import httpx


def _daemon_url() -> str:
    return os.getenv("EFFICIENT_DAEMON_URL", "http://localhost:7432")


PID_FILE = Path.home() / ".efficient" / "daemon.pid"


@click.group()
def cli():
    """efficient — token optimization toolkit."""
    pass


@cli.command()
def start():
    """Start the efficient daemon in the background."""
    if PID_FILE.exists():
        pid = int(PID_FILE.read_text().strip())
        try:
            os.kill(pid, 0)  # liveness check — raises if process is gone
            click.echo("Daemon already running. Run 'efficient stop' first.")
            return
        except ProcessLookupError:
            click.echo(f"Removing stale PID file (process {pid} is gone).")
            PID_FILE.unlink(missing_ok=True)
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    port = os.getenv("EFFICIENT_PORT", "7432")
    host = os.getenv("EFFICIENT_HOST", "127.0.0.1")
    proc = subprocess.Popen(
        ["uvicorn", "efficient.daemon.app:app", "--host", host, "--port", port],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    PID_FILE.write_text(str(proc.pid))
    click.echo(f"Daemon started (PID {proc.pid}) at {_daemon_url()}")


@cli.command()
def stop():
    """Stop the efficient daemon."""
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


@cli.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def claude(args):
    """Launch Claude Code routed through the efficient gateway."""
    url = _daemon_url()
    try:
        httpx.get(f"{url}/health", timeout=2.0)
    except Exception:
        click.echo(f"warning: efficient daemon not reachable at {url} — "
                   "model calls will fail until it is up (docker compose up -d daemon)")
    env = {**os.environ, "ANTHROPIC_BASE_URL": url}
    os.execvpe("claude", ["claude", *args], env)


@cli.command()
@click.option("--once", is_flag=True, help="Sync configured directories once and exit.")
def watch(once):
    """Watch configured directories (~/.efficient/watch.json) and ingest text
    files into retrieval corpora."""
    from efficient.ingest import watcher
    watches = watcher.load_watches()
    if not watches:
        click.echo(f"No watches configured. Create {watcher.CONFIG_PATH} with "
                   '{"watches": [{"path": "~/notes", "corpus_id": "notes"}]}')
        return
    if once:
        summary = watcher.sync_once(watches)
        for corpus_id, s in summary.items():
            click.echo(f"  [{corpus_id}] {s['files']} files, {s['chunks']} chunks")
        click.echo("Sync complete.")
        return
    import asyncio
    roots = ", ".join(str(w["root"]) for w in watches)
    click.echo(f"Watching: {roots}  (Ctrl-C to stop)")
    try:
        asyncio.run(watcher.watch_forever(watches))
    except KeyboardInterrupt:
        click.echo("\nStopped.")


@cli.command()
def warmup():
    """Download and cache local models (embeddings + compressor)."""
    from efficient.modules.embeddings import _get_model
    click.echo("Loading embedding model (voyageai/voyage-4-nano)...")
    _get_model()
    click.echo("  embedding model ready.")
    from efficient.modules.context_compressor import _get_compressor
    click.echo("Loading compressor model (LLMLingua-2)...")
    _get_compressor()
    click.echo("  compressor model ready.")
    click.echo("Warmup complete.")


if __name__ == "__main__":
    cli()
