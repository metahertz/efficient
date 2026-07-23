import subprocess
from pathlib import Path

HELPER = Path(__file__).parent.parent.parent / "plugin" / "scripts" / "_repo_id.sh"


def _compute(env_extra=None, cwd=None):
    script = f'. "{HELPER}"; compute_repo_id'
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin"}
    if env_extra:
        env.update(env_extra)
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                          env=env, cwd=cwd).stdout.strip()


def test_env_override_wins(tmp_path):
    out = _compute({"EFFICIENT_REPO_ID": "custom-id", "CLAUDE_PROJECT_DIR": str(tmp_path)})
    assert out == "custom-id"


def test_stable_across_calls(tmp_path):
    env = {"CLAUDE_PROJECT_DIR": str(tmp_path)}
    assert _compute(env) == _compute(env)


def test_differs_by_path(tmp_path):
    a = _compute({"CLAUDE_PROJECT_DIR": str(tmp_path / "a")})
    b = _compute({"CLAUDE_PROJECT_DIR": str(tmp_path / "b")})
    assert a != b


def test_format_basename_and_hash(tmp_path):
    d = tmp_path / "myrepo"
    d.mkdir()
    out = _compute({"CLAUDE_PROJECT_DIR": str(d)})
    assert out.startswith("myrepo-")
    suffix = out.split("myrepo-", 1)[1]
    assert len(suffix) >= 6 and suffix.isalnum()
