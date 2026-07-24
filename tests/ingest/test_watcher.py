import json

from efficient.ingest import watcher


class _FakeResp:
    def raise_for_status(self): pass
    def json(self): return {}


class _FakeClient:
    def __init__(self):
        self.posts = []

    def __enter__(self): return self
    def __exit__(self, *a): pass

    def post(self, url, json=None, headers=None, timeout=None):
        self.posts.append((url, json))
        return _FakeResp()


def test_load_watches(tmp_path, monkeypatch):
    cfg = tmp_path / "watch.json"
    cfg.write_text(json.dumps({"watches": [
        {"path": str(tmp_path / "notes"), "corpus_id": "notes"}]}))
    watches = watcher.load_watches(cfg)
    assert watches[0]["corpus_id"] == "notes"
    assert watches[0]["extensions"] == watcher._DEFAULT_EXTS


def test_sync_once_ingests_files(tmp_path, monkeypatch):
    root = tmp_path / "notes"
    (root / "sub").mkdir(parents=True)
    (root / "a.md").write_text("# Title\n\nBody paragraph.")
    (root / "sub" / "b.txt").write_text("plain note")
    (root / "ignore.png").write_bytes(b"\x89PNG")

    fake = _FakeClient()
    monkeypatch.setattr(watcher.httpx, "Client", lambda: fake)
    summary = watcher.sync_once([
        {"root": root, "corpus_id": "notes", "extensions": [".md", ".txt"]}])

    assert summary["notes"]["files"] == 2
    add_posts = [p for p in fake.posts if p[0].endswith("/corpus/add-chunks")]
    sources = {p[1]["chunks"][0]["source_file"] for p in add_posts}
    assert sources == {"a.md", "sub/b.txt"}  # relative paths, png excluded


def test_ingest_missing_dir_skipped(tmp_path, monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(watcher.httpx, "Client", lambda: fake)
    summary = watcher.sync_once([
        {"root": tmp_path / "nope", "corpus_id": "x", "extensions": [".md"]}])
    assert summary == {}
    assert fake.posts == []
