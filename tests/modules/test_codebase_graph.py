import pytest
from pathlib import Path
from finops.modules.codebase_graph import CodebaseGraph
from finops.modules._base import OptimizeRequest
from finops.db.collections import CODEBASE_NODES

FIXED_EMBEDDING = [0.1] * 1024
FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "sample.py"
GRAPH_FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "graph_sample.py"


@pytest.fixture(autouse=True)
def mock_embed(monkeypatch):
    monkeypatch.setattr("finops.modules.codebase_graph.embed_query", lambda t: FIXED_EMBEDDING)
    monkeypatch.setattr("finops.modules.codebase_graph.embed_documents", lambda ts: [FIXED_EMBEDDING] * len(ts))


@pytest.fixture
async def graph(finops_db):
    return CodebaseGraph(finops_db, {"repo_paths": []})


@pytest.fixture
def sample_source():
    return FIXTURE_PATH.read_text()


@pytest.fixture
def graph_source():
    return GRAPH_FIXTURE_PATH.read_text()


async def test_index_file_returns_symbol_count(graph, sample_source):
    count = await graph.index_file("repo1", "sample.py", sample_source)
    assert count >= 3


async def test_index_file_stores_symbols_in_mongo(graph, finops_db, sample_source):
    await graph.index_file("repo1", "sample.py", sample_source)
    count = await finops_db[CODEBASE_NODES].count_documents({"repo_id": "repo1"})
    assert count >= 3


async def test_index_file_captures_function_metadata(graph, finops_db, sample_source):
    await graph.index_file("repo1", "sample.py", sample_source)
    doc = await finops_db[CODEBASE_NODES].find_one({"repo_id": "repo1", "symbol": "add"})
    assert doc is not None
    assert doc["type"] == "function"
    assert doc["file_path"] == "sample.py"
    assert doc["line_start"] >= 1
    assert "def add" in doc["source_snippet"]


async def test_index_file_captures_class(graph, finops_db, sample_source):
    await graph.index_file("repo1", "sample.py", sample_source)
    doc = await finops_db[CODEBASE_NODES].find_one({"repo_id": "repo1", "symbol": "Calculator"})
    assert doc is not None
    assert doc["type"] == "class"


async def test_process_no_op_when_no_repo_configured(graph):
    req = OptimizeRequest(prompt="find add function", context="orig", agent_id="a1", framework="test")
    new_req, result = await graph.process(req)
    assert new_req is req
    assert new_req.context == "orig"
    assert result.tokens_saved == 0
    assert result.short_circuit is False


async def test_index_file_stores_both_same_named_symbols(graph, finops_db, sample_source):
    await graph.index_file("repo1", "sample.py", sample_source)
    count = await finops_db[CODEBASE_NODES].count_documents({"repo_id": "repo1", "symbol": "add"})
    assert count == 2


async def test_process_no_op_when_no_symbols_match(finops_db):
    cg = CodebaseGraph(finops_db, {"repo_paths": ["repo1"]})
    req = OptimizeRequest(prompt="find something", context="ORIG", agent_id="a", framework="f")
    new_req, result = await cg.process(req)
    assert new_req.context == "ORIG"
    assert result.tokens_saved == 0


async def test_index_populates_references(graph, finops_db, graph_source):
    await graph.index_file("grepo", "graph_sample.py", graph_source)
    doc = await finops_db[CODEBASE_NODES].find_one({"repo_id": "grepo", "symbol": "main"})
    assert doc is not None
    assert "helper" in doc["references"]


async def test_callees(graph, graph_source):
    await graph.index_file("grepo", "graph_sample.py", graph_source)
    callees = await graph.callees("grepo", "main")
    assert any(c["symbol"] == "helper" for c in callees)


async def test_callers(graph, graph_source):
    await graph.index_file("grepo", "graph_sample.py", graph_source)
    callers = await graph.callers("grepo", "helper")
    symbols = {c["symbol"] for c in callers}
    assert "main" in symbols
    assert "run" in symbols


async def test_callers_empty_for_unreferenced(graph, graph_source):
    await graph.index_file("grepo", "graph_sample.py", graph_source)
    callers = await graph.callers("grepo", "main")
    assert callers == []
