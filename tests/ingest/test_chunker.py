from efficient.ingest.chunker import chunk_text


def test_empty():
    assert chunk_text("") == []
    assert chunk_text("\n\n  \n") == []


def test_small_text_single_chunk():
    assert chunk_text("one para\n\ntwo para", target_chars=1000) == ["one para\n\ntwo para"]


def test_packs_to_target():
    paras = "\n\n".join(["x" * 400 for _ in range(5)])
    chunks = chunk_text(paras, target_chars=1000)
    assert len(chunks) == 3  # 400+400 per chunk (902 with sep), 5 paras -> 2,2,1
    assert all(len(c) <= 1000 for c in chunks)


def test_oversized_paragraph_hard_split():
    chunks = chunk_text("y" * 2500, target_chars=1000)
    assert len(chunks) == 3
    assert "".join(chunks) == "y" * 2500


def test_deterministic():
    t = "a\n\n" + "b" * 1500 + "\n\nc"
    assert chunk_text(t) == chunk_text(t)
