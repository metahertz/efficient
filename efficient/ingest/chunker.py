"""Deterministic text chunker for corpus ingestion. Packs paragraphs into
~target_chars chunks without splitting a paragraph unless it alone exceeds
the target (then hard-split on char boundaries)."""


def chunk_text(text: str, target_chars: int = 1200) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        if len(para) > target_chars:
            if buf:
                chunks.append(buf)
                buf = ""
            for i in range(0, len(para), target_chars):
                chunks.append(para[i:i + target_chars])
            continue
        if buf and len(buf) + 2 + len(para) > target_chars:
            chunks.append(buf)
            buf = para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if buf:
        chunks.append(buf)
    return chunks
