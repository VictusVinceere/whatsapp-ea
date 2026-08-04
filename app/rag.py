"""
Retrieval-augmented generation over document chunks.

The idea in one line: instead of sending Claude a whole document library
(too many tokens, too expensive), send it the three or four passages whose
*meaning* sits closest to the question.

That "closest in meaning" is what embeddings buy you. Each chunk becomes a
384-number vector positioned so that semantically similar text lands
nearby, and Postgres finds the nearest ones with `<=>`. Keyword search
can't do this: a document saying "revenue" won't match "how much did we
make", but their embeddings are neighbours.
"""

import asyncio

import structlog
from fastembed import TextEmbedding
from sqlalchemy import text

from app.db import async_session
from app.llm import ask_claude

log = structlog.get_logger()

# Multilingual, 384 dimensions -- the same width as the English-only
# bge-small-en-v1.5 it replaces, so vector(384) still fits and no
# migration is needed. Everything already stored must still be
# re-embedded: vectors from a different model are not comparable, and
# mixing them silently ranks nonsense above real matches.
#
# The switch was forced by a real failure. With the English model, every
# chunk of a Russian-language PDF scored 0.48-0.50 while *unrelated
# English documents* scored 0.39-0.44 -- so the answer ranked below
# documents about expenses and onboarding. An English-only model cannot
# represent Russian meaning, so relevance stops working rather than
# degrading gracefully.
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_model: TextEmbedding | None = None

# Characters, not tokens -- close enough at this scale and far simpler.
# Too small and a chunk loses the context that makes it meaningful; too
# large and one chunk covers several topics, so it matches everything
# vaguely and nothing precisely.
CHUNK_SIZE = 800
# Overlap stops a sentence that straddles a boundary from being lost to
# both chunks. Costs some duplicate storage, buys recall at the seams.
CHUNK_OVERLAP = 150


def _get_model() -> TextEmbedding:
    """Loaded on first use, not at import -- keeps app startup fast."""
    global _model
    if _model is None:
        _model = TextEmbedding(model_name=EMBED_MODEL)
    return _model


def _embed_sync(texts: list[str]) -> list[list[float]]:
    return [v.tolist() for v in _get_model().embed(texts)]


async def embed(texts: list[str]) -> list[list[float]]:
    """Vectors for each text.

    fastembed is synchronous and CPU-bound, so it runs in a thread -- the
    same reason transcription.py wraps Deepgram. Calling it directly would
    block the event loop and stall every other in-flight request.
    """
    return await asyncio.to_thread(_embed_sync, texts)


def chunk_text(content: str) -> list[str]:
    """Split into overlapping windows, preferring paragraph boundaries.

    Cutting mid-sentence produces chunks that read as nonsense and embed
    badly, so the split point is nudged backwards to the last blank line
    or newline when there's one nearby.
    """
    content = content.strip()
    if not content:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(content):
        end = start + CHUNK_SIZE
        if end < len(content):
            # Look for a paragraph break in the last quarter of the window.
            window = content[start:end]
            for sep in ("\n\n", "\n", ". "):
                cut = window.rfind(sep, int(CHUNK_SIZE * 0.5))
                if cut != -1:
                    end = start + cut + len(sep)
                    break

        piece = content[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= len(content):
            break
        start = max(start + 1, end - CHUNK_OVERLAP)

    return chunks


def _to_vector_literal(vector: list[float]) -> str:
    """pgvector accepts '[0.1,0.2,...]' text and casts it. Simpler than
    registering a custom asyncpg codec through SQLAlchemy's pool."""
    return "[" + ",".join(str(x) for x in vector) + "]"


async def ingest_document(
    tenant_id: str,
    source_id: str,
    source_name: str,
    content: str,
) -> int:
    """Chunk, embed and store one document. Returns the chunk count.

    Deletes this source's existing chunks first, so re-ingesting an edited
    file replaces it rather than leaving stale passages behind that would
    keep surfacing in search results.
    """
    chunks = chunk_text(content)
    if not chunks:
        return 0

    vectors = await embed(chunks)

    async with async_session() as session:
        await session.execute(
            text(
                "DELETE FROM document_chunks "
                "WHERE tenant_id = :tenant_id AND source_id = :source_id"
            ),
            {"tenant_id": tenant_id, "source_id": source_id},
        )
        for index, (piece, vector) in enumerate(zip(chunks, vectors)):
            await session.execute(
                text(
                    "INSERT INTO document_chunks "
                    "(tenant_id, source_id, source_name, chunk_index, content, embedding) "
                    "VALUES (:tenant_id, :source_id, :source_name, :chunk_index, "
                    ":content, CAST(:embedding AS vector))"
                ),
                {
                    "tenant_id": tenant_id,
                    "source_id": source_id,
                    "source_name": source_name,
                    "chunk_index": index,
                    "content": piece,
                    "embedding": _to_vector_literal(vector),
                },
            )
        await session.commit()

    log.info("document_ingested", source_name=source_name, chunks=len(chunks))
    return len(chunks)


# Cosine distance beyond which a chunk is treated as noise. Defaults off:
# a first attempt at 0.45 was measured against an English-only model on a
# partly-Russian corpus, so the numbers described the model's blind spot
# rather than the documents, and the cutoff removed exactly the chunks
# holding the answer. Re-measure on your own corpus before enabling one.
MAX_DISTANCE = 1.0

# Eight rather than four. Four could not answer questions about a 43-chunk
# PDF: the top matches were all table-of-contents pages, and the real
# terms ranked below them. The cost is real (618 -> ~1500 tokens per
# query) and a wrong answer costs more.
DEFAULT_LIMIT = 8


async def search(
    tenant_id: str,
    query: str,
    limit: int = DEFAULT_LIMIT,
    max_distance: float = MAX_DISTANCE,
) -> list[dict]:
    """The chunks closest in meaning to `query`, nearest first.

    `<=>` is cosine distance: 0 is identical, 2 is opposite. It must match
    the operator class the index was built with (vector_cosine_ops) or
    Postgres silently ignores the index and scans every row.

    tenant_id is in the WHERE clause and that is not optional -- a
    similarity search without it happily returns another customer's
    documents, which is a data breach rather than a bug.
    """
    [query_vector] = await embed([query])

    async with async_session() as session:
        result = await session.execute(
            text(
                "SELECT source_name, content, "
                "       embedding <=> CAST(:query AS vector) AS distance "
                "FROM document_chunks "
                "WHERE tenant_id = :tenant_id "
                "ORDER BY distance "
                "LIMIT :limit"
            ),
            {
                "tenant_id": tenant_id,
                "query": _to_vector_literal(query_vector),
                "limit": limit,
            },
        )
        rows = result.mappings().all()

    # Filtered in Python rather than SQL so the cutoff can be tuned without
    # touching the query, and so a run that returns nothing is still
    # visible in the logs as "everything was too far away".
    kept = [dict(r) for r in rows if r["distance"] <= max_distance]
    if len(kept) < len(rows):
        log.info("chunks_filtered", kept=len(kept), dropped=len(rows) - len(kept))
    return kept


async def answer_with_context(tenant_id: str, question: str) -> str:
    """Retrieve, then generate. The whole point of RAG in six lines.

    Claude never sees the document library -- only the passages retrieved
    for this question. Telling it to say when the answer isn't there is
    what stops it filling the gap with plausible invention.
    """
    hits = await search(tenant_id, question)
    if not hits:
        return "I don't have any documents to search yet."

    context = "\n\n---\n\n".join(
        f"[{h['source_name']}]\n{h['content']}" for h in hits
    )
    prompt = (
        "Answer the question using only the context below. If the context "
        "doesn't contain the answer, say so plainly rather than guessing.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )

    log.info("rag_answering", chunks_used=len(hits),
             best_distance=round(hits[0]["distance"], 4))
    return await ask_claude(prompt)
