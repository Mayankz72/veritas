"""Chunk/query embeddings via fastembed (ONNX runtime, no torch needed).

BAAI/bge-small-en-v1.5 outputs 384-dim vectors, matching the pgvector
column on Chunk.embedding (see app/models.py: EMBEDDING_DIM).
"""

import os
from functools import lru_cache

from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"

# fastembed's default batch_size is 256, so a whole paper's ~50-100 chunks
# get run through the ONNX model as one batch - fine locally, but it OOM-
# killed the process on a 512MB free-tier host (confirmed: a 1-chunk upload
# succeeded, a 58-chunk real paper didn't, with no Python traceback - the
# signature of a SIGKILL, not an app-level error). A small batch size trades
# a bit of wall-clock time for bounded peak memory regardless of document
# size. Override via EMBEDDING_BATCH_SIZE if your host has more headroom.
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))

# bge models are trained with an instruction prefix on the query side only;
# document/chunk text is embedded as-is.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache
def _model() -> TextEmbedding:
    return TextEmbedding(model_name=MODEL_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return [vec.tolist() for vec in _model().embed(texts, batch_size=EMBEDDING_BATCH_SIZE)]


def embed_query(query: str) -> list[float]:
    return embed_texts([QUERY_PREFIX + query])[0]
