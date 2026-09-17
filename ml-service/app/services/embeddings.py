"""Chunk/query embeddings via fastembed (ONNX runtime, no torch needed).

BAAI/bge-small-en-v1.5 outputs 384-dim vectors, matching the pgvector
column on Chunk.embedding (see app/models.py: EMBEDDING_DIM).
"""

from functools import lru_cache

from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"

# bge models are trained with an instruction prefix on the query side only;
# document/chunk text is embedded as-is.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache
def _model() -> TextEmbedding:
    return TextEmbedding(model_name=MODEL_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return [vec.tolist() for vec in _model().embed(texts)]


def embed_query(query: str) -> list[float]:
    return embed_texts([QUERY_PREFIX + query])[0]
