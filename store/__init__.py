"""Store: SQLite metadata + vector index + KV block store.

One directory, one process, no daemons.

  metadata.db   SQLite: chunks (id, text, source, timestamp, version,
                supersedes), plus relay trace log.
  vectors/      sqlite-vec (or LanceDB) for first-stage candidate search.
  kv_blocks/    one safetensors file per chunk: the precomputed KV
                block (int8 default), position-independent layout.

Write path (remember): chunk -> embed -> upsert -> contradiction
check -> precompute KV block -> write to kv_blocks/.
Read path: vector search -> load blocks (LRU) -> hand to agents.
"""

from .metadata import MetadataStore
from .kv_store import KVBlockStore

__all__ = ["MetadataStore", "KVBlockStore"]
