"""On-disk KV block store: one file per chunk.

Format: safetensors with tensors k{i}/v{i} per layer, plus a JSON
sidecar (.meta) for positions dtype, quant mode, scales, and
provenance. Position-independent layout: positions are always
0..L-1 on disk; re-indexing happens at concat time.

An in-RAM LRU keeps hot blocks resident (configurable cap).
Cold blocks are mmap-friendly safetensors loads.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from relay.block import KVBlock

try:
    from safetensors.numpy import save_file, load_file
    _HAVE_SAFE = True
except ImportError:
    _HAVE_SAFE = False


class KVBlockStore:
    def __init__(self, root: str | Path, lru_cap: int = 512):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lru: OrderedDict[str, KVBlock] = OrderedDict()
        self.lru_cap = lru_cap

    def _paths(self, cid: str) -> tuple[Path, Path]:
        return self.root / f"{cid}.safetensors", self.root / f"{cid}.meta.json"

    def put(self, block: KVBlock) -> None:
        if not _HAVE_SAFE:
            raise RuntimeError("safetensors not installed; pip install safetensors")
        tensors = {}
        for i, (k, v) in enumerate(zip(block.keys, block.values)):
            tensors[f"k{i}"] = np.ascontiguousarray(k)
            tensors[f"v{i}"] = np.ascontiguousarray(v)
        data_path, meta_path = self._paths(block.chunk_id)
        save_file(tensors, str(data_path))
        meta = {
            "chunk_id": block.chunk_id,
            "quant": block.quant,
            "scales": {kk: float(vv) for kk, vv in block.scales.items()},
            "positions_dtype": str(block.positions.dtype),
            "source": block.source,
            "token_count": block.token_count,
            "n_layers": block.n_layers,
        }
        meta_path.write_text(json.dumps(meta))
        self._lru_put(block.chunk_id, block)

    def get(self, cid: str) -> KVBlock:
        if cid in self.lru:
            self.lru.move_to_end(cid)
            return self.lru[cid]
        data_path, meta_path = self._paths(cid)
        if not data_path.exists():
            raise KeyError(f"KV block not found: {cid}")
        if not _HAVE_SAFE:
            raise RuntimeError("safetensors not installed")
        meta = json.loads(meta_path.read_text())
        tensors = load_file(str(data_path))
        n = meta["n_layers"]
        keys = [tensors[f"k{i}"] for i in range(n)]
        values = [tensors[f"v{i}"] for i in range(n)]
        L = keys[0].shape[1]
        scales = {kk: np.asarray(vv, dtype=np.float32)
                  for kk, vv in meta["scales"].items()}
        block = KVBlock(
            chunk_id=cid, keys=keys, values=values,
            positions=np.arange(L).astype(meta["positions_dtype"]),
            quant=meta["quant"], scales=scales,
            source=meta.get("source", ""), token_count=meta.get("token_count", L),
        )
        self._lru_put(cid, block)
        return block

    def _lru_put(self, cid: str, block: KVBlock) -> None:
        self.lru[cid] = block
        self.lru.move_to_end(cid)
        while len(self.lru) > self.lru_cap:
            self.lru.popitem(last=False)

    def __contains__(self, cid: str) -> bool:
        return (self.root / f"{cid}.safetensors").exists()
