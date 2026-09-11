"""MCP server implementation (skeleton -- wire up on the Mac).

Depends on: the MCP Python SDK (pip install mcp), a configured
Backend, and the store. Run:

    python -m server.server --backend mlx --store-dir ./data

Stdio transport is the default (Claude Code / Claude Desktop).
Add --http --port 8000 for streamable HTTP.
"""

from __future__ import annotations

import argparse
import json


def build_app(backend_name: str, store_dir: str):
    """Construct the MCP app. TODO(morning): implement with the MCP SDK."""
    raise NotImplementedError(
        "Wire up on the Mac: "
        "1. pip install mcp; "
        "2. instantiate the Backend (mlx) and stores; "
        "3. register remember/retrieve/why tools per server/__init__.py; "
        "4. remember: chunk (~400 tok, sentence boundaries) -> embed "
        "(Qwen3-Embedding-0.6B) -> MetadataStore.add_chunk -> "
        "contradiction check (top-5 nearest, reader hop) -> "
        "backend.prefill -> quantize_int8 -> KVBlockStore.put; "
        "5. retrieve: run_latent_pipeline (agents/) with vector search "
        "as retrieve_fn, log trace via MetadataStore.log_trace; "
        "6. why: read the trace, decode with the base model + "
        "'describe what you concluded' prompt."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Latent memory MCP server")
    parser.add_argument("--backend", default="mlx", choices=["mlx", "numpy-mock"])
    parser.add_argument("--store-dir", default="./data")
    parser.add_argument("--http", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    build_app(args.backend, args.store_dir)


if __name__ == "__main__":
    main()
