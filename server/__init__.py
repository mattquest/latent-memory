"""MCP server: text-in / text-out boundary.

Three tools. Keep the surface tiny; every extra tool costs host-model
context on every session.

  remember(text, source="") -> {chunk_id, tokens}
      Chunk, embed, contradiction-check, precompute KV block, store.
  retrieve(question, budget=3) -> {answer, hops, trace_id}
      Run the latent pipeline (falls back to text pipeline if the
      latent kill criteria fail -- see README).
  why(trace_id) -> {explanation}
      Decode the trace log into text: which chunks were attended,
      attention mass per block, verifier verdicts.

Transport: stdio for Claude Code / Claude Desktop; optional
streamable HTTP for Cowork / remote.

The host model never sees latent state. The boundary is text on
both sides -- that is the entire security model for the relay.
"""

from .server import main

__all__ = ["main"]
