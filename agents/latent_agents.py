"""Latent agents: the EXPERIMENT pipeline.

Hops pass KVBlock relays; nothing is decoded until the synthesizer.
Structure mirrors the text pipeline exactly (same hop budget, same
retrieval) so E1 (text vs latent at equal hops) is a fair fight.

Status: relay mechanics (prefill -> block -> concat -> reindex) are
implemented and tested. The verifier's learned head and the
planner->retriever projection are stubs -- wire them up on the Mac.

The one trained projection in the system (planner hidden state ->
retriever embedding space, Coconut-style) lives in LatentPlanner.
It is optional in v1 and must pass the E3 mismatched-cache audit
before being trusted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from relay import KVBlock, concat_blocks, quantize_int8
from relay.bridge import select_bridge_tokens
from .interface import Agent, Relay
from engine.backends import Backend


class LatentPlanner(Agent):
    role = "planner"

    def __call__(self, question: str) -> Relay:
        toks = self.backend.encode(question)
        block = self.backend.prefill(toks, self.adapter)
        # Coconut-style latent query: final hidden state, projected.
        # STUB: identity projection. Train the linear head in train/.
        h = self.backend.hidden_state(toks, self.adapter)
        return Relay(blocks=block, latent_query=h, source_roles=[self.role],
                     hop=0, trace={"question": question})


class LatentReader(Agent):
    role = "reader"

    def __call__(self, chunk_blocks: list[KVBlock], hop: int = 0) -> Relay:
        # Chunks were prefilled with independent attention at remember
        # time; here we just concat + reindex and hand the relay on.
        # No decoding happens.
        merged = concat_blocks(chunk_blocks)
        return Relay(blocks=merged, source_roles=[self.role] * len(chunk_blocks),
                     hop=hop,
                     trace={"chunk_ids": [b.chunk_id for b in chunk_blocks]})


class LatentVerifier(Agent):
    role = "verifier"

    def __call__(self, relay: Relay, planner_relay: Relay) -> tuple[str, Relay]:
        """Score support per block; decide continue vs stop.

        Returns (decision, relay_for_next_hop).
        STUB: the learned head over attention-mass features is not
        trained yet. Current heuristic: continue while hop < 2, else
        stop. Replace with the trained head before running E1/E2.
        """
        # In production: consume relay.blocks + planner_relay.blocks,
        # run one prefill with the verifier adapter, read attention
        # mass per block, feed to the learned head.
        decision = "continue" if relay.hop < 2 else "stop"
        out = Relay(blocks=relay.blocks, latent_query=relay.latent_query,
                    source_roles=relay.source_roles + [self.role],
                    hop=relay.hop + 1,
                    trace={"decision": decision, **relay.trace})
        return decision, out


class LatentSynthesizer(Agent):
    role = "synthesizer"

    def __call__(self, relay: Relay, question: str,
                 bridge_ratio: float = 0.15) -> str:
        qtoks = self.backend.encode(question)
        out = self.backend.decode(relay.blocks, qtoks, self.adapter,
                                  max_tokens=128)
        return self.backend.decode_tokens(out)


@dataclass
class LatentResult:
    answer: str
    hops: int
    trace: list[dict]
    tokens_to_host: int  # text tokens returned; latent hops cost 0


def run_latent_pipeline(
    backend: Backend,
    question: str,
    retrieve_fn,
    max_hops: int = 3,
    bridge_ratio: float = 0.15,
) -> LatentResult:
    """Run the latent pipeline. retrieve_fn(query, k) -> list[KVBlock]."""
    planner = LatentPlanner(backend)
    reader = LatentReader(backend)
    verifier = LatentVerifier(backend)
    synth = LatentSynthesizer(backend)

    plan = planner(question)
    trace: list[dict] = [{"hop": 0, "planner_tokens": plan.n_tokens}]
    relay = plan
    hops = 0
    for hop in range(1, max_hops + 1):
        hops = hop
        blocks = retrieve_fn(plan.latent_query, k=8)
        r = reader(blocks, hop=hop)
        # merge reader relay with planner context
        merged = Relay(blocks=concat_blocks([plan.blocks, r.blocks]),
                       latent_query=r.latent_query or plan.latent_query,
                       hop=hop)
        decision, relay = verifier(merged, plan)
        trace.append({"hop": hop, "n_blocks": len(blocks), "decision": decision,
                      "relay_tokens": relay.n_tokens})
        if decision == "stop":
            break

    answer = synth(relay, question, bridge_ratio)
    return LatentResult(answer=answer, hops=hops, trace=trace,
                        tokens_to_host=len(answer))
