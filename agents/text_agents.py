"""Text agents: the BASELINE pipeline (control arm).

Each hop decodes a short message; the next agent re-prefills it.
Same retrieval, same hop budget as the latent pipeline -- the only
difference is what crosses the hop boundary (text vs KV cache).

This is also the fallback product: if the latent experiment fails
its kill criteria, this pipeline ships.

With the NumpyMockBackend the text is garbage (random weights), but
the hop mechanics -- plan -> retrieve -> read -> verify -> (loop) ->
synthesize -- execute faithfully. Swap in MLXBackend for real text.
"""

from __future__ import annotations

from dataclasses import dataclass

from .interface import Agent, TextMessage
from engine.backends import Backend


# Prompts are short and stable: they become cached prefixes in production.
PLANNER_PROMPT = "Plan retrieval for the question. Emit search queries."
READER_PROMPT = "Read the chunks. Extract evidence for the question."
VERIFIER_PROMPT = "Verify evidence supports the question. Verdict: continue or stop."
SYNTH_PROMPT = "Synthesize the final answer from the evidence."


class TextPlanner(Agent):
    role = "planner"

    def __call__(self, question: str) -> TextMessage:
        toks = self.backend.encode(PLANNER_PROMPT + "\nQ: " + question)
        out = self.backend.decode(None, toks, self.adapter, max_tokens=32)
        return TextMessage(
            role=self.role, kind="query",
            text=self.backend.decode_tokens(out),
            trace={"question": question},
        )


class TextReader(Agent):
    role = "reader"

    def __call__(self, question: str, chunks: list[str]) -> TextMessage:
        joined = "\n".join(f"[{i}] {c[:400]}" for i, c in enumerate(chunks))
        toks = self.backend.encode(READER_PROMPT + f"\nQ: {question}\n{joined}")
        out = self.backend.decode(None, toks, self.adapter, max_tokens=64)
        return TextMessage(
            role=self.role, kind="evidence",
            text=self.backend.decode_tokens(out),
            trace={"n_chunks": len(chunks)},
        )


class TextVerifier(Agent):
    role = "verifier"

    def __call__(self, question: str, evidence: TextMessage) -> TextMessage:
        toks = self.backend.encode(
            VERIFIER_PROMPT + f"\nQ: {question}\nEvidence: {evidence.text[:800]}"
        )
        out = self.backend.decode(None, toks, self.adapter, max_tokens=16)
        verdict = self.backend.decode_tokens(out)
        # Mock heuristic: continue iff the word 'continue' appears.
        # Production: a trained head over attention-mass features.
        decision = "continue" if "continue" in verdict.lower() else "stop"
        return TextMessage(
            role=self.role,
            kind="verdict" if decision == "stop" else "refine",
            text=verdict,
            trace={"decision": decision},
        )


class TextSynthesizer(Agent):
    role = "synthesizer"

    def __call__(self, question: str, evidence: list[TextMessage]) -> TextMessage:
        joined = "\n".join(e.text[:600] for e in evidence)
        toks = self.backend.encode(SYNTH_PROMPT + f"\nQ: {question}\n{joined}")
        out = self.backend.decode(None, toks, self.adapter, max_tokens=128)
        return TextMessage(
            role=self.role, kind="answer",
            text=self.backend.decode_tokens(out),
        )


@dataclass
class PipelineResult:
    answer: TextMessage
    hops: int
    trace: list[dict]


def run_text_pipeline(
    backend: Backend,
    question: str,
    retrieve_fn,
    max_hops: int = 3,
) -> PipelineResult:
    """Run the text baseline. retrieve_fn(query, k) -> list[str]."""
    planner, reader = TextPlanner(backend), TextReader(backend)
    verifier, synth = TextVerifier(backend), TextSynthesizer(backend)

    plan = planner(question)
    evidence: list[TextMessage] = []
    trace: list[dict] = [{"hop": 0, "plan": plan.text[:120]}]
    query = question
    hops = 0
    for hop in range(1, max_hops + 1):
        hops = hop
        chunks = retrieve_fn(query, k=8)
        ev = reader(query, chunks)
        evidence.append(ev)
        verdict = verifier(query, ev)
        trace.append({"hop": hop, "n_chunks": len(chunks),
                      "decision": verdict.trace["decision"]})
        if verdict.trace["decision"] == "stop":
            break
        query = query + " " + verdict.text[:100]  # refined query

    answer = synth(question, evidence)
    return PipelineResult(answer=answer, hops=hops, trace=trace)
