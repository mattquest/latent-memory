"""Eval: experiment harnesses E1-E6.

E3 (mismatched-cache audit) is implemented and runnable here -- it is
the evidence standard the field adopted (Cheng et al., Aug 2026) and
the gate for trusting any latent relay.

E1/E2/E4/E5/E6 are scaffolded: they need the MLX backend and (for
E6) the BEAM benchmark. Their runners live here with clear TODOs.
"""

from .audit import mismatched_cache_audit, AuditResult
from .runners import run_e1, run_e2, run_e4, run_e5, run_e6

__all__ = [
    "mismatched_cache_audit",
    "AuditResult",
    "run_e1",
    "run_e2",
    "run_e4",
    "run_e5",
    "run_e6",
]
