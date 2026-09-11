"""Real-model E1–E6 diagnostics and a separate legacy mock audit.

The real runners require native MLX and explicit data/output paths. They
implement the recorded fixed-evidence protocol, not the proposed trained
adaptive pipeline. audit.py remains a NumPy mechanical-proxy smoke test.
"""

from .audit import mismatched_cache_audit, AuditResult
from .runners import run_e1, run_e2, run_e3, run_e4, run_e5, run_e6

__all__ = [
    "mismatched_cache_audit",
    "AuditResult",
    "run_e1",
    "run_e2",
    "run_e3",
    "run_e4",
    "run_e5",
    "run_e6",
]
