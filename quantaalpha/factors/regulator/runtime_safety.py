"""
runtime_safety.py — Runtime expression safety checker.

This module is intentionally kept free of any intra-package imports so that
it can be imported by both factor_regulator.py and proposal.py without
triggering the circular-import chain:
  factor_regulator → factor_ast → coder/__init__ → evaluators → factor_regulator

NOTE (Round 14 Codex review):
The runtime-safety blacklist was removed because the execution template
(template.jinjia2:13-18) already auto-rewrites bare variable names like 'close'
to df['$close'] before eval. Blocking bare variables during proposal validation
provided no actual safety benefit — it only rejected expressions that would
execute successfully after the auto-rewrite, unnecessarily shrinking the factor
search space. The executor's auto-rewrite behavior is the canonical validation.
"""
from typing import Tuple


def check_runtime_safety(expression: str) -> Tuple[bool, str]:
    """
    Check whether *expression* contains patterns that pass AST validation
    but will fail at Python eval() time.

    Currently disabled: the execution template auto-rewrites bare OHLCV variable
    names (close → df['$close']), so rejecting them here would only discard
    factors that the executor can successfully evaluate.

    Returns:
        (True, "")  — always returns safe (blacklist removed per Round 14 review).
    """
    return True, ""
