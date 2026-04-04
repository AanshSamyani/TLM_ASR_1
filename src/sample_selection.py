"""Perplexity-based sample selection strategy from the TLM paper.

High-perplexity samples are up-weighted; samples below the threshold P0
are skipped entirely (weight = 0).
"""

import math


class SampleSelector:
    """Compute per-sample weight S(x) from Eq. 6 of the paper.

    S(x) = lambda_ * exp(log P(x) - log P0)   if P(x) > P0
           0                                     otherwise
    """

    def __init__(self, lambda_val: float = 0.10, p0: float = None):
        """
        Args:
            lambda_val: scaling coefficient (lambda in the paper).
            p0: perplexity threshold.  Defaults to e^3 ~ 20.09.
        """
        self.lambda_val = lambda_val
        self.p0 = p0 if p0 is not None else math.exp(3)
        self.log_p0 = math.log(self.p0)

    def compute_weight(self, perplexity: float) -> float:
        """Return the sample weight.  0 means skip."""
        if perplexity <= self.p0:
            return 0.0
        return self.lambda_val * math.exp(math.log(perplexity) - self.log_p0)
