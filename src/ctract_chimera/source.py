"""Area pulse primitive from sources/glottis.rs::shape.

This is the pulse shape, NOT ParametricGlottis::next: it does not turn glottal
area into volume velocity. In CHIMERA that requires pressure feedback and the
nonlinear glottal port. Keeping the units explicit prevents a misleading area
pulse → linear-flow substitution. The full interactive source is not ported.
"""

import torch
from torch import Tensor


def glottal_shape(phase: Tensor, open_quotient: Tensor, skew: Tensor, sharpness: Tensor) -> Tensor:
    """Normalised CHIMERA area pulse for phase in [0,1), broadcast inputs.

    ``sharpness`` blends raised-cosine and cosine closure; ``skew`` is clamped to
    [0.3,0.9] exactly as Rust. open_quotient must be strictly between 0 and 1.
    No phase accumulator, jitter, shimmer, fry, voicing gate or pressure feedback
    is implied by this primitive. Branches remain piecewise differentiable.
    """
    s = skew.clamp(0.3, 0.9)
    t = phase / open_quotient
    rise = 0.5 - 0.5 * torch.cos(torch.pi * t / s)
    u = (t - s) / (1 - s)
    soft = 0.5 + 0.5 * torch.cos(torch.pi * u)
    hard = torch.cos(0.5 * torch.pi * u)
    fall = soft + (hard - soft) * sharpness
    return torch.where(phase >= open_quotient, 0.0, torch.where(t < s, rise, fall))
