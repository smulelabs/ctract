"""Local smooth extensions with algebraic tails, not straight-through gradients.

These are numerical relaxations, not CHIMERA physics. Width has the units of x.
They converge pointwise to positive-part and the declared hard predicates.
Geometry's complete topology extension is in geometry_smooth.py; these local
helpers alone do not relax changes in adjacency or section ownership.
"""

import torch
from torch import Tensor


def positive(x: Tensor, width: float | Tensor) -> Tensor:
    """Smooth max(x,0): width·exp(asinh(x/width))/2.

    Algebraically (x+hypot(x,width))/2, written without catastrophic negative
    tail cancellation. Algebraic tails keep useful derivatives deeper into
    closures than exp(x/width), although no finite dtype can promise nonzero
    derivatives for arbitrarily small widths or arbitrarily large inputs.
    """
    w = torch.as_tensor(width, dtype=x.dtype, device=x.device)
    # r+x = w exp(asinh(x/w)); equivalent stable two-branch-free form.
    return 0.5 * w * torch.exp(torch.asinh(x / w))


def lower_bound(x: Tensor, floor: float, width: float | Tensor) -> Tensor:
    """Smooth max(x,floor), tending to the floor as width→0 when x<floor."""
    return floor + positive(x - floor, width)


def opening(x: Tensor, width: float | Tensor, reference: float) -> Tensor:
    """Smooth x>=0 with inclusive-tie bias, algebraic (not exponential) tails.

    sigmoid(2 asinh(x/w) − log(w/reference)). For x=0 this tends to 1;
    x<0 tends to 0, and x>0 to 1. reference only nondimensionalises the bias.
    """
    w = torch.as_tensor(width, dtype=x.dtype, device=x.device)
    return torch.sigmoid(2 * torch.asinh(x / w) - torch.log(w / reference))


def strict_opening(x: Tensor, width: float | Tensor, reference: float) -> Tensor:
    """Smooth x>0, including the strict closed tie in its cold limit."""
    w = torch.as_tensor(width, dtype=x.dtype, device=x.device)
    return torch.sigmoid(2 * torch.asinh(x / w) + torch.log(w / reference))
