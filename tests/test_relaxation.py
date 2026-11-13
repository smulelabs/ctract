"""Check smooth local operators without relying on autograd alone."""

import torch
from ctract_chimera.relaxation import lower_bound, opening, positive


def test_algebraic_positive_and_opening() -> None:
    x = torch.tensor([-10.0, -0.01, 0.0, 0.01, 10.0], dtype=torch.float64, requires_grad=True)
    width = torch.tensor(0.001, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(positive, (x, width))
    assert torch.autograd.gradcheck(lambda a, w: opening(a, w, 0.002), (x, width))
    assert torch.autograd.gradcheck(lambda a, w: lower_bound(a, 0.02, w), (x, width))
    (positive(x, width) + opening(x, width, 0.002)).sum().backward()
    assert x.grad.isfinite().all() and (x.grad > 0).all()
    cold = positive(x, 1e-10)
    torch.testing.assert_close(cold, x.clamp_min(0), atol=1e-9, rtol=1e-9)
    inclusive = opening(x, 1e-10, 0.002)
    torch.testing.assert_close(inclusive, (x >= 0).to(x.dtype), atol=1e-6, rtol=1e-6)
