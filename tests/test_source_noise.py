"""Gradient checks for the explicitly scoped source and filter primitives."""

import torch
from ctract_chimera import glottal_shape
from ctract_chimera.noise import band_noise, turbulence_amplitude


def test_glottal_shape_gradcheck() -> None:
    args = (
        torch.tensor([0.15, 0.47, 0.56], dtype=torch.float64, requires_grad=True),
        *(torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in (0.62, 0.62, 0.6)),
    )
    assert torch.autograd.gradcheck(glottal_shape, args)
    glottal_shape(*args).sum().backward()
    assert all(a.grad.isfinite().all() and a.grad.abs().sum() > 0 for a in args)


def test_filter_gradcheck() -> None:
    white = torch.linspace(-1, 1, 12, dtype=torch.float64, requires_grad=True)
    lp = torch.tensor(3500.0, dtype=torch.float64, requires_grad=True)
    hp = torch.tensor(300.0, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda *args: band_noise(args[0], 96000.0, *args[1:]), (white, lp, hp))


def test_turbulence_gradcheck_and_threshold() -> None:
    args = tuple(torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (0.2, 150.0, 3.2e-6))
    assert torch.autograd.gradcheck(lambda *x: turbulence_amplitude(*x, temperature=0.05), args)
    amplitude = turbulence_amplitude(*args, temperature=0.05)
    amplitude.backward()
    assert all(a.grad.isfinite() and a.grad != 0 for a in args)
    # Rust hard mode has no turbulence at zero flow or behind a closed source.
    assert turbulence_amplitude(torch.tensor(0.0), args[1], args[2]) == 0
    assert turbulence_amplitude(args[0], torch.tensor(0.0), args[2]) == 0
