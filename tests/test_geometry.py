"""Synthetic Rust parity, cold limits and local forward-derivative checks."""

import json
from pathlib import Path

import pytest
import torch
from ctract_chimera import Anatomy, ExactGeometry

FIXTURE = json.loads((Path(__file__).parent / "fixtures/geometry.json").read_text())
# Rust's built-in bodies only appear in parity tests, never in the library API.
CONFIGS = {"male": Anatomy(), "female": Anatomy(0.9, 0.88, 2.2, 4.4), "child": Anatomy(0.74, 0.8, 2.15, 4.0)}


@pytest.mark.parametrize("name", CONFIGS)
@pytest.mark.parametrize("dtype,tolerance", [(torch.float64, 1e-10), (torch.float32, 3e-3)])
def test_hard_rust(name: str, dtype: torch.dtype, tolerance: float) -> None:
    rows = FIXTURE[name]["rows"]
    model = ExactGeometry(CONFIGS[name]).to(dtype=dtype)
    articulation = torch.tensor([r["artic"] for r in rows], dtype=dtype).reshape(2, 16, 17)
    for key, value in model(articulation).items():
        expected = torch.tensor([r["or" if key == "orl" else key] for r in rows], dtype=dtype)
        torch.testing.assert_close(value.flatten(0, 1), expected, atol=tolerance, rtol=tolerance)


def test_local_gradcheck() -> None:
    model = ExactGeometry()
    generator = torch.Generator().manual_seed(73)
    a = model.lo + (model.hi - model.lo) * (0.2 + 0.6 * torch.rand(1, 17, dtype=torch.float64, generator=generator))
    assert torch.autograd.gradcheck(
        lambda x: tuple(model(x, mode="fit").values()), (a.requires_grad_(),), fast_mode=True
    )


@pytest.mark.parametrize("name", CONFIGS)
def test_contact_gradients_finite_and_local_relaxation_converges(name: str) -> None:
    model = ExactGeometry(CONFIGS[name])
    a = torch.tensor([r["artic"] for r in FIXTURE[name]["rows"]], dtype=torch.float64).requires_grad_()
    hard, cold = model(a), model(a, mode="fit", temperature=1e-6)
    for key in hard:
        torch.testing.assert_close(cold[key], hard[key], rtol=1e-5, atol=1e-6)
    sum(v.sum() for v in model(a, mode="fit").values()).backward()
    assert torch.isfinite(a.grad).all()
    # Per-input nonzero sensitivity and deep-contact sweeps are checked
    # separately in test_continuous_geometry.py at the declared temperature.


def test_anatomy_configuration() -> None:
    model = ExactGeometry(Anatomy(0.83, 0.91, 2.18, 4.3))
    assert model.n_ph >= 8 and model.n_or >= 8
    with pytest.raises(ValueError):
        Anatomy(scale=0)
