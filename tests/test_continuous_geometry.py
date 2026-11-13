"""Forward-derivative checks at contacts, topology changes and interpolation ties."""

import json
from pathlib import Path

import pytest
import torch
from ctract_chimera import NEUTRAL, ExactGeometry
from ctract_chimera.geometry_relaxation import Relaxation, adjacency


def test_sorting_and_visibility_adjacency_gradcheck_at_ties() -> None:
    r = Relaxation(torch.tensor(0.5, dtype=torch.float64))
    coordinates = torch.tensor([[0.4, 0.2, 0.2, 0.7, 0.5]], dtype=torch.float64, requires_grad=True)
    payload = torch.tensor([[0.1, 0.3, 0.6, 0.8, 0.4]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(r.sort, (coordinates, payload), eps=1e-7, fast_mode=True)
    visible = torch.tensor([[0.001, 0.4, 0.9, 0.999]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(adjacency, (visible,), fast_mode=True)
    hard = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0]], dtype=torch.float64)
    pairs, first, last = adjacency(hard)
    assert pairs[0, 0, 2] == pairs[0, 2, 4] == 1
    assert pairs.sum() == 2 and first[0, 0] == last[0, 4] == 1


def test_polyline_primitives_gradcheck() -> None:
    r = Relaxation(torch.tensor(0.5, dtype=torch.float64))
    wall = torch.tensor([[[0.0, 0.0], [1.0, 0.1], [1.1, 1.2], [0.0, 1.0]]], dtype=torch.float64, requires_grad=True)
    points = torch.tensor([[[0.3, 0.2], [0.7, 0.8]]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(r.project, (wall, points), eps=1e-7, fast_mode=True)
    assert torch.autograd.gradcheck(r.signed_gap, (wall, points), eps=1e-7, fast_mode=True)
    query = torch.tensor([[0.2, 0.95, 1.7]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(r.point_along, (wall, query), eps=1e-7, fast_mode=True)
    direction = torch.tensor([[[1.0, 0.2], [0.2, 1.0]]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(r.crossings, (points, direction, wall), eps=1e-7, fast_mode=True)


@pytest.mark.parametrize(
    "parameter,centre,span",
    [(7, 0.0, 0.4), (11, 0.0, 0.25), (15, 0.0, 0.2), (10, 0.5, 0.5), (3, 6.2, 0.5), (4, 9.2, 0.6)],
)
def test_transition_sensitivities(parameter: int, centre: float, span: float) -> None:
    """Compressed lips, velum seal, lateral release, dental and tongue contacts.

    All 17 input sensitivities must stay live, including parameters at their
    bounds. This is a finite-temperature claim, not an impossible uniform bound
    on every derivative of a discontinuous zero-temperature map.
    """
    model = ExactGeometry()
    pose = torch.tensor([NEUTRAL], dtype=torch.float64).expand(9, -1).clone()
    pose[:, parameter] = torch.linspace(centre - span, centre + span, 9, dtype=torch.float64)
    pose.requires_grad_()
    result = model(pose, mode="fit", temperature=0.05)
    loss = sum(value.sum() for value in result.values())
    gradient = torch.autograd.grad(loss, pose)[0]
    assert gradient.isfinite().all() and (gradient != 0).all()
    assert gradient.abs().max() < 1e7


def test_nonzero_bounded_gradients_on_synthetic_closure_fixtures() -> None:
    rows = json.loads((Path(__file__).parent / "fixtures/geometry.json").read_text())["male"]["rows"]
    model = ExactGeometry()
    for row in rows:
        pose = torch.tensor([row["artic"]], dtype=torch.float64, requires_grad=True)
        result = model(pose, mode="fit", temperature=0.05)
        gradient = torch.autograd.grad(sum(value.sum() for value in result.values()), pose)[0]
        assert gradient.isfinite().all() and (gradient != 0).all()
        assert gradient.abs().max() < 1e7


def test_fp32_controls_keep_reopening_gradients() -> None:
    model = ExactGeometry().float()
    pose = torch.tensor([NEUTRAL], dtype=torch.float32, requires_grad=True)
    result = model(pose, mode="fit", temperature=0.05)
    gradient = torch.autograd.grad(sum(value.sum() for value in result.values()), pose)[0]
    assert gradient.isfinite().all() and (gradient != 0).all()


def test_deep_lip_contact_log_sensitivity_survives() -> None:
    model = ExactGeometry()
    for aperture in [-0.4, -0.2, -0.05, 0.0, 0.05]:
        pose = torch.tensor([NEUTRAL], dtype=torch.float64)
        pose[:, 7] = aperture
        pose.requires_grad_()
        oral = model(pose, mode="fit", temperature=0.05)["orl"]
        derivative = torch.autograd.grad(oral[:, -1].log().sum(), pose)[0][0, 7]
        assert derivative.isfinite() and 1e-3 < derivative.abs() < 1e4
    # A positive local gap derivative is guaranteed by the relaxation itself;
    # global articulator→area maps are nonconvex and need not be monotone.
    r = Relaxation(torch.tensor(0.05, dtype=torch.float64))
    gap = torch.tensor([-0.4, -0.2, -0.05, 0.0, 0.05], dtype=torch.float64, requires_grad=True)
    local = torch.autograd.grad(r.positive(gap).log().sum(), gap)[0]
    assert (local > 0).all() and local.isfinite().all()
