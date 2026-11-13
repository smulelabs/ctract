"""Complex Rust transfer oracle, element parity, and a local fp64 gradcheck."""

import json
from pathlib import Path

import torch
from ctract_chimera import HumanAcoustics, build_human


def oracle() -> tuple[dict, list[dict]]:
    lines = (Path(__file__).parent / "fixtures/transfer.jsonl").read_text().splitlines()
    return json.loads(lines[0]), [json.loads(line) for line in lines[1:]]


def stack(rows: list[dict], key: str) -> torch.Tensor:
    return torch.tensor([r[key] for r in rows], dtype=torch.float64)


def test_elements_and_complex_transfer() -> None:
    header, rows = oracle()
    ac = HumanAcoustics(build_human(header["fs"]))
    assert ac.h.topo.a == header["a"] and ac.h.topo.b == header["b"]
    el = ac.elements(*(stack(rows, key) for key in ("ph", "or", "ph_len", "or_len", "velum")))
    assert torch.equal(el.open, torch.tensor([r["open"] for r in rows]))
    for key in ("c", "wall_area", "rad_area"):
        torch.testing.assert_close(getattr(el, key), stack(rows, key), rtol=1e-12, atol=0)
    for key in ("l", "r"):
        torch.testing.assert_close(getattr(el, key)[el.open], stack(rows, key)[el.open], rtol=1e-12, atol=0)
    actual = ac.transfer(el, torch.tensor(header["freqs"], dtype=torch.float64))
    expected = torch.complex(stack(rows, "re"), stack(rows, "im"))
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)


def test_acoustic_gradcheck_all_geometry_inputs() -> None:
    ac = HumanAcoustics(build_human())
    inputs = (
        torch.full((1, ac.h.n_ph), 2.0, dtype=torch.float64, requires_grad=True),
        torch.full((1, ac.h.n_or), 3.0, dtype=torch.float64, requires_grad=True),
        torch.tensor([8.0], dtype=torch.float64, requires_grad=True),
        torch.tensor([7.0], dtype=torch.float64, requires_grad=True),
        torch.tensor([0.2], dtype=torch.float64, requires_grad=True),
    )
    frequencies = torch.tensor([100.0, 650.0, 2100.0], dtype=torch.float64)

    def response(*args: torch.Tensor) -> torch.Tensor:
        return ac.transfer(ac.elements(*args, temperature=1e-4), frequencies)

    assert torch.autograd.gradcheck(response, inputs, fast_mode=True)
    sum(response(*inputs).abs().flatten()).backward()
    assert all(x.grad.isfinite().all() and (x.grad != 0).all() for x in inputs)


def test_reciprocal_dipoles_against_dense_nodal_solve() -> None:
    """Independent dense solve checks dipole sign and the glottal cut."""
    ac = HumanAcoustics(build_human())
    el = ac.elements(
        torch.full((1, ac.h.n_ph), 2.0, dtype=torch.float64),
        torch.full((1, ac.h.n_or), 3.0, dtype=torch.float64),
        torch.tensor([8.0], dtype=torch.float64),
        torch.tensor([7.0], dtype=torch.float64),
        torch.tensor([0.2], dtype=torch.float64),
        temperature=1e-3,
    )
    freqs = torch.tensor([510.0], dtype=torch.float64)
    y, radiation, impedance = ac.shunts(el, freqs)
    order = ac.order_l
    position = {node: i for i, node in enumerate(order)}
    matrix = torch.diag(y[0, 0, list(order)])
    for j, (a, b) in enumerate(zip(ac.h.topo.a, ac.h.topo.b, strict=True)):
        if j == ac.h.glottis_port or a not in position or b not in position:
            continue
        i, k = position[a], position[b]
        conductance = el.open[0, j] / impedance[0, 0, j]
        matrix[i, i] += conductance
        matrix[k, k] += conductance
        matrix[i, k] -= conductance
        matrix[k, i] -= conductance
    expected = torch.linalg.solve(matrix, radiation[0, 0, list(order)])
    nodes, dipoles = ac.responses(el, freqs)
    torch.testing.assert_close(nodes[0, 0, list(order)], expected, rtol=1e-11, atol=1e-12)
    assert dipoles[0, 0, ac.h.glottis_port] == 0
    assert (dipoles.abs() > 0).any()


def test_soft_velum_reopening_gradient_is_finite_and_nonzero() -> None:
    ac = HumanAcoustics(build_human())
    velum = torch.tensor([-0.2, -1e-4, 0.0, 2e-4, 0.01], dtype=torch.float64, requires_grad=True)
    batch = velum.numel()
    el = ac.elements(
        torch.full((batch, ac.h.n_ph), 2.0, dtype=torch.float64),
        torch.full((batch, ac.h.n_or), 3.0, dtype=torch.float64),
        torch.full((batch,), 8.0, dtype=torch.float64),
        torch.full((batch,), 7.0, dtype=torch.float64),
        velum,
        temperature=1e-3,
    )
    ac.transfer(el, torch.tensor([610.0, 1700.0], dtype=torch.float64)).abs().sum().backward()
    assert velum.grad.isfinite().all() and (velum.grad != 0).all()
