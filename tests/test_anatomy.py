"""Continuous landmarks, material elements and changing CHIMERA graph counts."""

import json
from pathlib import Path

import torch
from ctract_chimera import NEUTRAL, Anatomy, ExactGeometry, HumanAcoustics, build_human
from ctract_chimera.discretisation import AnatomicalRenderer, AnatomyDomain, count_weights
from ctract_chimera.topology import SectionCounts


def test_graph_weight_partition_and_inclusive_floor() -> None:
    for q in [17.9, 18.0, 18.1]:
        coordinate = torch.tensor(q, dtype=torch.float64, requires_grad=True)
        weights = count_weights(coordinate, (17, 18, 19), 0.05)
        assert weights.min() >= 0
        torch.testing.assert_close(weights.sum(), torch.tensor(1.0, dtype=torch.float64))
        assert torch.autograd.gradcheck(lambda z: count_weights(z, (17, 18, 19), 0.05), (coordinate,))
        cold = count_weights(coordinate, (17, 18, 19), 1e-8)
        expected = torch.tensor([1.0, 0.0, 0.0] if q < 18 else [0.0, 1.0, 0.0], dtype=torch.float64)
        torch.testing.assert_close(cold, expected, rtol=1e-6, atol=1e-6)


def test_continuous_geometry_and_elements_vs_custom_rust_anatomy() -> None:
    rows = json.loads((Path(__file__).parent / "fixtures/anatomy.json").read_text())
    for row in rows:
        body = torch.tensor(row["body"], dtype=torch.float64)
        pose = torch.tensor([row["artic"]], dtype=torch.float64)
        reference = build_human(96000.0, Anatomy(*row["body"]))
        counts = SectionCounts(reference.n_ph, reference.n_or, reference.trachea.n, reference.nasal.n)
        model = ExactGeometry(counts=counts)
        g = model(pose, mode="fit", temperature=1e-8, anatomy=body)
        for key in g:
            torch.testing.assert_close(g[key], torch.tensor([row[key]], dtype=torch.float64), rtol=1e-6, atol=1e-6)
        ac = HumanAcoustics(build_human(counts=counts))
        # Use the Rust geometry here to isolate physical constant scaling from
        # the independent geometry-relaxation approximation.
        el = ac.elements(
            *(torch.tensor([row[k]], dtype=torch.float64) for k in ("ph", "orl", "ph_len", "or_len", "velum")),
            anatomy=body,
        )
        for key in ("c", "wall_area", "rad_area"):
            torch.testing.assert_close(
                getattr(el, key), torch.tensor([row[key]], dtype=torch.float64), rtol=1e-12, atol=1e-15
            )
        for key in ("l", "r"):
            expected = torch.tensor([row[key]], dtype=torch.float64)
            torch.testing.assert_close(getattr(el, key)[el.open], expected[el.open], rtol=1e-12, atol=1e-15)


def test_anatomy_and_discretisation_end_to_end_gradcheck() -> None:
    renderer = AnatomicalRenderer(block_size=8, ir_length=32, response_fft=64)
    anatomy = torch.tensor([1.0, 1.0, 2.25, 4.6], dtype=torch.float64, requires_grad=True)
    articulation = torch.tensor([[NEUTRAL]], dtype=torch.float64)
    flow = torch.linspace(0.001, 0.002, 8, dtype=torch.float64)[None]

    def render(body: torch.Tensor) -> torch.Tensor:
        return renderer(articulation, body, flow, 1, temperature=0.05)

    assert torch.autograd.gradcheck(render, (anatomy,), fast_mode=True)
    gradient = torch.autograd.grad(render(anatomy).square().sum(), anatomy)[0]
    assert gradient.isfinite().all() and (gradient != 0).all()


def test_graph_crossing_anatomy_gradients() -> None:
    domain = AnatomyDomain(scale=(0.98, 1.04), pharynx=(0.99, 1.01))
    renderer = AnatomicalRenderer(domain, block_size=8, ir_length=32, response_fft=64)
    # Oral count changes from 18 to 19 at this exact scale. Evaluate on both
    # sides and the boundary without rebuilding or choosing a graph.
    crossing = 19 * (35000 / 96000 / 0.93) / 7.3
    for scale in [crossing - 0.0001, crossing, crossing + 0.0001]:
        anatomy = torch.tensor([scale, 1.0, 2.25, 4.6], dtype=torch.float64, requires_grad=True)
        result = renderer(
            torch.tensor([[NEUTRAL]], dtype=torch.float64),
            anatomy,
            torch.linspace(0.001, 0.002, 8, dtype=torch.float64)[None],
            1,
        )
        gradient = torch.autograd.grad(result.square().sum(), anatomy)[0]
        assert gradient.isfinite().all() and (gradient != 0).all()


def test_complete_source_noise_discretisation_atlas_gradcheck() -> None:
    from ctract_chimera import AnatomicalSynth, NoiseInputs, SourceRandomness

    row = json.loads((Path(__file__).parent / "fixtures/noisy.json").read_text())
    model = AnatomicalSynth(block_size=32, ir_length=64, response_fft=128)
    assert len(model.components) > 1
    pose = torch.tensor([[row["artic"]]], dtype=torch.float64, requires_grad=True)
    anatomy = torch.tensor([1.0, 1.0, 2.25, 4.6], dtype=torch.float64, requires_grad=True)
    parameters = torch.tensor([row["parameters"]], dtype=torch.float64)
    parameters[0, 10] = 10.0
    parameters[0, 13:15] = 0.1
    parameters[0, 16] = 0.2
    parameters.requires_grad_()
    pressure = torch.tensor([20.0], dtype=torch.float64, requires_grad=True)
    randomness = SourceRandomness.chimera(32, 96000.0, cycles=4)
    noise = NoiseInputs.chimera(32)
    gains = noise.gains.clone().requires_grad_()
    bands = (noise.bands / 10000).clone().requires_grad_()

    def render(
        a: torch.Tensor, body: torch.Tensor, p: torch.Tensor, lung: torch.Tensor, g: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        return model(a, body, p, lung, randomness, 32, 1, noise=NoiseInputs(noise.white, g, b * 10000))

    args = (pose, anatomy, parameters, pressure, gains, bands)
    assert torch.autograd.gradcheck(render, args, eps=1e-8, fast_mode=True)
    gradients = torch.autograd.grad(render(*args).square().sum(), args)
    assert all(g.isfinite().all() and (g != 0).all() for g in gradients)
