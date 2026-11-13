"""Rust source/pressure-feedback parity and genuine recurrence derivatives."""

import json
from pathlib import Path

import torch
from ctract_chimera import NEUTRAL
from ctract_chimera.coupled import CoupledSynth
from ctract_chimera.glottis import GlottisState, ParametricGlottis
from ctract_chimera.source_random import SourceRandomness


def fixture() -> dict:
    return json.loads((Path(__file__).parent / "fixtures/coupled.json").read_text())


def test_hard_glottis_and_coupled_airway_vs_rust() -> None:
    row = fixture()
    parameters = torch.tensor([row["parameters"]], dtype=torch.float64)
    samples = len(row["audio"])
    randomness = SourceRandomness.chimera(samples, 96000.0)
    source = ParametricGlottis(parameters, 96000.0, randomness, None)
    state = GlottisState.initial(parameters[:, 0])
    area = []
    for i in range(samples):
        value, state = source.step(state, torch.tensor([7000.0], dtype=torch.float64), i)
        area.append(value)
    torch.testing.assert_close(
        torch.stack(area, -1), torch.tensor([row["source_area"]], dtype=torch.float64), rtol=1e-11, atol=1e-13
    )
    model = CoupledSynth()
    result = model.simulate(
        torch.tensor([[NEUTRAL]], dtype=torch.float64),
        parameters,
        torch.tensor([8.0], dtype=torch.float64),
        randomness,
        samples,
        1,
        temperature=None,
    )
    for got, key in [(result.audio, "audio"), (result.flow, "glottal_flow"), (result.glottal_area, "glottal_area")]:
        torch.testing.assert_close(got, torch.tensor([row[key]], dtype=torch.float64), rtol=1e-10, atol=1e-8)


def test_glottis_recurrence_gradcheck() -> None:
    parameters = torch.tensor([fixture()["parameters"]], dtype=torch.float64, requires_grad=True)
    pressure = torch.tensor([7000.0], dtype=torch.float64, requires_grad=True)
    randomness = SourceRandomness.chimera(16, 96000.0, cycles=4)

    def forward(params: torch.Tensor, dp: torch.Tensor) -> torch.Tensor:
        source = ParametricGlottis(params, 96000.0, randomness, 0.05)
        state = GlottisState.initial(dp)
        values = []
        for sample in range(16):
            area, state = source.step(state, dp, sample)
            values.append(area)
        return torch.stack(values, -1)

    assert torch.autograd.gradcheck(forward, (parameters, pressure), fast_mode=True)
    derivative = torch.autograd.grad(forward(parameters, pressure).sum(), (parameters, pressure))
    assert all(x.isfinite().all() and (x != 0).all() for x in derivative)


def test_pressure_coupling_through_convolution_gradcheck() -> None:
    model = CoupledSynth(block_size=16, ir_length=32, response_fft=64)
    parameters = torch.tensor([fixture()["parameters"]], dtype=torch.float64, requires_grad=True)
    pressure = torch.tensor([8.0], dtype=torch.float64, requires_grad=True)
    pose = torch.tensor([[NEUTRAL]], dtype=torch.float64, requires_grad=True)
    randomness = SourceRandomness.chimera(16, 96000.0, cycles=4)

    def forward(a: torch.Tensor, p: torch.Tensor, lung: torch.Tensor) -> torch.Tensor:
        return model(a, p, lung, randomness, 16, 1)

    assert torch.autograd.gradcheck(forward, (pose, parameters, pressure), fast_mode=True)
    derivative = torch.autograd.grad(forward(pose, parameters, pressure).square().sum(), (pose, parameters, pressure))
    assert all(x.isfinite().all() and (x != 0).all() for x in derivative)


def test_cycle_boundary_forward_derivatives() -> None:
    parameters = torch.tensor([fixture()["parameters"]], dtype=torch.float64, requires_grad=True)
    randomness = SourceRandomness.chimera(32, 96000.0, cycles=4)
    source = ParametricGlottis(parameters, 96000.0, randomness, 0.05)
    initial = GlottisState.initial(parameters[:, 0])
    _, advanced = source.step(initial, torch.tensor([7000.0], dtype=torch.float64), 0)
    crossing = 1.0 - advanced.phase.item()  # Construct test data, not a forward-path stop-gradient.
    for offset in [-1e-5, 0.0, 1e-5]:
        phase = torch.tensor([crossing + offset], dtype=torch.float64, requires_grad=True)

        def evaluate(p: torch.Tensor, initial_phase: torch.Tensor) -> torch.Tensor:
            source = ParametricGlottis(p, 96000.0, randomness, 0.05)
            state = GlottisState.initial(initial_phase)
            state.phase = initial_phase
            state.pressure = torch.ones_like(initial_phase) * 7000
            state.voice = torch.ones_like(initial_phase) * 0.8
            state.amplitude = torch.ones_like(initial_phase) * 0.1
            output = []
            for i in range(8):
                area, state = source.step(state, torch.ones_like(initial_phase) * 7000, i)
                output.append(area)
            return torch.stack(output, -1)

        assert torch.autograd.gradcheck(evaluate, (parameters, phase), eps=1e-8, fast_mode=True)
        grad = torch.autograd.grad(evaluate(parameters, phase).sum(), (parameters, phase))
        assert all(g.isfinite().all() and (g != 0).all() for g in grad)
        assert all(g.abs().max() < 1e7 for g in grad)


def test_source_cold_limit_and_pressure_threshold_sweep() -> None:
    row = fixture()
    p = torch.tensor([row["parameters"]], dtype=torch.float64)
    random = SourceRandomness.chimera(384, 96000.0)
    model = ParametricGlottis(p, 96000.0, random, 1e-8)
    state = GlottisState.initial(p[:, 0])
    values = []
    for i in range(384):
        area, state = model.step(state, torch.tensor([7000.0], dtype=torch.float64), i)
        values.append(area)
    torch.testing.assert_close(
        torch.stack(values, -1), torch.tensor([row["source_area"]], dtype=torch.float64), rtol=1e-4, atol=1e-7
    )
    for value in [0.0, 1961.33 - 1.0, 1961.33, 1961.33 + 1.0, 3922.66, 7000.0]:
        dp = torch.tensor([value], dtype=torch.float64, requires_grad=True)
        model = ParametricGlottis(p, 96000.0, random, 0.05)
        state = GlottisState.initial(dp)
        state.phase = dp * 0 + 0.2
        state.pressure = dp
        state.amplitude = dp * 0 + 0.1
        area, _ = model.step(state, dp, 0)
        gradient = torch.autograd.grad(area.sum(), dp)[0]
        assert gradient.isfinite().all() and (gradient != 0).all()


def test_source_all_parameter_gradients_over_several_real_cycles() -> None:
    parameters = torch.tensor([fixture()["parameters"]], dtype=torch.float64, requires_grad=True)
    pressure = torch.tensor([7000.0], dtype=torch.float64, requires_grad=True)
    random = SourceRandomness.chimera(1024, 96000.0, cycles=8)
    model = ParametricGlottis(parameters, 96000.0, random, 0.05)
    state = GlottisState.initial(pressure)
    energy = parameters.sum() * 0
    for i in range(1024):
        area, state = model.step(state, pressure, i)
        energy = energy + area.square().sum()
    gradient = torch.autograd.grad(energy, (parameters, pressure))
    assert all(g.isfinite().all() and (g != 0).all() for g in gradient)
    assert all(g.abs().max() < 1e7 for g in gradient)
