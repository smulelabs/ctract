"""Convolution identities, local gradients and actual CHIMERA time-domain audio."""

import importlib.util
import json
from pathlib import Path

import torch
from ctract_chimera import NEUTRAL, ExactGeometry, Renderer
from ctract_chimera.dynamics import AirwayDynamics


def neutral_geometry() -> dict[str, torch.Tensor]:
    pose = torch.tensor([NEUTRAL], dtype=torch.float64)
    return {k: v[:, None] for k, v in ExactGeometry()(pose).items()}


def test_stationary_block_partition_invariance() -> None:
    generator = torch.Generator().manual_seed(13)
    source = torch.randn(1, 96, dtype=torch.float64, generator=generator)
    g = neutral_geometry()
    a = Renderer(block_size=32, ir_length=128, response_fft=256)(source, g, 1)
    b = Renderer(block_size=96, ir_length=128, response_fft=256)(source, g, 1)
    torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-9)


def test_renderer_flow_pressure_and_placement_gradcheck() -> None:
    renderer = Renderer(block_size=8, ir_length=32, response_fft=64)
    g = neutral_geometry()
    source = torch.linspace(0.1, 1, 8, dtype=torch.float64).reshape(1, 8).requires_grad_()
    pressure = torch.linspace(0.02, 0.03, 8, dtype=torch.float64).reshape(1, 1, 8).requires_grad_()
    placement = torch.full((1, 1, 1, len(renderer.acoustics.h.topo.a)), 0.01, dtype=torch.float64, requires_grad=True)

    def render(x: torch.Tensor, p: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return renderer(x, g, 1, pressure_sources=p, placement=w)

    assert torch.autograd.gradcheck(render, (source, pressure, placement), fast_mode=True)
    render(source, pressure, placement).square().sum().backward()
    assert source.grad.abs().sum() > 0 and pressure.grad.abs().sum() > 0 and placement.grad.abs().sum() > 0


def test_local_articulation_to_audio_gradcheck() -> None:
    model = ExactGeometry()
    generator = torch.Generator().manual_seed(73)
    a = model.lo + (model.hi - model.lo) * (0.2 + 0.6 * torch.rand(1, 17, dtype=torch.float64, generator=generator))
    a.requires_grad_()
    source = torch.linspace(0.01, 0.1, 8, dtype=torch.float64).reshape(1, 8).requires_grad_()
    renderer = Renderer(block_size=8, ir_length=32, response_fft=64)

    def render(pose: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        g = {k: v[:, None] for k, v in model(pose, mode="fit").items()}
        return renderer(flow, g, 1, temperature=1e-4)

    assert torch.autograd.gradcheck(render, (a, source), fast_mode=True)


def test_stationary_actual_rust_audio() -> None:
    row = json.loads((Path(__file__).parent / "fixtures/stationary.json").read_text())
    source, reference = (torch.tensor([row[k]], dtype=torch.float64) for k in ("source", "audio"))
    rendered = Renderer(block_size=2048, ir_length=4096, response_fft=16384)(source, neutral_geometry(), 1)
    # No fitted gain or alignment. Uses Rust's discrete clock and native-rate
    # derivative; the bounds catch a clock or derivative mismatch.
    wave_error = (rendered[:, 256:] - reference[:, 256:]).norm() / reference[:, 256:].norm()
    interior_error = (rendered[:, 256:-256] - reference[:, 256:-256]).norm() / reference[:, 256:-256].norm()
    window = torch.hann_window(source.shape[-1], dtype=torch.float64)
    spectrum = lambda x: torch.fft.rfft(x * window).abs()  # noqa: E731
    spectral_error = (spectrum(rendered) - spectrum(reference)).norm() / spectrum(reference).norm()
    assert wave_error < 0.005
    assert spectral_error < 0.001
    assert interior_error < 0.001


def test_native_rate_rust_audio_without_resampling_boundary_effects() -> None:
    row = json.loads((Path(__file__).parent / "fixtures/stationary_native.json").read_text())
    source, reference = (torch.tensor([row[k]], dtype=torch.float64) for k in ("source", "audio"))
    rendered = Renderer(sample_rate=96000.0, block_size=16384, ir_length=32768, response_fft=65536)(
        source, neutral_geometry(), 1
    )
    # Full native-rate impulse support removes the projection/truncation floor.
    # The tiny remaining error includes the nonlinear loss absent in a linear
    # transfer, and is orders of magnitude below a derivative-clock error.
    assert (rendered - reference).norm() / reference.norm() < 1e-6


def test_native_pressure_impulse_matches_time_stepper_without_alignment() -> None:
    """Current-step branch pressure must not inherit the old-node-pressure delay.

    A small impulse isolates the linear response from Bernoulli losses. The
    long native-rate IR suppresses truncation error; compare the entire clip,
    without cropping, shifting or fitting gain, so a one-sample delay fails.
    """
    samples = 1024
    frame = ExactGeometry().frame(torch.tensor([NEUTRAL], dtype=torch.float64))
    geometry = {key: value[:, None] for key, value in frame.items()}
    dynamics = AirwayDynamics()
    h = dynamics.ac.h
    branch = list(zip(h.topo.a, h.topo.b, strict=True)).index((h.oral.last - 1, h.oral.last))
    routing = torch.nn.functional.one_hot(torch.tensor(branch), len(h.topo.a)).to(torch.float64)
    flow = torch.zeros(1, samples, dtype=torch.float64)
    pressure = torch.zeros(1, 1, samples, dtype=torch.float64)
    pressure[0, 0, 0] = 1e-6  # dyn/cm², applied to the last internal oral branch.
    state = dynamics.initial(flow, frame, temperature=None)
    zero = flow[:, 0]  # Closed glottis and zero lung pressure: no other excitation.
    reference = []
    for sample in range(samples):
        audio, state = dynamics.step(
            state,
            frame,
            zero,
            zero,
            temperature=None,
            pressure_sources=pressure[:, 0, sample, None] * routing[None],
        )
        reference.append(audio)
    reference = torch.stack(reference, -1)
    rendered = Renderer(sample_rate=96000.0, block_size=samples, ir_length=32768, response_fft=65536)(
        flow, geometry, 1, pressure_sources=pressure, placement=routing[None, None, None]
    )
    relative_error = (rendered - reference).norm() / reference.norm()
    assert relative_error < 1e-10, f"Unaligned pressure-impulse relative error: {relative_error.item():.12g}"


def test_synthetic_fit_converges() -> None:
    # Load the shipped example itself, without depending on the repo's import
    # path or copying the fitting loop into the test.
    path = Path(__file__).parents[1] / "examples/fit_synthetic.py"
    spec = importlib.util.spec_from_file_location("fit_synthetic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.fit()
    assert result.final_loss < result.initial_loss * 1e-5
    expected = {"jaw": 0.6, "lip_aperture": 0.7, "larynx": 0.0, "flow_gain": 1.0}
    for name, value in expected.items():
        assert abs(result.parameters[name] - value) < 0.01
