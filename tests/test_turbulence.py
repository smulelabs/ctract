"""Noise localisation, registration/RNG, gain/filter gradients and Rust bursts."""

import json
from pathlib import Path

import torch
from ctract_chimera import CoupledSynth, ExactGeometry, HumanAcoustics, SourceRandomness, build_human
from ctract_chimera.turbulence import NoiseInputs, Turbulence


def fixture(name: str) -> dict:
    return json.loads((Path(__file__).parent / "fixtures" / f"{name}.json").read_text())


def test_all_three_noise_sources_and_moving_burst_vs_rust() -> None:
    model = CoupledSynth()
    for name in ["noisy", "burst", "noise_gesture"]:
        row = fixture(name)
        articulators = row["artic_frames"] if name != "noisy" else [row["artic"]]
        samples = len(row["audio"])
        noise = NoiseInputs.chimera(samples) if name != "burst" else None
        result = model.simulate(
            torch.tensor([articulators], dtype=torch.float64),
            torch.tensor([row["parameters"]], dtype=torch.float64),
            torch.tensor([row["pressure"]], dtype=torch.float64),
            SourceRandomness.chimera(samples, 96000.0),
            samples,
            row.get("frame_size", 1),
            temperature=None,
            noise=noise,
        )
        reference = torch.tensor([row["audio"]], dtype=torch.float64)
        assert (result.audio - reference).norm() / reference.norm() < 1e-9
        if name != "burst":
            assert result.pressure_sources.abs().max() > 1
        else:
            assert result.moving_flow.abs().max() > 1


def test_noise_true_minima_and_teeth_match_rust() -> None:
    row = fixture("noisy")
    frame = ExactGeometry().frame(torch.tensor([row["artic"]], dtype=torch.float64))
    for key in ["ph_min", "or_min", "teeth_s"]:
        torch.testing.assert_close(frame[key], torch.tensor([row[key]], dtype=torch.float64), rtol=1e-12, atol=1e-12)
    model = Turbulence(HumanAcoustics(build_human()), NoiseInputs.chimera(8), None)
    p = model.placement(frame)
    assert torch.equal(p.registered, torch.ones_like(p.registered))
    for j in range(2):
        expected = row["noise_specs"][j + 1]
        index = expected["src"] * model.e + expected["fwd"]
        assert p.identity[0, j, index] == 1
    con_gain = (p.constriction * p.area_gain[..., None]).sum() * 2.2e-6
    teeth_gain = (p.teeth * p.obstacle_gain[..., None]).sum() * 3e-6
    torch.testing.assert_close(
        con_gain, torch.tensor(row["noise_specs"][1]["gain"], dtype=torch.float64), rtol=1e-12, atol=1e-15
    )
    torch.testing.assert_close(
        teeth_gain, torch.tensor(row["noise_specs"][2]["gain"], dtype=torch.float64), rtol=1e-12, atol=1e-15
    )


def test_localisation_cold_limit_and_tie_first_choice() -> None:
    row = fixture("noisy")
    frame = ExactGeometry().frame(torch.tensor([row["artic"]], dtype=torch.float64))
    frame["ph_min"] = torch.full_like(frame["ph_min"], 2.0)
    frame["or_min"] = torch.full_like(frame["or_min"], 2.0)
    frame["or_min"][:, 6] = 0.05
    frame["teeth_s"] = frame["teeth_s"] * 0 + 13.5
    hard = Turbulence(HumanAcoustics(build_human()), NoiseInputs.chimera(8), None).placement(frame)
    cold = Turbulence(HumanAcoustics(build_human()), NoiseInputs.chimera(8), 1e-8).placement(frame)
    for attr in ["constriction", "teeth", "teeth_branch", "registered"]:
        torch.testing.assert_close(getattr(cold, attr), getattr(hard, attr), rtol=1e-5, atol=1e-6)
    frame["or_min"][:, 7] = 0.05
    cold = Turbulence(HumanAcoustics(build_human()), NoiseInputs.chimera(8), 1e-8).placement(frame)
    assert cold.constriction.sum(-1)[0, 18 + 6] > 0.999


def test_localisation_gains_and_band_gradcheck() -> None:
    row = fixture("noisy")
    frame = ExactGeometry().frame(torch.tensor([row["artic"]], dtype=torch.float64))
    minima = torch.cat((frame["ph_min"], frame["or_min"]), -1).requires_grad_()
    teeth = frame["teeth_s"].clone().requires_grad_()
    inputs = NoiseInputs.chimera(8)
    gains = inputs.gains.clone().requires_grad_()
    bands = inputs.bands.clone().requires_grad_()
    ac = HumanAcoustics(build_human())
    branch_area = torch.full((1, len(ac.h.topo.a)), 0.1, dtype=torch.float64)
    flow = torch.full_like(branch_area, 300.0)

    def forward(area: torch.Tensor, tooth: torch.Tensor, g: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        local = {**frame, "ph_min": area[:, : ac.h.n_ph], "or_min": area[:, ac.h.n_ph :], "teeth_s": tooth}
        model = Turbulence(ac, NoiseInputs(inputs.white, g, b), 0.05)
        p = model.placement(local)
        state = model.control(model.initial(), p)
        values = []
        for _ in range(4):
            pressure, state = model.step(state, p, branch_area, flow)
            values.append(pressure)
        return torch.stack(values, -1)

    assert torch.autograd.gradcheck(forward, (minima, teeth, gains, bands), eps=1e-8, fast_mode=True)
    gradients = torch.autograd.grad(forward(minima, teeth, gains, bands).square().sum(), (minima, teeth, gains, bands))
    assert all(g.isfinite().all() and (g != 0).all() for g in gradients)


def test_noise_registration_uses_one_stream_and_resets_identity() -> None:
    row = fixture("noisy")
    frame = ExactGeometry().frame(torch.tensor([row["artic"]], dtype=torch.float64))
    ac = HumanAcoustics(build_human())
    model = Turbulence(ac, NoiseInputs.chimera(8), None)
    p = model.placement(frame)
    state = model.control(model.initial(), p)
    area = torch.full((1, len(ac.h.topo.a)), 0.1, dtype=torch.float64)
    flow = torch.full_like(area, 300.0)
    _, state = model.step(state, p, area, flow)
    assert state.cursor.item() == 3
    frame["ph_min"] = torch.full_like(frame["ph_min"], 2.0)
    frame["or_min"] = torch.full_like(frame["or_min"], 2.0)
    p = model.placement(frame)
    state = model.control(state, p)
    assert (state.hp[:, 1:] == 0).all()
    _, state = model.step(state, p, area, flow)
    assert state.cursor.item() == 4


def test_full_noise_and_moving_readout_gradcheck() -> None:
    row = fixture("noisy")
    model = CoupledSynth(block_size=32, ir_length=64, response_fft=128)
    pose = torch.tensor([[row["artic"], row["artic"], row["artic"]]], dtype=torch.float64)
    pose[:, 1, 7] += 0.01
    pose.requires_grad_()
    parameters = torch.tensor([row["parameters"]], dtype=torch.float64)
    parameters[0, 10] = 10.0  # Vibrato rate is physically inactive at zero depth.
    parameters[0, 13:15] = 0.1
    parameters[0, 16] = 0.2
    parameters.requires_grad_()
    pressure = torch.tensor([row["pressure"]], dtype=torch.float64, requires_grad=True)
    anatomy = torch.tensor([1.0, 1.0, 2.25, 4.6], dtype=torch.float64, requires_grad=True)
    inputs = NoiseInputs.chimera(32)
    gains = inputs.gains.clone().requires_grad_()
    # Check cutoff directions in units of 10 kHz. A 1e-8 Hz perturbation
    # changes this ~1e3-amplitude moving-wall output below fp64 roundoff.
    # Rescaling the test input preserves the exact API/chain-rule Jacobian.
    bands = (inputs.bands / 10000).clone().requires_grad_()
    randomness = SourceRandomness.chimera(32, 96000.0, cycles=4)

    def forward(
        a: torch.Tensor, p: torch.Tensor, lung: torch.Tensor, body: torch.Tensor, g: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        return model(a, p, lung, randomness, 32, 16, anatomy=body, noise=NoiseInputs(inputs.white, g, b * 10000))

    args = (pose, parameters, pressure, anatomy, gains, bands)
    assert torch.autograd.gradcheck(forward, args, eps=1e-8, fast_mode=True)
    gradients = torch.autograd.grad(forward(*args).square().sum(), args)
    assert all(g.isfinite().all() and (g != 0).all() for g in gradients)
