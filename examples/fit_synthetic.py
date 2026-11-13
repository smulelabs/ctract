"""Recover jaw, lip aperture, larynx height and source-flow gain from Rust audio.

The target is a known 24-tone synthetic ideal flow rendered by CHIMERA's actual
AirwayNetwork::step. The model fits the same physical controls using a spectral
loss and a gradient-based line search (L-BFGS variable-metric descent).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from ctract_chimera import ExactGeometry, Renderer


@dataclass
class FitResult:
    initial_loss: float
    final_loss: float
    parameters: dict[str, float]
    history: list[float]


def fit(steps: int = 40) -> FitResult:
    """Four-parameter spectral analysis-by-synthesis, CPU runnable.

    L-BFGS uses the exact autograd gradient through geometry and convolution;
    its line search handles the differently conditioned physical parameters.
    It is a descent optimiser, not a learned initializer or an alternate synth.
    The log flow-gain coordinate enforces a positive source amplitude.
    """
    row = json.loads((Path(__file__).parents[1] / "tests/fixtures/fit.json").read_text())
    source = torch.tensor([row["source"]], dtype=torch.float64)
    target = torch.tensor([row["audio"]], dtype=torch.float64)
    base = torch.tensor([row["artic"]], dtype=torch.float64)
    basis = torch.nn.functional.one_hot(torch.tensor([0, 7, 12]), 17).to(torch.float64)
    parameters = torch.tensor([0.85, 1.05, 0.2, 0.1], dtype=torch.float64, requires_grad=True)
    geometry = ExactGeometry()
    renderer = Renderer(block_size=2048, ir_length=4096, response_fft=16384)
    window = torch.hann_window(source.shape[-1], dtype=torch.float64)
    reference = torch.fft.rfft(target * window)
    normalizer = reference.abs().square().sum()
    optimizer = torch.optim.LBFGS(
        [parameters],
        lr=0.8,
        max_iter=steps,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
    )
    history = []

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        pose = base * (1 - basis.sum(0)) + parameters[:3] @ basis
        g = {key: value[:, None] for key, value in geometry(pose, mode="fit", temperature=0.05).items()}
        audio = renderer(source * parameters[3].exp(), g, 1, temperature=1e-5)
        spectrum = torch.fft.rfft(audio * window)
        # Magnitude constrains resonances/gain; a small phase term helps
        # distinguish correlated articulators after correcting the source clock.
        loss = (
            (spectrum.abs() - reference.abs()).square().sum() + 0.2 * (spectrum - reference).abs().square().sum()
        ) / normalizer
        loss.backward()
        history.append(loss.item())  # Reporting outside the differentiable forward calculation.
        return loss

    optimizer.step(closure)
    final = closure().item()
    values = {name: parameters[i].item() for i, name in enumerate(("jaw", "lip_aperture", "larynx"))}
    values["flow_gain"] = parameters[3].exp().item()
    return FitResult(history[0], final, values, history)


if __name__ == "__main__":
    torch.set_num_threads(2)
    result = fit()
    print(f"Spectral loss: {result.initial_loss:.6g} -> {result.final_loss:.6g}")
    print(json.dumps(result.parameters, indent=2))
