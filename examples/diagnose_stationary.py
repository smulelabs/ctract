"""Reproduce the stationary phase diagnosis on PUBLIC SYNTHETIC Rust audio.

Only the synthetic Rust fixture is read. Alternative clock
conventions are spelled out here for comparison; they are not renderer options.
"""

import json
from pathlib import Path

import torch
from ctract_chimera import ExactGeometry, Renderer


def diagnose() -> dict[str, object]:
    """Return error attribution and fractional alignment, without fitting gain."""
    row = json.loads((Path(__file__).parents[1] / "tests/fixtures/stationary.json").read_text())
    x, reference = (torch.tensor([row[k]], dtype=torch.float64) for k in ("source", "audio"))
    geometry = ExactGeometry()(torch.tensor([row["artic"]], dtype=torch.float64))
    renderer = Renderer(block_size=2048, ir_length=4096, response_fft=16384)
    ac = renderer.acoustics
    el = ac.elements(*(geometry[k] for k in ("ph", "orl", "ph_len", "or_len", "velum")))
    continuous = ac.transfer(el, renderer.freqs)
    fraction = renderer.freqs / 16000
    correction = torch.exp(1j * torch.pi * fraction * (1 - 16000 / ac.fs))
    correction = correction * torch.sinc(renderer.freqs / ac.fs) / torch.sinc(fraction)
    length = x.shape[-1]

    def convolve(response: torch.Tensor) -> torch.Tensor:
        impulse = torch.fft.irfft(response, n=16384)[..., :4096]
        n = 8192
        return torch.fft.irfft(torch.fft.rfft(x, n=n) * torch.fft.rfft(impulse, n=n), n=n)[..., :length]

    flow = convolve(continuous)
    legacy = (flow - torch.nn.functional.pad(flow[:, :-1], (1, 0))) * 16000
    corrected_flow = convolve(continuous * correction)
    corrected_derivative = (corrected_flow - torch.nn.functional.pad(corrected_flow[:, :-1], (1, 0))) * 16000
    corrected = renderer(x, {k: v[:, None] for k, v in geometry.items()}, 1)
    window = torch.hann_window(length, dtype=torch.float64)

    def errors(audio: torch.Tensor) -> dict[str, float]:
        spectrum = torch.fft.rfft(audio * window).abs()
        target = torch.fft.rfft(reference * window).abs()
        return {
            "waveform": ((audio[:, 256:] - reference[:, 256:]).norm() / reference[:, 256:].norm()).item(),
            "spectral": ((spectrum - target).norm() / target.norm()).item(),
        }

    def align(audio: torch.Tensor, delay: float) -> torch.Tensor:
        # Fractional advance through a zero-padded bandlimited interpolator.
        # This is diagnosis ONLY: no latency is fitted inside the library.
        frequencies = torch.fft.rfftfreq(8192, 1 / 16000, dtype=torch.float64)
        return torch.fft.irfft(torch.fft.rfft(audio, n=8192) * torch.exp(2j * torch.pi * frequencies * delay), n=8192)[
            ..., :length
        ]

    delays = torch.linspace(-0.75, 0.75, 301, dtype=torch.float64) / 16000

    def aligned_error(audio: torch.Tensor) -> dict[str, float]:
        # Exclude BOTH edges when estimating alignment, so finite-clip padding
        # cannot masquerade as a physical transport delay.
        target = reference[:, 256:-256]
        scores = torch.stack(
            [(align(audio, delay.item())[:, 256:-256] - target).norm() / target.norm() for delay in delays]
        )
        index = int(scores.argmin())  # Offline diagnostic; never a fitted forward operation.
        return {"delay_us": delays[index].item() * 1e6, "aligned_waveform": scores[index].item()}

    result = {
        name: {**errors(value), **aligned_error(value)}
        for name, value in [
            ("legacy", legacy),
            ("internal_derivative_only", corrected_derivative),
            ("discrete_renderer", corrected),
        ]
    }
    result["predicted_derivative_delay_us"] = 0.5 * (1 / 16000 - 1 / 96000) * 1e6
    sweep = []
    for impulse_length in [1024, 2048, 4096, 8192]:
        r = Renderer(block_size=2048, ir_length=impulse_length, response_fft=max(16384, 2 * impulse_length))
        audio = r(x, {k: v[:, None] for k, v in geometry.items()}, 1)
        sweep.append({"ir_samples": impulse_length, **errors(audio)})
    native = json.loads((Path(__file__).parents[1] / "tests/fixtures/stationary_native.json").read_text())
    native_source, native_ref = (torch.tensor([native[k]], dtype=torch.float64) for k in ("source", "audio"))
    native_audio = Renderer(sample_rate=96000.0, block_size=16384, ir_length=32768, response_fft=65536)(
        native_source, {k: v[:, None] for k, v in geometry.items()}, 1
    )
    result["native_full_clip_waveform"] = ((native_audio - native_ref).norm() / native_ref.norm()).item()
    # Only this known synthetic formula is continued. The library never
    # extrapolates a caller's waveform or invents future source samples.
    time = torch.arange(4096, dtype=torch.float64) / 16000
    extended = (
        0.001
        * (1 - torch.exp(-time / 0.005))
        * (
            torch.sin(2 * torch.pi * 137 * time)
            + 0.3 * torch.sin(2 * torch.pi * 823 * time)
            + 0.1 * torch.sin(2 * torch.pi * 2173 * time)
        )
    )
    extended_audio = renderer(extended[None], {k: v[:, None] for k, v in geometry.items()}, 1)[:, :length]
    result["known_synthetic_continuation"] = errors(extended_audio)
    result["ir_sweep"] = sweep
    return result


if __name__ == "__main__":
    torch.set_num_threads(2)
    print(json.dumps(diagnose(), indent=2))
