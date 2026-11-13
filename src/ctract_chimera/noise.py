"""Local turbulence primitives from airway/network.rs::compute_noise.

These primitives consume a FIXED white realisation. They do not invent a
plosive envelope: CHIMERA's bursts arise from pressure release in its dynamic
network. Automatic constriction/teeth selection is deliberately not implied.
"""

import math

import torch
from torch import Tensor

from .relaxation import lower_bound, positive, strict_opening
from .topology import CLOSED_AREA, MU, RHO

BANDS = {"aspiration": (3500.0, 300.0), "constriction": (2600.0, 350.0), "teeth": (5000.0, 1200.0)}
GAINS = {"aspiration": 3.2e-6, "constriction": 2.2e-6, "teeth": 3.0e-6}


def band_noise(white: Tensor, fs: float, lowpass_hz: Tensor, highpass_hz: Tensor) -> Tensor:
    """Causal HP + two LP recurrences from compute_noise, initially at rest.

    white: (...,L), normally unit variance; frequencies are Hz. Unlike a circular
    FFT filter, this has CHIMERA's exact zero initial conditions and no wrap.
    This function applies neither the source gain nor sample-rate normalisation.
    """
    hp_coefficient = 1 - torch.exp(-2 * torch.pi * highpass_hz / fs)
    lp_coefficient = 1 - torch.exp(-2 * torch.pi * lowpass_hz / fs)
    hp = torch.zeros_like(white[..., 0])
    lp1, lp2 = hp, hp
    samples = []
    for w in white.unbind(-1):
        hp = hp + hp_coefficient * (w - hp)
        lp1 = lp1 + lp_coefficient * (w - hp - lp1)
        lp2 = lp2 + lp_coefficient * (lp1 - lp2)
        samples.append(lp2)
    return torch.stack(samples, -1)


def turbulence_amplitude(
    area: Tensor, flow: Tensor, gain: Tensor, fs: float = 96000.0, temperature: float | Tensor | None = None
) -> Tensor:
    """CHIMERA Reynolds pressure amplitude (dyn/cm²), with supplied gain.

    area: cm², flow: cm³/s. Positive dimensionless temperature relaxes area
    opening by tau*closed_area and Reynolds excess by tau*1800². Algebraic
    tails retain deep-threshold derivatives; None uses the exact Rust rules.
    """
    safe = (
        area.clamp_min(CLOSED_AREA)
        if temperature is None
        else lower_bound(area, CLOSED_AREA, temperature * CLOSED_AREA)
    )
    re2 = 4 * flow.square() / ((MU / RHO) ** 2 * torch.pi * safe)
    excess = re2 - 1800.0**2
    saturation = 12000.0**2 - 1800.0**2
    if temperature is None:
        excess = excess.clamp_min(0)
        gate = (area > CLOSED_AREA).to(area.dtype)
    else:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        excess = positive(excess, temperature * 1800.0**2)
        gate = strict_opening(area - CLOSED_AREA, temperature * CLOSED_AREA, CLOSED_AREA)
    return gain * math.sqrt(96000 / fs) * excess / (1 + excess / saturation) * gate
