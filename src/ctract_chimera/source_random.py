"""CHIMERA math.rs::{Rng,Drift} realised BEFORE a differentiable simulation.

Random draws are indexed by physical cycle/control-clock events, not redrawn
inside backward. Integer PRNG arithmetic here is independent of fitted values.
It matches CHIMERA's bounded four-uniform 'gauss', not Box-Muller or torch.randn.
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor


class Random:
    """Exact mulberry32 wrapping arithmetic of math.rs::Rng."""

    def __init__(self, seed: int) -> None:
        self.state = seed or 1

    def uniform(self) -> float:
        self.state = (self.state + 0x6D2B79F5) & 0xFFFFFFFF
        t = self.state
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0

    def gauss(self) -> float:
        return (sum(self.uniform() for _ in range(4)) - 2) * 1.7320508


@dataclass(frozen=True)
class SourceRandomness:
    """Cycle perturbations (K,2), drift every 32 samples (C,), no hidden state."""

    cycles: Tensor
    drift: Tensor

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "SourceRandomness":
        """Move fixed realisations without introducing a stopped gradient."""
        return SourceRandomness(self.cycles.to(device=device, dtype=dtype), self.drift.to(device=device, dtype=dtype))

    @classmethod
    def chimera(cls, samples: int, sample_rate: float, cycles: int = 64) -> "SourceRandomness":
        """Realise CHIMERA's glottis seeds 777 and 4242, initially at rest."""
        if samples < 1 or cycles < 1 or not math.isfinite(sample_rate) or sample_rate <= 0:
            raise ValueError("random realisations need positive samples, cycles and finite sample rate")
        rng = Random(777)
        draws = torch.tensor([[rng.gauss(), rng.gauss()] for _ in range(cycles)], dtype=torch.float64)
        rng = Random(4242)
        coefficients = [math.exp(-2 * math.pi * f / (sample_rate / 32)) for f in (0.3, 0.9, 2.7)]
        state = [0.0, 0.0, 0.0]
        drift = []
        for _ in range((samples + 31) // 32):
            state = [a * y + math.sqrt(1 - a * a) * rng.gauss() for a, y in zip(coefficients, state, strict=True)]
            drift.append(sum(y / (i + 1) for i, y in enumerate(state)) / 1.4)
        return cls(draws, torch.tensor(drift, dtype=torch.float64))
