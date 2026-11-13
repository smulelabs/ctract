"""ParametricGlottis::next (sources/glottis.rs), with continuous cycle events.

This produces AREA, cm². Network coupling is in dynamics.py; no area→flow
substitution occurs. Hard mode is a scalar-equation reference with exact cycle
bins. Positive temperature differentiates the actual smoothed forward process.
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .geometry_relaxation import Relaxation
from .relaxation import opening, strict_opening
from .source import glottal_shape
from .source_random import SourceRandomness

GLOTTIS_NAMES = (
    "f0",
    "amplitude",
    "abduction",
    "chink",
    "open_quotient",
    "skew",
    "closure_sharpness",
    "effort",
    "jitter",
    "shimmer",
    "vibrato_depth",
    "vibrato_rate",
    "drift",
    "fry",
    "diplophonia",
    "closure",
    "chaos",
)
DEFAULT_GLOTTIS = (110.0, 0.16, 0.0, 0.006, 0.62, 0.62, 0.6, 0.7, 0.004, 0.03, 0.0, 6.0, 6.0, 0.0, 0.0, 0.0, 0.0)
CM_H2O = 980.665


@dataclass
class GlottisState:
    """Differentiable recurrence state; no history is detached between steps."""

    phase: Tensor  # Unwrapped phase: avoids integer cycle selection in the smooth path.
    pressure: Tensor
    voice: Tensor
    static: Tensor
    closure: Tensor
    amplitude: Tensor
    vibrato_phase: Tensor
    pitch_factor: Tensor
    period_scale: Tensor

    @classmethod
    def initial(cls, like: Tensor) -> "GlottisState":
        zero, one = torch.zeros_like(like), torch.ones_like(like)
        return cls(zero, zero, zero, zero, zero, zero, zero, one, one)


class ParametricGlottis:
    """Constant per-clip GlottisParams, with pressure-dependent sample updates.

    Explicit fixed cycle randomness is supplied by the caller. Its number of
    cycles defines a domain of phase, checked by the enclosing simulation.
    Parameters are (B,17) in GLOTTIS_NAMES order. Shared constant controls match
    Rust; time-varying control interpolation is a separate modelling decision.
    """

    def __init__(
        self,
        params: Tensor,
        sample_rate: float,
        randomness: SourceRandomness,
        temperature: float | Tensor | None = 0.05,
    ) -> None:
        if params.ndim != 2 or params.shape[-1] != len(GLOTTIS_NAMES):
            raise ValueError("glottis parameters must be (B,17)")
        self.params, self.fs, self.temperature = params, sample_rate, temperature
        self.r = (
            None
            if temperature is None
            else Relaxation(params.new_tensor(temperature) if not isinstance(temperature, Tensor) else temperature)
        )
        self.randomness = randomness
        self.cycle_period, self.cycle_amplitude = self._cycles()

    def _clamp(self, x: Tensor, lo: float, hi: float) -> Tensor:
        return x.clamp(lo, hi) if self.r is None else self.r.clamp(x, lo, hi)

    def _positive(self, x: Tensor) -> Tensor:
        return x.clamp_min(0) if self.r is None else self.r.positive(x)

    def _ramp(self, lo: float, hi: float, x: Tensor) -> Tensor:
        u = self._clamp((x - lo) / (hi - lo), 0.0, 1.0)
        return u.square() * (3 - 2 * u)

    def _cycles(self) -> tuple[Tensor, Tensor]:
        """Per-cycle perturbations, retaining every logistic-map derivative."""
        p = self.params
        jitter, shimmer, diplo, chaos = p[:, 8], p[:, 9], p[:, 14], p[:, 16]
        x = torch.zeros_like(chaos) + 0.37
        period, amplitude = [torch.ones_like(x)], [torch.ones_like(x)]
        for i, noise in enumerate(self.randomness.cycles.to(p)):
            candidate = (2.9 + 1.1 * self._clamp(chaos, 0.0, 1.0)) * x * (1 - x)
            if self.r is None:
                x = torch.where(chaos > 0, torch.where((candidate > 0) & (candidate < 1), candidate, 0.37), x)
            else:
                valid = strict_opening(candidate, self.r.width, 1.0) * strict_opening(1 - candidate, self.r.width, 1.0)
                active = strict_opening(chaos, self.r.width, 1.0)
                x = x + (candidate - x) * active * valid + (0.37 - x) * active * (1 - valid)
            deviation = x - 0.5
            alt = 1.0 if i % 2 == 0 else -1.0
            period.append(self._clamp(1 + jitter * noise[0] + diplo * 0.08 * alt + 0.45 * chaos * deviation, 0.55, 1.8))
            amplitude.append(
                0.12
                + self._positive(1 + shimmer * noise[1] - diplo * 0.35 * (alt > 0) + 1.1 * chaos * deviation - 0.12)
            )
        return torch.stack(period, -1), torch.stack(amplitude, -1)

    def _lookup(self, phase: Tensor) -> tuple[Tensor, Tensor]:
        """Continuous cycle-bin partition; hard mode selects floor(phase)."""
        thresholds = torch.arange(1, self.cycle_period.shape[-1], dtype=phase.dtype, device=phase.device)
        if self.r is None:
            cdf = (phase[:, None] >= thresholds).to(phase.dtype)
        else:
            cdf = opening(phase[:, None] - thresholds, self.r.tau * 1e-4, 1.0)
        weights = torch.cat((1 - cdf[:, :1], cdf[:, :-1] - cdf[:, 1:], cdf[:, -1:]), -1)
        return (weights * self.cycle_period).sum(-1), (weights * self.cycle_amplitude).sum(-1)

    def _phase(self, unwrapped: Tensor) -> Tensor:
        if self.r is None:
            return torch.remainder(unwrapped, 1.0)
        # Abel-summed periodic sawtooth; positive denominator removes the atan2
        # branch cut. A vanishing positive phase bias gives Rust's phase=0 at
        # integer cycles, instead of the Fourier midpoint phase=.5.
        width = self.r.tau * 1e-4
        angle = 2 * torch.pi * (unwrapped + width.sqrt())
        damping = torch.exp(-width)
        return 0.5 - torch.atan2(damping * angle.sin(), 1 - damping * angle.cos()) / torch.pi

    def step(self, state: GlottisState, pressure: Tensor, sample: int) -> tuple[Tensor, GlottisState]:
        """Advance one sample using OLD transglottal pressure, as HumanVoice."""
        p = self.params
        f0, amp, abduction, chink, oq, skew, sharp, effort = p[:, :8].unbind(-1)
        vibrato_depth, vibrato_rate, drift, fry, closure = p[:, 10], p[:, 11], p[:, 12], p[:, 13], p[:, 15]
        dp = state.pressure + (1 - math.exp(-2 * math.pi * 30 / self.fs)) * (pressure - state.pressure)
        vib_phase, pitch = state.vibrato_phase, state.pitch_factor
        if sample % 32 == 0:  # Fixed source clock, never a fitted-value-dependent event.
            value = self.randomness.drift.to(p)[sample // 32]
            vib_phase = vib_phase + vibrato_rate * 32 / self.fs * (1 + 0.04 * value)
            cents = vibrato_depth * torch.sin(2 * torch.pi * vib_phase) + drift * value
            pitch = torch.exp(cents * (math.log(2) / 1200))
        f0 = f0 * pitch
        f0 = f0 * (1 - fry) + (f0 - self._positive(f0 - 48)) * fry
        target = self._ramp(2.5 * CM_H2O * 0.8, 2.5 * CM_H2O * 1.6, dp)
        target = target * (1 - self._ramp(0.08, 0.22, state.static - chink)) * (1 - state.closure)
        voice = state.voice + (1 - math.exp(-2 * math.pi * 25 / self.fs)) * (target - state.voice)
        unwrapped = state.phase + f0 / self.fs / state.period_scale
        period, cycle_amp = self._lookup(unwrapped)
        phase = self._phase(unwrapped)
        oq = self._clamp(oq - 0.18 * (effort - 0.5), 0.25, 0.98) * (1 - fry) + 0.28 * fry
        if self.r is None:
            pulse = glottal_shape(phase, oq, skew, sharp)
            second = torch.where(phase > 0.38, glottal_shape(phase - 0.38, oq * 0.8, skew, sharp), 0.0)
        else:

            def shape(ph: Tensor, open_q: Tensor) -> Tensor:
                t = ph / open_q
                s = self._clamp(skew, 0.3, 0.9)
                rise = 0.5 - 0.5 * torch.cos(torch.pi * t / s)
                u = (t - s) / (1 - s)
                fall = 0.5 + 0.5 * torch.cos(torch.pi * u)
                fall = fall + (torch.cos(0.5 * torch.pi * u) - fall) * sharp
                weight = self.r.gate(t - s)
                return (rise + (fall - rise) * weight) * (1 - self.r.gate(ph - open_q))

            pulse = shape(phase, oq)
            second = shape(phase - 0.38, oq * 0.8) * self.r.gate(phase - 0.38)
        pulse = pulse + 0.55 * self._ramp(0.3, 0.8, fry) * second
        coefficient = 1 - math.exp(-1 / (0.004 * self.fs))
        amp_sm = state.amplitude + coefficient * (amp * (0.45 + 0.75 * effort) - state.amplitude)
        closure_sm = state.closure + coefficient * (closure - state.closure)
        static_sm = state.static + coefficient * (abduction + chink - state.static)
        area = self._positive(amp_sm * cycle_amp * voice * pulse + static_sm * (1 - 0.92 * closure_sm))
        return area, GlottisState(unwrapped, dp, voice, static_sm, closure_sm, amp_sm, vib_phase, pitch, period)
