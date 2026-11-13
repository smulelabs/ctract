"""Continuous anatomy, including CHIMERA's changing integer tube discretisation.

Each discrete graph in an explicit anatomy domain is evaluated and its audio is
weighted by a smooth floor-bin partition. This is deliberately a reference
implementation: it does not pick a graph from fitted values or omit a low-weight
graph. The cold limit picks precisely Rust's graph.
"""

import math
from dataclasses import dataclass
from itertools import product

import torch
from torch import Tensor, nn

from .anatomy import Anatomy
from .coupled import CoupledSynth
from .geometry import ExactGeometry
from .relaxation import lower_bound, opening
from .renderer import Renderer
from .source_random import SourceRandomness
from .topology import C_SOUND, SectionCounts, build_human
from .turbulence import NoiseInputs


@dataclass(frozen=True)
class AnatomyDomain:
    """Finite graph-atlas domain, not a sex/age preset or anatomy estimate.

    A finite computational graph cannot contain the unbounded number of cells
    required by arbitrarily large anatomy. Choose this domain explicitly for
    the research question; increasing it adds graph components, not a prior.
    Radius and lip width do not affect counts and need no upper domain bound.
    """

    scale: tuple[float, float] = (0.98, 1.02)
    pharynx: tuple[float, float] = (0.98, 1.02)

    def __post_init__(self) -> None:
        for low, high in (self.scale, self.pharynx):
            if not all(math.isfinite(x) for x in (low, high)) or not 0 < low < high:
                raise ValueError("anatomy domain intervals must be positive, finite and ordered")

    def states(self, internal_rate: float) -> tuple[SectionCounts, ...]:
        """Enumerate all possible count tuples from monotone endpoint bounds."""
        corners = [build_human(internal_rate, Anatomy(k, ph)) for k, ph in product(self.scale, self.pharynx)]
        values = [(h.n_ph, h.n_or, h.trachea.n, h.nasal.n) for h in corners]
        ranges = [range(min(v[i] for v in values), max(v[i] for v in values) + 1) for i in range(4)]
        return tuple(SectionCounts(*v) for v in product(*ranges))


def count_coordinates(anatomy: Tensor, internal_rate: float, temperature: float | Tensor) -> Tensor:
    """Pre-floor coordinates from HumanAirway::new, with smooth max(dx_min,...)."""
    k, ph = anatomy[:2].unbind()
    dx_min = C_SOUND / internal_rate / 0.93
    width = torch.as_tensor(temperature, dtype=anatomy.dtype, device=anatomy.device) * 0.001
    return torch.stack(
        (
            (8.7 * k * (0.55 + 0.45 * ph) - 1.6 * k) / dx_min,
            7.3 * k / dx_min,
            12 * k / lower_bound(0.9 * k, dx_min, width),
            11.4 * k / lower_bound(0.5 * k, dx_min, width),
        )
    )


def count_weights(coordinate: Tensor, values: tuple[int, ...], temperature: float | Tensor) -> Tensor:
    """Floor-bin partition; the inclusive opening bias chooses floor(integer).

    End bins hold the remaining tail mass. The caller validates the anatomy
    domain, so that holding cannot hide an out-of-domain hard discretisation.
    """
    if len(values) == 1:
        return torch.ones_like(coordinate)[None]
    boundaries = coordinate.new_tensor(values[1:])
    cdf = opening(coordinate - boundaries, temperature, 1.0)
    return torch.cat((1 - cdf[:1], cdf[:-1] - cdf[1:], cdf[-1:]))


class GraphAtlas(nn.Module):
    """Common count-domain partition for prescribed-flow and coupled models."""

    def __init__(self, domain: AnatomyDomain, internal_rate: float) -> None:
        super().__init__()
        self.domain, self.internal_rate = domain, internal_rate
        self.states = domain.states(internal_rate)
        self.count_values = tuple(
            tuple(sorted({getattr(s, key) for s in self.states})) for key in ("pharynx", "oral", "trachea", "nasal")
        )

    def weights(self, anatomy: Tensor, temperature: float | Tensor) -> Tensor:
        """Joint continuous graph weights in the same order as ``states``."""
        coordinates = count_coordinates(anatomy, self.internal_rate, temperature)
        factors = [
            count_weights(q, values, temperature) for q, values in zip(coordinates, self.count_values, strict=True)
        ]
        # The product wiring depends only on the static atlas definition.
        return torch.stack(
            [
                factors[0][i] * factors[1][j] * factors[2][k] * factors[3][nasal]
                for i, j, k, nasal in product(*(range(len(v)) for v in self.count_values))
            ]
        )

    def validate(self, anatomy: Tensor) -> None:
        """Validate the physical and finite discretisation domain, without clipping."""
        if anatomy.shape != (4,) or not bool(torch.isfinite(anatomy).all() & (anatomy > 0).all()):
            raise ValueError("anatomy must be four finite positive scalars")
        if not (
            self.domain.scale[0] <= anatomy[0] <= self.domain.scale[1]
            and self.domain.pharynx[0] <= anatomy[1] <= self.domain.pharynx[1]
        ):
            raise ValueError("anatomy lies outside the explicit discretisation atlas domain")


class AnatomicalRenderer(GraphAtlas):
    """Continuous anatomy/discretisation with prescribed ideal flow (B,L)."""

    def __init__(
        self,
        domain: AnatomyDomain = AnatomyDomain(),
        sample_rate: float = 16000.0,
        internal_rate: float = 96000.0,
        block_size: int = 256,
        ir_length: int = 2048,
        response_fft: int = 8192,
    ) -> None:
        super().__init__(domain, internal_rate)
        self.geometry = nn.ModuleList(ExactGeometry(internal_fs=internal_rate, counts=s) for s in self.states)
        self.renderers = nn.ModuleList(
            Renderer(
                sample_rate=sample_rate,
                internal_rate=internal_rate,
                block_size=block_size,
                ir_length=ir_length,
                response_fft=response_fft,
                counts=s,
            )
            for s in self.states
        )

    def forward(
        self, artic: Tensor, anatomy: Tensor, flow: Tensor, frame_size: int, temperature: float | Tensor = 0.05
    ) -> Tensor:
        """Render clip with articulators (B,T,17), anatomy (4,), flow (B,L)."""
        self.validate(anatomy)
        weights = self.weights(anatomy, temperature)
        audio = []
        for weight, geometry, renderer in zip(weights, self.geometry, self.renderers, strict=True):
            g = geometry(artic, mode="fit", temperature=temperature, anatomy=anatomy)
            y = renderer(flow, g, frame_size, temperature=temperature * 2e-4, anatomy=anatomy)
            audio.append(weight * y)
        return torch.stack(audio).sum(0)


class AnatomicalSynth(GraphAtlas):
    """Complete source/noise/geometry/coupling/convolution over changing counts.

    Every component gets the same continuous anatomy/source controls and fixed
    random realisations. Its complete recurrent solve carries gradients; only
    output audio, not incompatible node-state arrays, is mixed across graphs.
    """

    def __init__(
        self,
        domain: AnatomyDomain = AnatomyDomain(),
        sample_rate: float = 96000.0,
        block_size: int = 256,
        ir_length: int = 2048,
        response_fft: int = 8192,
    ) -> None:
        super().__init__(domain, sample_rate)
        self.components = nn.ModuleList(
            CoupledSynth(
                sample_rate=sample_rate, counts=s, block_size=block_size, ir_length=ir_length, response_fft=response_fft
            )
            for s in self.states
        )

    def forward(
        self,
        articulators: Tensor,
        anatomy: Tensor,
        parameters: Tensor,
        pressure: Tensor,
        randomness: SourceRandomness,
        samples: int,
        frame_size: int,
        temperature: float | Tensor = 0.05,
        noise: NoiseInputs | None = None,
    ) -> Tensor:
        """Fully coupled differentiable graph atlas; source/pressure units as CoupledSynth."""
        self.validate(anatomy)
        weights = self.weights(anatomy, temperature)
        return torch.stack(
            [
                weight
                * component(
                    articulators, parameters, pressure, randomness, samples, frame_size, temperature, anatomy, noise
                )
                for weight, component in zip(weights, self.components, strict=True)
            ]
        ).sum(0)

    def time_domain(
        self,
        articulators: Tensor,
        anatomy: Tensor,
        parameters: Tensor,
        pressure: Tensor,
        randomness: SourceRandomness,
        samples: int,
        frame_size: int,
        temperature: float | Tensor = 0.05,
        noise: NoiseInputs | None = None,
    ) -> Tensor:
        """Weighted nonlinear time-domain audio, without the linear readout approximation."""
        self.validate(anatomy)
        weights = self.weights(anatomy, temperature)
        return torch.stack(
            [
                weight
                * component.simulate(
                    articulators, parameters, pressure, randomness, samples, frame_size, temperature, anatomy, noise
                ).audio
                for weight, component in zip(weights, self.components, strict=True)
            ]
        ).sum(0)
