"""Pressure-coupled glottis and airway, with a separate convolutional readout.

simulate() returns the actual time-stepper output and glottal volume velocity.
forward() convolves that pressure-coupled flow with the frozen-block linear
tract. These outputs are distinct: nonlinear supraglottal losses are present in
simulate(), while convolution is the explicitly linear frozen-geometry readout.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .anatomy import Anatomy
from .dynamics import AirwayDynamics
from .geometry import ExactGeometry
from .glottis import CM_H2O, GlottisState, ParametricGlottis
from .renderer import Renderer, sample_frames
from .source_random import SourceRandomness
from .topology import SectionCounts
from .turbulence import NoiseInputs, Turbulence


@dataclass
class Simulation:
    """All outputs remain in the same autograd graph, including recurrent flow."""

    audio: Tensor
    flow: Tensor
    glottal_area: Tensor
    geometry: dict[str, Tensor]
    pressure_sources: Tensor
    moving_flow: Tensor


class CoupledSynth(nn.Module):
    """Instantaneous articulators, GlottisParams and lung pressure → audio.

    This directly exposes CHIMERA GlottisParams; it does not choose a source or
    anatomy from pitch. Lung pressure is cmH₂O. sample_rate is the internal
    physical integration rate, not an implicit oversampling/decimation preset.
    """

    def __init__(
        self,
        anatomy: Anatomy = Anatomy(),
        sample_rate: float = 96000.0,
        counts: SectionCounts | None = None,
        block_size: int = 256,
        ir_length: int = 2048,
        response_fft: int = 8192,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.geometry = ExactGeometry(anatomy, sample_rate, counts)
        self.dynamics = AirwayDynamics(anatomy, sample_rate, counts)
        self.renderer = Renderer(anatomy, sample_rate, sample_rate, block_size, ir_length, response_fft, counts)

    def simulate(
        self,
        articulators: Tensor,
        parameters: Tensor,
        pressure: Tensor,
        randomness: SourceRandomness,
        samples: int,
        frame_size: int,
        temperature: float | Tensor | None = 0.05,
        anatomy: Tensor | None = None,
        noise: NoiseInputs | None = None,
    ) -> Simulation:
        """Sample-ordered ParametricGlottis::next then AirwayNetwork::step.

        Articulators (B,T,17), constant source controls (B,17), lung pressure
        (B,), fixed cycle/drift realisations; no random resampling in backward.
        State advances without detachment. Controls interpolate on a fixed
        sample clock; live HumanVoice command dynamics are not implied.
        """
        if samples < 1 or frame_size < 1 or randomness.drift.numel() < (samples + 31) // 32:
            raise ValueError("nonempty simulation and sufficient fixed drift samples required")
        mode = "hard" if temperature is None else "fit"
        g = self.geometry.frame(articulators, mode, 1.0 if temperature is None else temperature, anatomy)
        positions = (torch.arange(samples, dtype=articulators.dtype, device=articulators.device) + 1) / frame_size
        frames = {key: sample_frames(value, positions) for key, value in g.items()}
        state = self.dynamics.initial(parameters, {key: value[:, 0] for key, value in g.items()}, anatomy, temperature)
        turbulence = None if noise is None else Turbulence(self.dynamics.ac, noise, temperature, anatomy)
        noise_state = None if turbulence is None else turbulence.initial()
        placements = (
            []
            if turbulence is None
            else [
                turbulence.placement({key: value[:, i] for key, value in g.items()})
                for i in range(articulators.shape[1])
            ]
        )
        if noise is not None and noise.white.shape[-1] < 3 * samples:
            raise ValueError("fixed white stream must cover every possible source registration")
        glottis = ParametricGlottis(parameters, self.sample_rate, randomness, temperature)
        source_state = GlottisState.initial(pressure)
        audio, flows, areas, pressure_history, moving_history = [], [], [], [], []
        ac = self.dynamics.ac
        for sample in range(samples):
            area, source_state = glottis.step(
                source_state, state.pressure[:, ac.h.trachea.last] - state.pressure[:, ac.inject], sample
            )
            frame = {key: value[:, sample] for key, value in frames.items()}
            target_index = min(sample // frame_size + 1, articulators.shape[1] - 1)
            targets = {key: value[:, target_index] for key, value in g.items()}
            source_pressure = torch.zeros_like(state.flow)
            if turbulence is not None:
                if sample % frame_size == 0 and sample // frame_size < articulators.shape[1]:
                    noise_state = turbulence.control(noise_state, placements[target_index])
                source_pressure, noise_state = turbulence.step(
                    noise_state, placements[target_index], state.branch_area, state.flow
                )
            y, state = self.dynamics.step(
                state,
                frame,
                area,
                pressure * CM_H2O,
                temperature,
                anatomy,
                pressure_sources=source_pressure,
                targets=targets,
            )
            pressure_history.append(source_pressure)
            moving_history.append(state.volume_source2)
            audio.append(y)
            flows.append(state.flow[:, ac.h.glottis_port])
            areas.append(area)
        if bool((source_state.phase >= randomness.cycles.shape[0]).any()):
            raise ValueError("fixed source randomness does not cover the simulated cycle domain")
        return Simulation(
            torch.stack(audio, -1),
            torch.stack(flows, -1),
            torch.stack(areas, -1),
            g,
            torch.stack(pressure_history, -1),
            torch.stack(moving_history, -1),
        )

    def forward(
        self,
        articulators: Tensor,
        parameters: Tensor,
        pressure: Tensor,
        randomness: SourceRandomness,
        samples: int,
        frame_size: int,
        temperature: float | Tensor = 0.05,
        anatomy: Tensor | None = None,
        noise: NoiseInputs | None = None,
    ) -> Tensor:
        """Frozen-block linear convolution of the pressure-coupled glottal flow."""
        result = self.simulate(
            articulators, parameters, pressure, randomness, samples, frame_size, temperature, anatomy, noise
        )
        placement = None
        if noise is not None:
            e = result.pressure_sources.shape[1]
            placement = torch.eye(e, dtype=parameters.dtype, device=parameters.device)[None, None]
            placement = placement.expand(articulators.shape[0], articulators.shape[1], -1, -1)
        return self.renderer(
            result.flow,
            result.geometry,
            frame_size,
            temperature * 2e-4,
            pressure_sources=None if noise is None else result.pressure_sources,
            placement=placement,
            anatomy=anatomy,
            node_sources=result.moving_flow if articulators.shape[1] > 1 else None,
        )
