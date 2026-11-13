"""Frozen-block convolution (VocalTrax-style), using the CHIMERA linear airway.

Geometry is sampled at each excitation block's midpoint. Each block keeps that
geometry for its whole impulse-response tail. This is a moving-geometry
approximation, not a time-domain port of AirwayNetwork::step.
"""

from dataclasses import fields

import torch
from torch import Tensor, nn

from .acoustics import Elements, HumanAcoustics
from .anatomy import Anatomy
from .topology import SectionCounts, build_human


def sample_frames(x: Tensor, positions: Tensor) -> Tensor:
    """Linear interpolation of (B,T,...) controls at FIXED frame positions.

    Integer indices depend on sample clocks only, never on articulator/source
    values. T=1 is a stationary configuration. Values after the final control
    frame hold that frame, rather than extrapolating.
    """
    if x.shape[1] == 1:
        return x[:, :1].expand(-1, positions.numel(), *x.shape[2:])
    left = positions.floor().long().clamp(0, x.shape[1] - 2)
    weight = (positions - left).clamp(0, 1).to(x.dtype)
    weight = weight.reshape(1, -1, *([1] * (x.ndim - 2)))
    return torch.lerp(x.index_select(1, left), x.index_select(1, left + 1), weight)


class Renderer(nn.Module):
    """Ideal-flow input (cm³/s) → radiated dU/dt (cm³/s²), arbitrary gain.

    All time constants are in seconds; sample_rate/internal_rate are Hz.
    response_fft sets frequency-sampling aliasing; ir_length is a centred
    rectangular window of the bandlimited IR and must not exceed response_fft.
    """

    def __init__(
        self,
        anatomy: Anatomy = Anatomy(),
        sample_rate: float = 16000.0,
        internal_rate: float = 96000.0,
        block_size: int = 256,
        ir_length: int = 2048,
        response_fft: int = 8192,
        counts: SectionCounts | None = None,
    ) -> None:
        super().__init__()
        if min(sample_rate, internal_rate) <= 0:
            raise ValueError("sample rates must be positive")
        if min(block_size, ir_length, response_fft) < 1 or ir_length > response_fft:
            raise ValueError("invalid convolution sizes")
        self.acoustics = HumanAcoustics(build_human(internal_rate, anatomy, counts))
        self.sample_rate = sample_rate
        self.block_size, self.ir_length, self.response_fft = block_size, ir_length, response_fft
        # Sampling the internal-rate response onto an output-rate frequency
        # interval is a bandlimited projection, which has negative-time tails.
        # Centre the finite IR window and remove this FIXED indexing offset
        # after overlap-add. Dropping those tails biases phase and DC gain.
        self.pre_response = ir_length // 2
        self.register_buffer("freqs", torch.fft.rfftfreq(response_fft, 1 / sample_rate, dtype=torch.float64))

    def forward(
        self,
        flow: Tensor,
        geometry: dict[str, Tensor],
        frame_size: int,
        temperature: float | Tensor | None = None,
        pressure_sources: Tensor | None = None,
        placement: Tensor | None = None,
        anatomy: Tensor | None = None,
        node_sources: Tensor | None = None,
    ) -> Tensor:
        """Render (B,L) flow with geometry (B,T,sections) / lengths (B,T).

        Optional pressure_sources (B,S,L), dyn/cm², are routed using continuous
        placement (B,T,S,E) over ALL network branches. Their signs follow the
        branch a→b orientation. Weights are not silently normalised: magnitude
        and negative weights are useful for superposition tests. Noise gains,
        waveforms and placement all carry gradients.
        """
        if flow.ndim != 2 or flow.shape[-1] == 0 or frame_size < 1:
            raise ValueError("flow must be nonempty (B,L), frame_size positive")
        batch, length = flow.shape
        bs = self.block_size
        blocks = (length + bs - 1) // bs
        positions = (torch.arange(blocks, device=flow.device, dtype=flow.dtype) * bs + (bs - 1) / 2) / frame_size
        keys = ("ph", "orl", "ph_len", "or_len", "velum")
        values = [sample_frames(geometry[key], positions).flatten(0, 1) for key in keys]
        el = self.acoustics.elements(*values, temperature=temperature, anatomy=anatomy)
        fft_size = 1 << (bs + self.ir_length - 2).bit_length()
        src = torch.nn.functional.pad(flow, (0, blocks * bs - length)).reshape(batch * blocks, bs)
        # Solve separately per block to keep the example implementation readable
        # and avoid materialising B*blocks*F*nodes activations at once.
        outputs = []
        if (pressure_sources is None) != (placement is None):
            raise ValueError("pressure_sources and placement must be supplied together")
        if pressure_sources is not None:
            pressure = torch.nn.functional.pad(pressure_sources, (0, blocks * bs - length))
            pressure = pressure.reshape(batch, -1, blocks, bs).permute(0, 2, 1, 3).flatten(0, 1)
            weights = sample_frames(placement, positions).flatten(0, 1)
        if node_sources is not None:
            node_source = torch.nn.functional.pad(node_sources, (0, blocks * bs - length))
            node_source = node_source.reshape(batch, -1, blocks, bs).permute(0, 2, 1, 3).flatten(0, 1)
        for i in range(batch * blocks):
            item = Elements(**{f.name: getattr(el, f.name)[i : i + 1] for f in fields(Elements)})
            nodes, dipoles = self.acoustics.discrete_responses(item, self.freqs, temperature)
            # Factor D_internal = D_output * correction. Truncating the
            # pressure IR directly leaks DC (and its band-edge discontinuity
            # produces a long anticausal tail). Factorisation keeps the exact
            # DC zero in the final first difference, while implementing Rust's
            # derivative clock and its frequency-dependent fractional advance.
            fraction = self.freqs / self.sample_rate
            correction = torch.exp(1j * torch.pi * fraction * (1 - self.sample_rate / self.acoustics.fs))
            correction = correction * torch.sinc(self.freqs / self.acoustics.fs) / torch.sinc(fraction)
            nodes, dipoles = nodes * correction[None, :, None], dipoles * correction[None, :, None]
            response = nodes[..., self.acoustics.inject]
            full_ir = torch.fft.irfft(response, n=self.response_fft)
            ir = torch.roll(full_ir, self.pre_response, -1)[..., : self.ir_length]
            spectrum = torch.fft.rfft(src[i : i + 1], n=fft_size) * torch.fft.rfft(ir, n=fft_size)
            if pressure_sources is not None:
                response_p = torch.einsum("bfe,se->bsf", dipoles, weights[i].to(dipoles.dtype))
                full_p = torch.fft.irfft(response_p, n=self.response_fft)
                ir_p = torch.roll(full_p, self.pre_response, -1)[..., : self.ir_length]
                spectrum = spectrum + (torch.fft.rfft(pressure[i], n=fft_size) * torch.fft.rfft(ir_p, n=fft_size)).sum(
                    1
                )
            if node_sources is not None:
                full_node = torch.fft.irfft(nodes.transpose(1, 2), n=self.response_fft)
                ir_node = torch.roll(full_node, self.pre_response, -1)[..., : self.ir_length]
                spectrum = spectrum + (
                    torch.fft.rfft(node_source[i], n=fft_size) * torch.fft.rfft(ir_node, n=fft_size)
                ).sum(1)
            outputs.append(torch.fft.irfft(spectrum, n=fft_size)[..., : bs + self.ir_length - 1])
        # Each excitation block contributes its complete tail; slice only at the
        # caller's requested end. Padding/addition keeps autograd intact.
        out_length = blocks * bs + self.ir_length - 1
        rows = torch.cat(outputs).reshape(batch, blocks, -1)
        out = sum(
            torch.nn.functional.pad(rows[:, i], (i * bs, out_length - i * bs - rows.shape[-1])) for i in range(blocks)
        )
        out = (out - torch.nn.functional.pad(out[:, :-1], (1, 0))) * self.sample_rate
        return out[:, self.pre_response : self.pre_response + length]
