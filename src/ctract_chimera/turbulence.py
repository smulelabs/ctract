"""HumanAirway::update_noise + AirwayNetwork::compute_noise, continuously routed.

Localisation uses TRUE sub-section minima and teeth_s. Discrete offsets and
incisor rounding are CDF-bin partitions. The white stream has one continuously
advanced draw cursor, so registering/removing a source has the Rust hard limit
without silently changing to three independently seeded noise streams.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from .acoustics import HumanAcoustics
from .geometry_relaxation import Relaxation
from .noise import BANDS, GAINS, turbulence_amplitude
from .relaxation import opening, strict_opening
from .source_random import Random


@dataclass(frozen=True)
class NoiseInputs:
    """Fixed white stream (B,K), gains (3,), bands (3,2): LP/HP in Hz.

    Source order is aspiration, constriction, teeth. Gains and both cutoffs are
    tensor inputs; no filter, gain, or placement is detached. A stream of 3*L
    samples covers every possible source registration for an L-sample clip.
    """

    white: Tensor
    gains: Tensor
    bands: Tensor

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "NoiseInputs":
        """Move all fixed/fit inputs together; Tensor.to retains gradient edges."""
        return NoiseInputs(*(value.to(device=device, dtype=dtype) for value in (self.white, self.gains, self.bands)))

    @classmethod
    def chimera(cls, samples: int, batch: int = 1) -> "NoiseInputs":
        if samples < 1 or batch < 1:
            raise ValueError("noise realisations need positive samples and batch size")
        rng = Random(12345)
        stream = torch.tensor([(2 * rng.uniform() - 1) * 1.732 for _ in range(3 * samples)], dtype=torch.float64)
        return cls(
            stream[None].expand(batch, -1),
            torch.tensor([GAINS[name] for name in ("aspiration", "constriction", "teeth")], dtype=torch.float64),
            torch.tensor([BANDS[name] for name in ("aspiration", "constriction", "teeth")], dtype=torch.float64),
        )


@dataclass
class NoiseState:
    hp: Tensor
    lp1: Tensor
    lp2: Tensor
    cursor: Tensor
    identity: Tensor  # Probability of each (src,fwd) identity, con/teeth only.


@dataclass
class Placement:
    constriction: Tensor  # (B,N,offsets)
    teeth: Tensor
    teeth_branch: Tensor  # (B,n_or-1)
    area_gain: Tensor  # (B,N)
    obstacle_gain: Tensor
    registered: Tensor  # (B,3)
    identity: Tensor  # (B,2,E²)


class Turbulence:
    """Explicit geometry-control and sample-filter state, not a hidden cache."""

    def __init__(
        self,
        acoustics: HumanAcoustics,
        inputs: NoiseInputs,
        temperature: float | Tensor | None,
        anatomy: Tensor | None = None,
    ) -> None:
        self.ac, self.inputs, self.temperature = acoustics, inputs, temperature
        self.k = acoustics.h.topo.k if anatomy is None else anatomy[0]
        self.r = (
            None
            if temperature is None
            else Relaxation(torch.as_tensor(temperature, dtype=inputs.white.dtype, device=inputs.white.device))
        )
        h = acoustics.h
        self.n, self.e = h.n_ph + h.n_or, len(h.topo.a)

        # Tube node and internal-branch offsets differ after trachea. Find the
        # actual branch by endpoints, so no anatomical numbering is assumed.
        def tube_branch(i: int) -> int:
            if i == h.n_ph:
                return -1
            tube, j = (h.pharynx, i) if i < h.n_ph else (h.oral, i - h.n_ph)
            j = min(max(j, 0), tube.n - 2)
            return h.topo.a.index(tube.node(j))

        src = [tube_branch(min(i, self.n - 2)) for i in range(self.n)]
        fwd = [[tube_branch(min(self.n - 2, i + offset)) for offset in range(1, self.n + 1)] for i in range(self.n)]
        bwd = [[tube_branch(max(0, i - offset)) for offset in range(1, self.n + 1)] for i in range(self.n)]
        device = inputs.white.device
        self.valid = torch.tensor(
            [[src[i] >= 0 and fwd[i][j] >= 0 and bwd[i][j] >= 0 for j in range(self.n)] for i in range(self.n)],
            device=device,
        )
        self.src = torch.tensor([max(i, 0) for i in src], device=device)
        self.fwd = torch.tensor([[max(i, 0) for i in row] for row in fwd], device=device)
        self.bwd = torch.tensor([[max(i, 0) for i in row] for row in bwd], device=device)
        self.teeth_branches = torch.tensor([h.topo.a.index(h.oral.node(i)) for i in range(h.n_or - 1)], device=device)
        self.asp_fwd = h.topo.a.index(h.pharynx.node(1))
        self.asp_bwd = h.topo.a.index(h.trachea.node(h.trachea.n - 2))

    def initial(self) -> NoiseState:
        b = self.inputs.white.shape[0]
        zero = self.inputs.white.new_zeros(b, 3)
        return NoiseState(zero, zero, zero, zero[:, 0], self.inputs.white.new_zeros(b, 2, self.e * self.e))

    def _ramp(self, lo: float | Tensor, hi: float | Tensor, x: Tensor) -> Tensor:
        if self.r is not None:
            return self.r.ramp(lo, hi, x)
        u = ((x - lo) / (hi - lo)).clamp(0, 1)
        return u.square() * (3 - 2 * u)

    def _strict(self, x: Tensor, unit: float = 1.0) -> Tensor:
        return (x > 0).to(x.dtype) if self.r is None else strict_opening(x, self.r.tau * unit, unit)

    def _bins(self, coordinate: Tensor, boundaries: Tensor) -> Tensor:
        if self.r is None:
            cdf = (coordinate[..., None] >= boundaries).to(coordinate.dtype)
        else:
            cdf = opening(coordinate[..., None] - boundaries, self.r.tau * 0.001, 1.0)
        return torch.cat((1 - cdf[..., :1], cdf[..., :-1] - cdf[..., 1:], cdf[..., -1:]), -1)

    def placement(self, frame: dict[str, Tensor]) -> Placement:
        """Continuous extension of update_noise, including its seam exclusions."""
        ac, k = self.ac, self.k
        area = torch.cat((frame["ph_min"], frame["or_min"]), -1)
        if self.r is None:
            visible = (area > 0).to(area.dtype)
            score = torch.where(area > 0, area, torch.inf)
            index = score.argmin(-1)
            location = torch.nn.functional.one_hot(index, self.n).to(area.dtype) * (area > 0).any(-1, keepdim=True)
        else:
            width = self.r.tau.square() * 0.001
            visible = torch.sigmoid(area / width + self.r.tau.log())
            cost = area * visible + (1 - visible) * (1 + area.sum(-1, keepdim=True))
            # Vanishing lexicographic bias gives Rust's FIRST equal minimum;
            # unlike an un-biased softmin, ties do not retain equal mass.
            tie = torch.arange(self.n, dtype=area.dtype, device=area.device) * 0.001 * self.r.tau.sqrt()
            # Pairwise algebraic ranks keep all finite-temperature minima
            # sensitivities live, unlike exp(-area/width) underflow on ordinary
            # cm² gaps. Cooling the rank distribution still selects argmin.
            biased = cost + tie
            rank = self.r.gate(biased[..., :, None] - biased[..., None, :], self.r.tau * 0.001).sum(-1)
            location = torch.softmax(-rank / self.r.tau.sqrt(), -1)
            location = location * (1 - (1 - visible).prod(-1, keepdim=True))
        dx = torch.cat(
            (
                (frame["ph_len"] / ac.h.n_ph)[:, None].expand(-1, ac.h.n_ph),
                (frame["or_len"] / ac.h.n_or)[:, None].expand(-1, ac.h.n_or),
            ),
            -1,
        )
        position = dx.cumsum(-1) - 0.5 * dx
        boundaries = torch.arange(1.5, self.n + 0.5, dtype=area.dtype, device=area.device)
        offsets = self._bins(0.9 * k / dx, boundaries)
        area_gain = 1 - self._ramp(0.35 * k * k, 0.9 * k * k, area)
        area_registration = self._strict(area_gain, 0.001)
        base = location[..., None] * offsets * self.valid
        constriction = base * area_registration[..., None]
        behind = (frame["teeth_s"][:, None] - position) / k
        obstacle = self._ramp(0.15, 0.5, behind) * (1 - self._ramp(2.6, 4.0, behind))
        obstacle = obstacle * (1 - self._ramp(0.2 * k * k, 0.5 * k * k, area))
        teeth = base * area_registration[..., None] * self._strict(obstacle - 0.001, 0.001)[..., None]
        coordinate = (frame["teeth_s"] - frame["ph_len"]) / (frame["or_len"] / ac.h.n_or)
        teeth_w = self._bins(coordinate, torch.arange(0.5, ac.h.n_or - 1.5, dtype=area.dtype, device=area.device))
        b = area.shape[0]
        con_identity = torch.zeros(b, self.e * self.e, dtype=area.dtype, device=area.device)
        identity_index = (self.src[:, None] * self.e + self.fwd).flatten()[None].expand(b, -1)
        con_identity = con_identity.scatter_add(1, identity_index, constriction.flatten(1))
        teeth_identity = torch.zeros_like(con_identity)
        identity_index = (self.src[:, None] * self.e + self.teeth_branches[None]).flatten()[None].expand(b, -1)
        mass = teeth.sum(-1)[..., None] * teeth_w[:, None]
        teeth_identity = teeth_identity.scatter_add(1, identity_index, mass.flatten(1))
        registered = torch.stack((torch.ones_like(area[:, 0]), constriction.sum((-1, -2)), teeth.sum((-1, -2))), -1)
        return Placement(
            constriction,
            teeth,
            teeth_w,
            area_gain,
            obstacle,
            registered,
            torch.stack((con_identity, teeth_identity), 1),
        )

    def control(self, state: NoiseState, placement: Placement) -> NoiseState:
        """set_noise_sources retains filters only for matching (src,fwd)."""
        retain = (state.identity * placement.identity).sum(-1)
        retain = torch.cat((torch.ones_like(retain[:, :1]), retain), -1)
        return NoiseState(state.hp * retain, state.lp1 * retain, state.lp2 * retain, state.cursor, placement.identity)

    def _white(self, index: Tensor) -> Tensor:
        n = self.inputs.white.shape[-1]
        boundaries = torch.arange(0.5, n - 0.5, dtype=index.dtype, device=index.device)
        if self.r is None:
            cdf = (index[:, None] >= boundaries).to(index.dtype)
        else:
            cdf = self.r.gate(index[:, None] - boundaries, self.r.tau * 0.001)
        weights = torch.cat((1 - cdf[:, :1], cdf[:, :-1] - cdf[:, 1:], cdf[:, -1:]), -1)
        return (weights * self.inputs.white).sum(-1)

    def _amplitude(self, area: Tensor, flow: Tensor) -> Tensor:
        """Shared compute_noise Reynolds law, without the caller's family gain."""
        return turbulence_amplitude(area, flow, torch.ones_like(area), self.ac.fs, self.temperature)

    def step(
        self, state: NoiseState, placement: Placement, branch_area: Tensor, flow: Tensor
    ) -> tuple[Tensor, NoiseState]:
        """One causal filter/routing sample, using PREVIOUS branch flow/area."""
        p = placement
        prefix = torch.cat((torch.zeros_like(p.registered[:, :1]), p.registered[:, :-1].cumsum(-1)), -1)
        white = torch.stack([self._white(state.cursor + prefix[:, j]) for j in range(3)], -1)
        hp_coefficient = 1 - torch.exp(-2 * torch.pi * self.inputs.bands[:, 1] / self.ac.fs)
        lp_coefficient = 1 - torch.exp(-2 * torch.pi * self.inputs.bands[:, 0] / self.ac.fs)
        hp = state.hp + p.registered * hp_coefficient * (white - state.hp)
        lp1 = state.lp1 + p.registered * lp_coefficient * (white - hp - state.lp1)
        lp2 = state.lp2 + p.registered * lp_coefficient * (lp1 - state.lp2)
        pressure = torch.zeros(flow.shape[0], self.e, dtype=flow.dtype, device=flow.device)
        gains = self.inputs.gains
        gp = self.ac.h.glottis_port
        amplitude = self._amplitude(branch_area[:, gp], flow[:, gp]) * gains[0] * lp2[:, 0]
        forward = (flow[:, gp] >= 0).to(flow.dtype) if self.r is None else self.r.gate(flow[:, gp], self.r.tau * 0.001)
        # Fixed branch masks, independent of fitted source positions.
        aspiration = (
            torch.nn.functional.one_hot(torch.tensor(self.asp_fwd, device=flow.device), self.e) * forward[:, None]
        )
        aspiration = aspiration - torch.nn.functional.one_hot(
            torch.tensor(self.asp_bwd, device=flow.device), self.e
        ) * (1 - forward[:, None])
        pressure = pressure + amplitude[:, None] * aspiration
        source_flow, source_area = flow[:, self.src], branch_area[:, self.src]
        amplitude = self._amplitude(source_area, source_flow)
        forward = (source_flow >= 0).to(flow.dtype) if self.r is None else self.r.gate(source_flow, self.r.tau * 0.001)
        con = p.constriction * amplitude[..., None] * p.area_gain[..., None] * gains[1] * lp2[:, 1, None, None]
        front = (con * forward[..., None]).flatten(1)
        back = (-con * (1 - forward[..., None])).flatten(1)
        pressure = pressure.scatter_add(1, self.fwd.flatten()[None].expand(flow.shape[0], -1), front)
        pressure = pressure.scatter_add(1, self.bwd.flatten()[None].expand(flow.shape[0], -1), back)
        teeth = p.teeth * amplitude[..., None] * p.obstacle_gain[..., None] * gains[2] * lp2[:, 2, None, None]
        front = (teeth * forward[..., None]).sum(-1).sum(-1)
        pressure = pressure.scatter_add(
            1, self.teeth_branches[None].expand(flow.shape[0], -1), front[:, None] * p.teeth_branch
        )
        back = (-teeth * (1 - forward[..., None])).flatten(1)
        pressure = pressure.scatter_add(1, self.bwd.flatten()[None].expand(flow.shape[0], -1), back)
        return pressure, NoiseState(hp, lp1, lp2, state.cursor + p.registered.sum(-1), state.identity)
