"""Differentiable AirwayNetwork::step, including the interactive glottal port.

CGS units. The branch momentum equation uses Rust's stable analytic quadratic
root (a semi-implicit Bernoulli/inertance solve); yielding walls are staggered,
and node/radiation coupling is Crank–Nicolson. Autograd differentiates these
algebraic solves and every recurrent state, with no detached state.
Hard mode retains Rust predicates; positive temperature relaxes them.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .acoustics import HumanAcoustics
from .anatomy import Anatomy
from .relaxation import lower_bound, opening, positive
from .topology import CLOSED_AREA, RHO, RHO_C2, SectionCounts, build_human


@dataclass
class NetworkState:
    pressure: Tensor
    flow: Tensor
    momentum: Tensor
    inertance: Tensor
    wall_velocity: Tensor
    wall_displacement: Tensor
    radiation_flow: Tensor
    radiation_inertance: Tensor
    radiated_total: Tensor
    branch_area: Tensor
    area: Tensor
    minimum: Tensor
    volume: Tensor
    volume_source1: Tensor
    volume_source2: Tensor


class AirwayDynamics(nn.Module):
    """Full subglottal/supraglottal network, unlike the ideal-flow linear cut.

    Geometry here is an instantaneous frame with true sub-section minima.
    State is passed explicitly so any caller can retain gradients over chunks.
    Moving-wall squeeze routing and its two-pole volume source preserve Rust
    node order. Pressure-release bursts emerge from this physical state.
    """

    def __init__(
        self, anatomy: Anatomy = Anatomy(), sample_rate: float = 96000.0, counts: SectionCounts | None = None
    ) -> None:
        super().__init__()
        self.ac = HumanAcoustics(build_human(sample_rate, anatomy, counts))
        self.fs = sample_rate
        h, t = self.ac.h, self.ac.h.topo
        entrance, exit_ = [0.0] * len(t.a), [0.0] * len(t.a)
        entrance[h.glottis_port], exit_[h.glottis_port] = 0.37, 1.0
        entrance[h.velum_port], exit_[h.velum_port] = 0.3, 0.8
        self.register_buffer("entrance", torch.tensor(entrance, dtype=torch.float64))
        self.register_buffer("exit", torch.tensor(exit_, dtype=torch.float64))
        adjacency = torch.zeros(t.n_nodes, t.n_nodes, dtype=torch.float64)
        for a, b in zip(t.a, t.b, strict=True):
            adjacency[a, b] += 1
            adjacency[b, a] += 1
        self.register_buffer("adjacency", adjacency)
        self.register_buffer("junction", adjacency.sum(-1) >= 3)
        radiation = torch.zeros(t.n_nodes, len(t.rad_nodes), dtype=torch.float64)
        for j, node in enumerate(t.rad_nodes):
            radiation[node, j] = 1
        self.register_buffer("radiation", radiation)

    def initial(
        self,
        like: Tensor,
        frame: dict[str, Tensor],
        anatomy: Tensor | None = None,
        temperature: float | Tensor | None = 0.05,
    ) -> NetworkState:
        """All dynamic states initially at rest, as AirwayNetwork::finalize."""
        b = like.shape[0]
        node = like.new_zeros(b, self.ac.h.topo.n_nodes)
        branch = like.new_zeros(b, len(self.ac.h.topo.a))
        radiation = like.new_zeros(b, len(self.ac.h.topo.rad_nodes))
        area, minimum, _ = self.geometry(frame, anatomy)
        el = self.ac.elements(
            frame["ph"],
            frame["orl"],
            frame["ph_len"],
            frame["or_len"],
            frame["velum"],
            None if temperature is None else temperature * CLOSED_AREA,
            anatomy,
        )
        return NetworkState(
            node,
            branch,
            branch,
            branch,
            node,
            node,
            radiation,
            radiation,
            like.new_zeros(b),
            branch,
            area,
            minimum,
            el.c * RHO_C2,
            node,
            node,
        )

    def geometry(self, frame: dict[str, Tensor], anatomy: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """Actual node area, sub-section minimum, and dx (cm²/cm²/cm)."""
        ac = self.ac
        ratio = 1.0 if anatomy is None else anatomy[0] / ac.h.topo.k
        b = frame["ph"].shape[0]
        area = (ac.area0 * torch.where(ac.fixed, 1.0, ratio**2)).expand(b, -1).clone()
        minimum = area.clone()
        dx = (ac.dx0 * torch.where(ac.fixed, 1.0, ratio)).expand(b, -1).clone()
        area[:, ac.ph], area[:, ac.orl] = frame["ph"], frame["orl"]
        minimum[:, ac.ph], minimum[:, ac.orl] = frame["ph_min"], frame["or_min"]
        dx[:, ac.ph], dx[:, ac.orl] = frame["ph_len"][:, None] / ac.h.n_ph, frame["or_len"][:, None] / ac.h.n_or
        return area, minimum, dx

    def step(
        self,
        state: NetworkState,
        frame: dict[str, Tensor],
        glottis: Tensor,
        lung_pressure: Tensor,
        temperature: float | Tensor | None = 0.05,
        anatomy: Tensor | None = None,
        pressure_sources: Tensor | None = None,
        external_flow: Tensor | None = None,
        targets: dict[str, Tensor] | None = None,
    ) -> tuple[Tensor, NetworkState]:
        """One Rust-ordered momentum/wall/node step; lung_pressure is dyn/cm².

        pressure_sources: (B,E), dyn/cm²; external_flow: (B,N), cm³/s. These
        allow explicit fixed turbulence realisations and ideal-flow oracle tests.
        No random generator is called and no state is detached here.
        """
        ac, fs = self.ac, self.fs

        def pos(x: Tensor, unit: float = 1.0) -> Tensor:
            return x.clamp_min(0) if temperature is None else positive(x, temperature * unit)

        def floor(x: Tensor, limit: float, unit: float) -> Tensor:
            return x.clamp_min(limit) if temperature is None else lower_bound(x, limit, temperature * unit)

        def gate(x: Tensor, unit: float = 1.0) -> Tensor:
            return (x >= 0).to(x.dtype) if temperature is None else opening(x, temperature * unit, unit)

        def minimum(x: Tensor, y: Tensor, unit: float) -> Tensor:
            if temperature is None:
                return torch.minimum(x, y)
            weight = torch.sigmoid(2 * torch.asinh((y - x) / (temperature * unit)))
            return weight * x + (1 - weight) * y

        area, mins, dx = self.geometry(frame, anatomy)
        p = torch.where(ac.fixed, lung_pressure[:, None], state.pressure)
        el = ac.elements(
            frame["ph"],
            frame["orl"],
            frame["ph_len"],
            frame["or_len"],
            frame["velum"],
            None if temperature is None else temperature * CLOSED_AREA,
            anatomy,
            glottis,
        )
        aa, ab = area[:, ac.ba], area[:, ac.bb]
        ma, mb = mins[:, ac.ba], mins[:, ac.bb]
        ia, ib = 1 / floor(aa, 1e-6, 1e-6), 1 / floor(ab, 1e-6, 1e-6)
        ima, imb = 1 / floor(ma, 1e-6, 1e-6), 1 / floor(mb, 1e-6, 1e-6)
        kf = 0.5 * RHO * (pos(ima - ib) ** 2 + 0.1 * pos(imb - ia) ** 2)
        kb = 0.5 * RHO * (pos(imb - ia) ** 2 + 0.1 * pos(ima - ib) ** 2)
        ratio = 1.0 if anatomy is None else anatomy[0] / ac.h.topo.k
        port = (ac.port_area0 * ratio**2).expand(area.shape[0], -1).clone()
        port[:, ac.velum_port], port[:, ac.h.glottis_port] = frame["velum"], glottis
        ip = 1 / floor(port, CLOSED_AREA, CLOSED_AREA)
        port_kf = 0.5 * RHO * (self.entrance * ip.square() + self.exit * pos(ip - ib) ** 2)
        port_kb = 0.5 * RHO * (self.entrance * ip.square() + self.exit * pos(ip - ia) ** 2)
        kf, kb = torch.where(ac.is_port, port_kf, kf), torch.where(ac.is_port, port_kb, kb)
        open_ = el.open.to(area.dtype) * torch.where(
            ac.is_port, 1.0, gate(minimum(ma, mb, CLOSED_AREA) - CLOSED_AREA, CLOSED_AREA)
        )
        # On opening Rust conserves flow; on closing it conserves momentum.
        opening_weight = gate(state.inertance - el.l, 0.001)
        previous = state.momentum + (el.l * state.flow - state.momentum) * opening_weight
        c = p[:, ac.ba] - p[:, ac.bb] + previous * fs
        if pressure_sources is not None:
            c = c + pressure_sources
        coefficient = kb + (kf - kb) * gate(c, 0.001)
        absolute = c.abs() if temperature is None else torch.hypot(c, torch.zeros_like(c) + temperature * 0.001)
        bq = el.l * fs + el.r
        flow = open_ * 2 * c / (bq + torch.sqrt(bq.square() + 4 * coefficient * absolute))
        momentum = el.l * flow
        inertance = state.inertance + (el.l - state.inertance) * open_
        net = torch.zeros_like(p) if external_flow is None else external_flow
        net = net.index_add(1, ac.ba, -flow).index_add(1, ac.bb, flow)
        wall_den = torch.where(ac.has_wall, ac.wall_m * fs + 0.5 * ac.wall_r, 1.0)
        velocity = (
            (ac.wall_m * fs - 0.5 * ac.wall_r) * state.wall_velocity + p - ac.wall_k * state.wall_displacement
        ) / wall_den
        velocity = torch.where(ac.has_wall, velocity, 0.0)
        displacement = state.wall_displacement + velocity / fs
        wall_flow = el.wall_area * velocity
        # Moving-wall source, network.rs step(1): squeeze out of a sealed cell
        # is shared among eligible neighbours before a two-pole 60 Hz filter.
        # Eligibility sees already-updated earlier nodes and old later nodes,
        # exactly as Rust's in-place geometry loop. No source mass is invented.
        current_volume = el.c * RHO_C2
        raw = -(current_volume - state.volume) * fs
        effective = minimum(area, mins, CLOSED_AREA)
        coordinate = effective / 0.12
        if temperature is None:
            coordinate = coordinate.clamp(0, 1)
        else:
            coordinate = positive(coordinate, temperature * 0.001) - positive(coordinate - 1, temperature * 0.001)
        keep = coordinate.square() * (3 - 2 * coordinate)
        old_open = gate(minimum(state.area, state.minimum, CLOSED_AREA) - 0.12, 0.12)
        new_open = gate(effective - 0.12, 0.12)
        target_min = mins if targets is None else self.geometry(targets, anatomy)[1]
        target_open = gate(target_min - 0.12, 0.12)
        lower_index = torch.tril(torch.ones_like(self.adjacency), diagonal=-1)
        neighbour_open = lower_index[None] * new_open[:, None, :] + (1 - lower_index)[None] * old_open[:, None, :]
        eligible = 1 - (1 - neighbour_open) * (1 - target_open[:, None, :])
        weights = self.adjacency[None] * eligible
        weights = weights / floor(weights.sum(-1), 1.0, 0.01)[..., None]
        injection = raw * keep + ((raw * (1 - keep))[..., None] * weights).sum(-2)
        coefficient = 1 - torch.exp(p.new_tensor(-2 * torch.pi * 60 / fs))
        volume_source1 = state.volume_source1 + coefficient * (injection - state.volume_source1)
        volume_source2 = state.volume_source2 + coefficient * (volume_source1 - state.volume_source2)
        net = net - wall_flow + volume_source2
        compliance = el.c
        inverse_sum = torch.zeros_like(p).index_add(1, ac.ba, open_ / el.l).index_add(1, ac.bb, open_ / el.l)
        minimum_c = inverse_sum / (fs * fs * 2.4)
        if temperature is None:
            junction_c = torch.maximum(compliance, minimum_c)
        else:
            junction_c = minimum_c + pos(compliance - minimum_c, 1e-9)
        compliance = torch.where(self.junction, junction_c, compliance)
        candidate = p + net / (fs * compliance)
        rad_p = p[:, ac.rad_nodes]
        rad_c = compliance[:, ac.rad_nodes]
        rad_net = net[:, ac.rad_nodes]
        ar = el.rad_area
        lr = 8 * RHO / (3 * torch.pi * (torch.pi * ar).sqrt())
        rr = 128 * RHO * 35000 / (9 * torch.pi**2 * ar)
        # Radiation momentum uses the SAME opening/closing convention.
        previous_lr = floor(state.radiation_inertance, 1e-30, 1e-30)
        closing = gate(lr - previous_lr, 0.001) * gate(state.radiation_inertance - 1e-30, 1e-30)
        ul_previous = state.radiation_flow * (1 + closing * (previous_lr / lr - 1))
        h, g = 1 / (2 * fs * lr), 0.5 / rr
        rad_new = (rad_p * (rad_c * fs - 0.5 * h - g) + rad_net - ul_previous) / (rad_c * fs + 0.5 * h + g)
        ul = ul_previous + h * (rad_p + rad_new)
        radiated = 0.5 * (ul_previous + ul) + g * (rad_new + rad_p)
        candidate = candidate * (1 - self.radiation.sum(-1)) + rad_new @ self.radiation.T
        node_open = gate(minimum(area, mins, CLOSED_AREA) - CLOSED_AREA, CLOSED_AREA)
        neighbour = self.adjacency[None] * node_open[:, None, :]
        count = neighbour.sum(-1)
        presence = gate(count - 0.5, 0.01)
        neighbour = neighbour / floor(count, 1.0, 0.01)[..., None]
        lower = torch.tril(neighbour, diagonal=-1)
        upper = torch.triu(neighbour)
        is_rad = self.radiation.sum(-1)
        relaxation = 0.02 * presence * (1 - is_rad) * (1 - node_open)
        # Rust updates sealed-node pressure in node order. A triangular solve
        # reproduces that order (including already-updated open neighbours),
        # while retaining every continuous dependency in autograd.
        base_closed = p * (1 - 0.02 * presence) * (1 - is_rad) + 0.98 * p * is_rad
        base = node_open * candidate + (1 - node_open) * base_closed + relaxation * (upper @ p[..., None])[..., 0]
        matrix = torch.eye(p.shape[-1], dtype=p.dtype, device=p.device)[None] - relaxation[..., None] * lower
        new_pressure = torch.linalg.solve_triangular(matrix, base[..., None], upper=False)[..., 0]
        new_pressure = torch.where(ac.fixed, lung_pressure[:, None], new_pressure)
        rad_open = node_open[:, ac.rad_nodes]
        ul = rad_open * ul + (1 - rad_open) * state.radiation_flow * 0.995
        radiated = rad_open * radiated + (1 - rad_open) * state.radiation_flow * 0.995
        total = (wall_flow * ac.wall_rad).sum(-1) + radiated.sum(-1)
        audio = (total - state.radiated_total) * fs
        return audio, NetworkState(
            new_pressure,
            flow,
            momentum,
            inertance,
            velocity,
            displacement,
            ul,
            state.radiation_inertance + (lr - state.radiation_inertance) * rad_open,
            total,
            torch.where(ac.is_port, port, minimum(ma, mb, CLOSED_AREA)),
            area,
            mins,
            current_volume,
            volume_source1,
            volume_source2,
        )
