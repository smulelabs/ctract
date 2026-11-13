"""CHIMERA's linear airway: CGS lumped elements and tree elimination.

Ports airway/network.rs::linear_elements and airway/frequency.rs::transfer_function.
The glottis is cut: the input is an ideal volume velocity, NOT an interactive fold model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .relaxation import lower_bound, opening, positive
from .topology import C_SOUND, CLOSED_AREA, MIN_NODE_AREA, MU, PORT, RHO, RHO_C2, VISC, HumanTopology, plan_tree


def _sqrt0(x: torch.Tensor) -> torch.Tensor:
    """sqrt(max(x, 0)) with a zero (not NaN) gradient at x ≤ 0 (a shut velum has area 0)."""
    pos = x > 0
    return torch.where(pos, torch.where(pos, x, torch.ones_like(x)).sqrt(), torch.zeros_like(x))


@dataclass
class Elements:
    l: torch.Tensor  # noqa: E741 -- conventional circuit inductance symbol; matches Rust
    r: torch.Tensor  # (B, nb) series resistance
    open: torch.Tensor  # (B, nb) boolean or continuous opening
    c: torch.Tensor  # (B, n) node compliance
    wall_area: torch.Tensor  # (B, n)
    rad_area: torch.Tensor  # (B, n_rad)


class HumanAcoustics(nn.Module):
    """Fixed-topology linear network, with differentiable tube areas and lengths.

    ``temperature=None`` uses Rust's exact branch-open predicate. A positive
    temperature (cm²) replaces it by a smooth conductance gate. Node counts are
    one fixed graph component; optional tensor anatomy scales
    all fixed physical constants. AnatomicalRenderer mixes count components.
    """

    def __init__(self, h: HumanTopology, dtype: torch.dtype = torch.float64) -> None:
        super().__init__()
        self.h = h
        plan = plan_tree(h)
        t = h.topo
        f = lambda x: torch.tensor(x, dtype=dtype)  # noqa: E731
        self.register_buffer("area0", f(t.area0))
        self.register_buffer("dx0", f(t.dx0))
        self.register_buffer("has_wall", torch.tensor(t.has_wall))
        self.register_buffer("wall_m", f(t.wall_m))
        self.register_buffer("wall_r", f(t.wall_r))
        self.register_buffer("wall_k", f(t.wall_k))
        self.register_buffer("wall_perim", f(t.wall_perim))
        self.register_buffer("wall_rad", f(t.wall_rad))
        self.register_buffer("visc_f", f(t.visc_f))
        self.register_buffer("fixed", torch.tensor(t.fixed))
        self.register_buffer("ba", torch.tensor(t.a))
        self.register_buffer("bb", torch.tensor(t.b))
        self.register_buffer("is_port", torch.tensor([k == PORT for k in t.kind]))
        self.register_buffer("port_area0", f(t.port_area0))
        self.register_buffer("port_len", f(t.port_len))
        self.register_buffer("port_slit", f(t.port_slit))
        self.register_buffer("port_half", torch.tensor(t.port_half, dtype=dtype))
        self.register_buffer("port_extra_r", f(t.port_extra_r))
        self.register_buffer("rad_nodes", torch.tensor(t.rad_nodes))
        # static walk as Python ints (the loop is unrolled at trace time)
        self.order_l = plan.order
        self.parent_l = plan.parent
        self.pb_l = plan.parent_branch
        reachable = set(self.pb_l[1:])
        self.register_buffer("source_branch", f([j in reachable for j in range(len(t.a))]))
        self.ph = slice(h.pharynx.first, h.pharynx.first + h.n_ph)
        self.orl = slice(h.oral.first, h.oral.first + h.n_or)
        self.velum_port = h.velum_port
        self.inject = h.inject
        self.fs = h.topo.fs
        degree = [t.a.count(k) + t.b.count(k) for k in range(t.n_nodes)]
        self.register_buffer("junction", torch.tensor([d >= 3 for d in degree]))

    def elements(
        self,
        ph_areas: Tensor,
        or_areas: Tensor,
        ph_len: Tensor,
        or_len: Tensor,
        velum: Tensor,
        temperature: float | Tensor | None = None,
        anatomy: Tensor | None = None,
        glottis: Tensor | None = None,
    ) -> Elements:
        """Control-rate geometry → linear elements. Shapes: (B, n_ph), (B, n_or), (B,), (B,), (B,).
        `open` is a bool mask, or with temperature a continuous conductance factor."""
        B = ph_areas.shape[0]
        soft = temperature is not None
        if soft and not bool(torch.as_tensor(temperature > 0).all()):
            raise ValueError("temperature must be positive")
        # Scale every fixed tube/cavity and port physically, not merely the
        # articulator geometry. Lungs are the sole unscaled pressure node;
        # their connecting neck stays 1 cm as in HumanAirway::new.
        ratio = 1.0 if anatomy is None else anatomy[0] / self.h.topo.k
        area0 = self.area0 * torch.where(self.fixed, 1.0, ratio**2)
        dx0 = self.dx0 * torch.where(self.fixed, 1.0, ratio)
        port_area0 = self.port_area0 * ratio**2
        port_length = self.port_len * torch.where(self.ba == self.h.lungs, 1.0, ratio)
        port_slit = self.port_slit * ratio
        port_extra_r = self.port_extra_r / ratio**2
        area = area0.expand(B, -1).clone()
        area[:, self.ph] = positive(ph_areas, temperature) if soft else ph_areas.clamp_min(0.0)
        area[:, self.orl] = positive(or_areas, temperature) if soft else or_areas.clamp_min(0.0)
        dx = dx0.expand(B, -1).clone()
        dx[:, self.ph] = (ph_len / self.h.n_ph)[:, None].expand(-1, self.h.n_ph)
        dx[:, self.orl] = (or_len / self.h.n_or)[:, None].expand(-1, self.h.n_or)
        port_area = port_area0.expand(B, -1).clone()
        port_area[:, self.velum_port] = positive(velum, temperature) if soft else velum.clamp_min(0.0)

        if glottis is not None:
            port_area[:, self.h.glottis_port] = positive(glottis, temperature) if soft else glottis.clamp_min(0.0)
        # soft closure: impedances at areas floored at CLOSED_AREA (finite for closed branches)
        zarea = lower_bound(area, CLOSED_AREA, temperature) if soft else area
        inv = 1.0 / zarea.clamp_min(1e-6)
        # index_select, not x[:, idx]: its backward is an index_add, not a sort-based index_put
        A_ = lambda x: x.index_select(1, self.ba)  # noqa: E731
        B_ = lambda x: x.index_select(1, self.bb)  # noqa: E731
        aa, ab = A_(area), B_(area)
        ia, ib = A_(inv), B_(inv)
        dxa, dxb = A_(dx), B_(dx)
        half_l = 0.5 * RHO * (dxa * ia + dxb * ib)
        sq = _sqrt0(zarea)
        node_r = self.wall_perim * dx * sq * inv * inv * self.visc_f  # per node half-term (×0.5·visc)
        series_r = 0.5 * VISC * (A_(node_r) + B_(node_r))
        ap = port_area
        zap = lower_bound(ap, CLOSED_AREA, temperature) if soft else ap
        ip = 1.0 / zap.clamp_min(1e-12)
        port_l = RHO * port_length * ip + self.port_half * half_l
        slit_r = 12.0 * MU * port_length * port_slit**2 * ip**3
        duct_r = VISC * 2.0 * torch.pi**0.5 * port_length * _sqrt0(zap) * ip * ip
        port_r = torch.where(self.port_slit > 0, slit_r, duct_r) + port_extra_r
        inertance = torch.where(self.is_port, port_l, half_l)
        r = torch.where(self.is_port, port_r, series_r)
        if soft:
            raw = area0.expand(B, -1).clone()  # unclamped: gradients reach areas ≤ 0 too
            raw[:, self.ph] = ph_areas
            raw[:, self.orl] = or_areas
            rawp = port_area0.expand(B, -1).clone()
            rawp[:, self.velum_port] = velum
            if glottis is not None:
                rawp[:, self.h.glottis_port] = glottis
            left, right = A_(raw), B_(raw)
            delta = left - right
            # Smooth min, used only in the opening predicate, may be negative.
            width = torch.as_tensor(temperature, dtype=delta.dtype, device=delta.device)
            minimum = 0.5 * (left + right - torch.hypot(delta, width))
            a_open = torch.where(self.is_port, rawp, minimum)
            open_ = opening(a_open - CLOSED_AREA, temperature, CLOSED_AREA)
        else:
            open_ = torch.where(self.is_port, ap >= CLOSED_AREA, torch.minimum(aa, ab) >= CLOSED_AREA)
        a_c = lower_bound(area, MIN_NODE_AREA, temperature) if soft else area.clamp_min(MIN_NODE_AREA)
        c = a_c * dx / RHO_C2
        wall_area = torch.where(self.has_wall, self.wall_perim * a_c.sqrt() * dx, torch.zeros_like(c))
        rad = area.index_select(1, self.rad_nodes)
        rad_area = lower_bound(rad, 1e-4, temperature) if soft else rad.clamp_min(1e-4)
        return Elements(inertance, r, open_, c, wall_area, rad_area)

    def shunts(self, el: Elements, freqs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return node admittance, radiation RHS, and branch impedance.

        Frequency floor 1e-3 Hz matches frequency.rs (avoids the radiation pole
        at DC). Wall parameters are mass, damping and stiffness per wall area.
        """
        w = (2 * torch.pi * freqs.clamp_min(1e-3))[None, :, None]
        jw = 1j * w
        zw = self.wall_r + 1j * (w * self.wall_m - self.wall_k / w)
        yw = torch.where(self.has_wall, el.wall_area[:, None, :] / zw, 0.0)
        a = el.rad_area[:, None, :]
        lr = 8 * RHO / (3 * torch.pi * (torch.pi * a).sqrt())
        rr = 128 * RHO * C_SOUND / (9 * torch.pi**2 * a)
        yr = 1 / rr + 1 / (jw * lr)
        y = jw * el.c[:, None, :] + yw
        y = y.index_add(2, self.rad_nodes, yr)
        y = y + self.fixed * 1e30
        radiation = (yw * self.wall_rad).index_add(2, self.rad_nodes, yr)
        z = el.r[:, None, :] + jw * el.l[:, None, :]
        return y, radiation, z

    def responses(self, el: Elements, freqs: Tensor) -> tuple[Tensor, Tensor]:
        """Radiated flow per node flow / branch pressure source: (B,F,N)/(B,F,E).

        Reciprocity solves A φ = radiation. A node injection at k radiates φ[k];
        a series pressure at a→b radiates (φ[b]−φ[a])·open/Z. The graph is a tree,
        so a Schur sweep eliminates leaves, then back-substitutes. Both sweeps
        include the nasal subtree for every input, including closed velums.
        """
        return self._solve(el, *self.shunts(el, freqs))

    def discrete_responses(
        self,
        el: Elements,
        freqs: Tensor,
        temperature: float | Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Exact z-domain linearisation of AirwayNetwork::step at internal fs.

        Branch resistance is backward Euler, walls use staggered velocity, and
        radiation is Crank–Nicolson. Continuous jω or a sine-frequency warp alone
        cannot represent all three clocks. Pressure is the POST-step value; the
        old pressure driving a branch therefore contributes z^-1 explicitly.
        This returns radiated FLOW; pressure differentiation is a separate factor.
        """
        theta = 2 * torch.pi * freqs.clamp_min(1e-3) / self.fs
        z1 = torch.exp(-1j * theta)[None, :, None]
        difference = -self.fs * torch.expm1(-1j * theta)[None, :, None]
        # The same junction CFL compliance floor used in the time stepper.
        inverse_sum = torch.zeros_like(el.c)
        inverse = el.open.to(el.c.dtype) / el.l
        inverse_sum = inverse_sum.index_add(1, self.ba, inverse).index_add(1, self.bb, inverse)
        minimum_c = inverse_sum / (self.fs**2 * 2.4)
        if temperature is None:
            effective = torch.maximum(el.c, minimum_c)
        else:
            # Match dynamics.py's compliance relaxation: temperature there is
            # dimensionless, whereas elements()/this API use cm² widths.
            effective = minimum_c + positive(el.c - minimum_c, temperature * 1e-9 / CLOSED_AREA)
        compliance = torch.where(self.junction, effective, el.c)
        wall_z = self.wall_m * difference + 0.5 * self.wall_r * (1 + z1) + self.wall_k * z1 / difference
        wall_y = torch.where(self.has_wall, el.wall_area[:, None] * z1 / wall_z, 0.0)
        area = el.rad_area[:, None]
        lr = 8 * RHO / (3 * torch.pi * (torch.pi * area).sqrt())
        rr = 128 * RHO * C_SOUND / (9 * torch.pi**2 * area)
        radiation_y = (1 + z1) / (2 * rr) + (1 + z1).square() / (4 * lr * difference)
        node_y = difference * compliance[:, None] + wall_y
        node_y = node_y.index_add(2, self.rad_nodes, radiation_y) + self.fixed * 1e30
        radiation = (wall_y * self.wall_rad).index_add(2, self.rad_nodes, radiation_y)
        impedance = (el.r[:, None] + el.l[:, None] * difference) / z1
        nodes, dipoles = self._solve(el, node_y, radiation, impedance)
        # network.rs::step drives momentum with PREVIOUS-step node pressures
        # but CURRENT-step series pressure sources:
        #   (R + L*difference)*U = z1*(P_a - P_b) + source.
        # Dividing impedance by z1 gives the correct nodal operator above.
        # Reciprocity then expresses a dipole per source/z1, so remove that
        # extra delay only from the source response. Node injections and the
        # continuous-frequency response retain their original timing.
        return nodes, dipoles / z1

    def _solve(self, el: Elements, y: Tensor, rhs: Tensor, z: Tensor) -> tuple[Tensor, Tensor]:
        """Shared reciprocal tree elimination for continuous and discrete physics."""
        yl, sl = list(y.unbind(-1)), list(rhs.unbind(-1))
        zl = z.unbind(-1)
        opening = el.open.to(el.c.dtype)[:, None, :].unbind(-1)
        for q in range(len(self.order_l) - 1, 0, -1):
            k, j, parent = self.order_l[q], self.pb_l[q], self.parent_l[q]
            t = opening[j] / (opening[j] + zl[j] * yl[k])
            yl[parent] = yl[parent] + yl[k] * t
            sl[parent] = sl[parent] + sl[k] * t
        root = self.order_l[0]
        phi = {root: sl[root] / yl[root]}
        for q in range(1, len(self.order_l)):
            k, j, parent = self.order_l[q], self.pb_l[q], self.parent_l[q]
            phi[k] = (zl[j] * sl[k] + opening[j] * phi[parent]) / (opening[j] + zl[j] * yl[k])
        zero = torch.zeros_like(phi[root])
        nodes = torch.stack([phi.get(k, zero) for k in range(self.h.topo.n_nodes)], -1)
        # Static indexing follows topology only; no data-dependent source selection.
        dipoles = nodes.index_select(-1, self.bb) - nodes.index_select(-1, self.ba)
        dipoles = dipoles * el.open[:, None, :] * self.source_branch / z
        return nodes, dipoles

    def transfer(self, el: Elements, freqs: Tensor) -> Tensor:
        """Complex U_rad/U_in (B,F), with the injection at pharynx.first."""
        return self.responses(el, freqs)[0][..., self.inject]
