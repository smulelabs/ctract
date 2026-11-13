"""Continuous extension of CHIMERA's static human geometry.

This follows the SAME contour and area laws as geometry.py. Continuous Bernoulli
visibility replaces missing-vertex/ray compaction; a fixed comparison network
replaces crossing order; section membership and interval interpolation are dense
weighted sums. All gradients are derivatives of the forward function itself.
"""

from __future__ import annotations

import math
from typing import Protocol

import torch
from torch import Tensor

from .geometry_primitives import bezier, catmull, mix
from .geometry_relaxation import Relaxation, adjacency
from .relaxation import positive, strict_opening


class GeometryReference(Protocol):
    """Constants shared by the independent hard and continuous geometry paths."""

    k: float
    pharynx: float
    radius: float
    lip_width: float
    wall_landmarks: Tensor
    palate_landmarks: Tensor
    wall: Tensor
    palate: Tensor
    n_ph: int
    n_or: int


class ContinuousGeometry:
    """Per-evaluation numerical context, with no cached state or learned weights."""

    def __init__(self, reference: GeometryReference, tau: Tensor, anatomy: Tensor | None = None) -> None:
        self.r = Relaxation(tau)
        self.k, self.pharynx = reference.k, reference.pharynx
        self.radius, self.lip_width = reference.radius, reference.lip_width
        self.wall, self.palate = reference.wall.to(tau.dtype), reference.palate.to(tau.dtype)
        if anatomy is not None:
            self.k, self.pharynx, self.radius, self.lip_width = anatomy.unbind()
            self.wall = catmull(self._template(reference.wall_landmarks.to(tau)), 4)
            self.palate = catmull(self._template(reference.palate_landmarks.to(tau)), 3)
        self.n_ph, self.n_or = reference.n_ph, reference.n_or

    def _template(self, xy: Tensor) -> Tensor:
        x, y = xy.unbind(-1)
        return (
            torch.stack((x - 0.14 * self.r.positive(x - 0.5), y + (1 - self.pharynx) * self.r.positive(7 - y)), -1)
            * self.k
        )

    def _contours(self, a: Tensor) -> tuple[Tensor, ...]:
        k, batch = self.k, a.shape[0]
        vals = a.unbind(-1)
        jaw, tbx, tby, ttx, tty, curl, root, ap, prot, spread, lab, vel, lar = vals[:13]

        def p(x: Tensor | float, y: Tensor | float) -> Tensor:
            return self._template(torch.stack((torch.zeros_like(jaw) + x, torch.zeros_like(jaw) + y), -1))

        def v(x: Tensor | float, y: Tensor | float) -> Tensor:
            return torch.stack((torch.zeros_like(jaw) + x, torch.zeros_like(jaw) + y), -1)

        condyle = p(-2.5, 10.5)
        phi = -self.r.clamp(jaw, -0.2, 3.0) * k / self.r.norm(p(6.95, 7.75) - condyle)

        def j(x: float, y: float) -> Tensor:
            dx, dy = (p(x, y) - condyle).unbind(-1)
            return condyle + v(dx * phi.cos() - dy * phi.sin(), dx * phi.sin() + dy * phi.cos())

        vo = self.r.clamp(vel / 1.2, 0.0, 1.0)
        ang = (194 + 46 * vo) * math.pi / 180
        u = torch.linspace(0, 1, 11, device=a.device, dtype=a.dtype)[None, :]
        flap = p(1.0, 9.75)[:, None, :] + 2.8 * k * u[..., None] * v(ang.cos(), ang.sin())[:, None, :]
        sag = (u * math.pi).sin() * k * (0.3 * vo[:, None] - 0.12)
        flap = flap + torch.stack((torch.zeros_like(sag), -sag), -1)
        flap = flap + ((0.48 - 0.2 * u) * k)[..., None] * v(-ang.sin(), ang.cos())[:, None, :]
        flap = mix(flap, self.palate[0], self.r.positive(1 - u / 0.35).square()[..., None])
        wall = self.wall.expand(batch, -1, -1)
        uv_y = flap[:, -1, 1] - 0.3 * k
        weights = self.r.interval_weights(wall[..., 1], uv_y[:, None])
        u_wall = self.r.clamp((uv_y[:, None] - wall[:, :-1, 1]) / (wall[:, 1:, 1] - wall[:, :-1, 1]), 0.0, 1.0)
        edge = (weights[:, 0, :, None] * mix(wall[:, :-1], wall[:, 1:], u_wall[..., None])).sum(1)
        keep = self.r.gate(uv_y[:, None] - wall[..., 1])
        wall = mix(edge[:, None, :], wall, keep[..., None])
        prot, spread, lab, ap = (
            self.r.clamp(prot, -0.2, 1.6) * k,
            self.r.clamp(spread, 0.0, 1.0),
            self.r.clamp(lab, 0.0, 1.0),
            ap * k,
        )
        mid_y = (7.85 - 0.3 * jaw) * k
        fx = p(8.15 + 0.12 * (1 - spread), 0.0)[:, 0] + prot * 0.95
        ul = v(fx - 0.15 * k, mid_y + ap * 0.35 + lab * 0.35 * k)
        ll = v(fx - 0.25 * k, mid_y - ap * 0.65)
        inc = self.palate[-1]
        ll = mix(ll, v(inc[0] + 0.05 * k, inc[1] - self.r.maximum(ap, -0.1)), lab[:, None])
        ul = mix(ul, v(inc[0] + 0.55 * k, inc[1] + 0.45 * k), (lab * 0.7)[:, None])
        ub = p(7.32, 8.35)
        upper = bezier([ub, ub + v(0.3 * k + prot * 0.4, 0.05 * k), ul + v(-0.35 * k, 0.12 * k), ul], 7)
        roof = torch.cat((flap.flip(1), self.palate[1:].expand(batch, -1, -1)), 1)
        outer = torch.cat((wall, roof, upper), 1)
        b, b0 = p(tbx, tby), p(2.9, 6.85)
        r, lar = self.radius * k, lar * k
        vallecula = p(0.75, 3.3) + v(0.45 * (b[:, 0] - b0[:, 0]) + root * k, 0.3 * (b[:, 1] - b0[:, 1]) + 0.55 * lar)
        epi_base = p(0.3, 2.4) + v(0.0, lar)
        epi_tip = p(0.0, 4.5) + v(0.35 * (vallecula[:, 0] - 0.75 * k), lar * 0.8 + 0.15 * (b[:, 1] - b0[:, 1]))
        thb = 200 * math.pi / 180
        pb = b + v(r * math.cos(thb), r * math.sin(thb))
        tanb = v(math.sin(thb), -math.cos(thb))
        hr = self.r.norm(vallecula - pb) / 3
        rt = bezier([vallecula, vallecula + v(0.08 * k, hr), pb - tanb * hr[:, None], pb], 9)
        tip = p(ttx, tty)
        delta = tip - b
        dt = self.r.norm(delta)
        normalizer = self.r.norm(delta)
        projected = b + delta / normalizer[:, None] * (r * 1.05)
        tip = mix(tip, projected, self.r.gate(r * 1.05 - dt)[:, None])
        delta = tip - b
        dt = self.r.norm(delta)
        # Analytic Abel sum of the wrapped angle. Its atan2 denominator stays
        # strictly positive for positive temperature, avoiding a branch cut.
        # It converges to thb-remainder(thb-thf,2*pi) off that cut; a vanishing
        # negative bias resolves the cut in Rust's direction (see derivation).
        angle = torch.acos(r / self.r.maximum(dt, r * 1.0001)) * self.r.gate(dt - r * 1.0001)
        ux, uy = (delta / dt[:, None]).unbind(-1)
        ca, sa = angle.cos(), angle.sin()
        c, sn = ux * ca - uy * sa, uy * ca + ux * sa
        bias = self.r.width * self.r.width.log()
        cb, sb = torch.cos(bias - thb), torch.sin(bias - thb)
        c, sn = c * cb - sn * sb, sn * cb + c * sb
        damping = torch.exp(-self.r.width)
        thf = thb - math.pi - 2 * torch.atan2(damping * sn, 1 - damping * c)
        coincidence = delta.square().sum(-1) / (delta.square().sum(-1) + (self.r.width * 1e-3).square())
        thf = thf * coincidence
        theta = thb + (thf - thb)[:, None] * torch.linspace(0, 1, 32, device=a.device, dtype=a.dtype)
        arc = b[:, None, :] + r * torch.stack((theta.cos(), theta.sin()), -1)
        pf = arc[:, -1]
        tanf = v(thf.sin(), -thf.cos())
        bl = self.r.norm(pf - tip)
        curl = self.r.clamp(curl, 0.0, 1.0)
        ctrl = pf + tanf * (bl * 0.5)[:, None] + v(0.0, (0.18 + 0.5 * curl) * k)
        blade = bezier([pf, ctrl, tip], 7)

        def dip(points: Tensor) -> Tensor:
            coordinate = (points[..., 0] - b[:, None, 0]) / self.r.maximum(tip[:, 0] - b[:, 0], 1e-6)[:, None]
            u = self.r.clamp(coordinate, 0.0, 1.0)
            w = (math.pi * self.r.maximum(u, 1e-12).pow(0.8)).sin().square()
            membership = self.r.gate(coordinate) * self.r.gate(1 - coordinate)
            return points + torch.stack(
                (torch.zeros_like(w), -self.r.positive(a[:, 16:17]) * 0.9 * k * w * membership), -1
            )

        arc = dip(arc)
        blade = torch.cat((blade[:, :1], dip(blade[:, 1:-1]), blade[:, -1:]), 1)
        fren = j(5.7, 6.15)
        under = bezier([tip, tip + v(-0.3 * k - curl * 0.4 * k, -0.6 * k), fren], 6)
        floor = torch.stack((fren, j(6.2, 6.25), j(6.45, 6.55)), 1)
        teeth = torch.stack((j(6.45, 6.55), j(6.72, 7.15), j(6.95, 7.75), j(7.12, 7.45)), 1)
        lb = j(7.12, 7.45)
        lower = bezier([lb, lb + v(0.25 * k + prot * 0.3, -0.05 * k), ll + v(-0.3 * k, -0.12 * k), ll], 7)
        back = torch.cat((rt, arc[:, 1:]), 1)
        yroot = epi_tip[:, 1] + 0.05 * k
        above = self.r.gate(back[..., 1] - yroot[:, None])
        previous = torch.cat((torch.ones_like(above[:, :1]), (1 - above[:, :-1]).cumprod(-1)), -1)
        first = above * previous
        fraction = self.r.clamp(
            (yroot[:, None] - back[:, :-1, 1]) / self.r.maximum(back[:, 1:, 1] - back[:, :-1, 1], 1e-9), 0.0, 1.0
        )
        starts = mix(back[:, :-1], back[:, 1:], fraction[..., None])
        absent = (1 - above).prod(-1)
        start = (first[:, 0] + absent)[:, None] * epi_tip + (first[:, 1:, None] * starts).sum(1)
        index = torch.arange(back.shape[1], device=a.device)
        keep = first.cumsum(-1) - first * (index >= 9)
        back = mix(start[:, None, :], back, keep[..., None])
        inner = torch.cat(
            (epi_base[:, None], epi_tip[:, None], back, blade[:, 1:], under[:, 1:], floor[:, 1:], teeth[:, 1:], lower),
            1,
        )
        outline = torch.cat((rt, arc[:, 1:], blade[:, 1:], under[:, 1:], floor[:, 1:], rt[:, :1]), 1)
        tip_points = torch.cat((blade[:, 2:], under[:, 1:3]), 1)
        return outer, inner, roof, outline, arc, tip_points, tip, ul, ll, lab, ap, spread

    def _measure(
        self, origin: Tensor, direction: Tensor, outer: Tensor, inner: Tensor, fan: bool
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Continuous ray entry/exit/tissue pairing with exact off-tie limit."""
        r, k = self.r, self.k
        log_sigmoid = torch.nn.functional.logsigmoid
        topology_width = r.width * r.tau
        penalty = r.width.sqrt()
        t, log_visible, _ = r.crossings(origin, direction, outer)
        log_visible = (
            log_visible + log_sigmoid(t / topology_width) + log_sigmoid(((8 if fan else 9) * k - t) / topology_width)
        )
        bounded = r.clamp(t, 0.0, (8 if fan else 9) * k)
        cost = bounded - penalty * log_visible
        weights = torch.softmax(-cost / r.width, -1)
        tout = (weights * bounded).sum(-1)
        visible = log_visible.exp()
        exists = 1 - (1 - visible).prod(-1)
        tout = tout * exists
        t, log_visible, exits = r.crossings(origin, direction, inner)
        log_visible = (
            log_visible
            + log_sigmoid(t / topology_width)
            + log_sigmoid((tout[..., None] + 1.5 * k - t) / topology_width)
        )
        bounded = r.clamp(t, 0.0, tout[..., None] + 1.5 * k)
        if fan:
            # For ordering, a CDF turns log visibility into a bounded absence
            # penalty. Its slope grows as tau cools: true crossings keep their
            # coordinates, absent candidates go to a fixed trailing sentinel.
            visible = torch.exp(log_visible)
            cost = mix(100 * k, bounded, visible)
            # Vertex ties may split probability between incident segments;
            # normalise their order score before sorting. At cold temperature
            # off ties, visibility is exactly its hard 0/1 limit.
            ts, ex = r.sort(cost, exits)
            is_exit = r.gate(50 * k - ts[..., 0]) * ex[..., 0]
            tb, log_vb, _ = r.crossings(origin, -direction, inner)
            log_vb = log_vb + log_sigmoid(tb / topology_width) + log_sigmoid((2.5 * k - tb) / topology_width)
            bounded_b = r.clamp(tb, 0.0, 2.5 * k)
            back_weight = torch.softmax(-(bounded_b - penalty * log_vb) / r.width, -1)
            back_exists = 1 - (1 - log_vb.exp()).prod(-1)
            tb = mix(2.5 * k, (back_weight * bounded_b).sum(-1), back_exists)
            tin = mix(-tb, ts[..., 0], is_exit)
            skip, tissue = is_exit, torch.zeros_like(tin)
            for i in range(ts.shape[-1]):
                entry = (1 - skip) * (1 - ex[..., i]) * r.gate(tout - ts[..., i])
                end = ts[..., i + 1] if i + 1 < ts.shape[-1] else tout + 1.5 * k
                tissue = tissue + entry * (r.minimum(end, tout) - ts[..., i])
                skip = entry
            tin = tin + tissue
        else:
            weights = torch.softmax(-(-bounded - penalty * log_visible) / r.width, -1)
            present = 1 - (1 - log_visible.exp()).prod(-1)
            tin = mix(r.positive(tout - 3.5 * k), (weights * bounded).sum(-1), present)
        pin = origin + direction * r.minimum(tin, tout)[..., None]
        pout = origin + direction * tout[..., None]
        return pin, pout, r.positive(tout - tin), exists

    def compute(self, a: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Keep all 49 rays; integrate continuous consecutive-ray adjacency."""
        r, k = self.r, self.k
        outer, inner, roof, outline, arc, tip_pts, tip, ul, ll, lab, ap, spread = self._contours(a)

        def const(x: list[float]) -> Tensor:
            return torch.stack([torch.zeros_like(a[:, 0]) + value for value in x], -1)

        y = mix(
            (a[:, 12] * k + 7 * (1 - self.pharynx) * k + 2.4 * k * self.pharynx + 0.12 * k)[:, None],
            const([(7 - 0.05 * self.pharynx) * k]),
            torch.arange(12, device=a.device, dtype=a.dtype) / 12,
        )
        origin = torch.stack((torch.zeros_like(y) + 4.6 * k, y), -1)
        ph = self._measure(origin, const([-1.0, 0.0])[:, None].expand_as(origin), outer, inner, False)
        u = torch.linspace(0, 1, 31, device=a.device, dtype=a.dtype)
        angle = (180 - 144 * u) * math.pi / 180
        w = ((u - 0.45) / 0.55).clamp_min(0)  # Fixed grid coordinate, not fitted data.
        c1 = self._template(const([2.6, 6.95]))[:, None]
        c2 = self._template(const([5.0, 6.85]))[:, None]
        origin = mix(c1, c2, (w.square() * (3 - 2 * w))[None, :, None])
        fan = self._measure(origin, torch.stack((angle.cos(), angle.sin()), -1).expand_as(origin), outer, inner, True)
        x0 = self.palate[-1, 0] + 0.03 * k
        x1 = r.maximum(x0 + 0.3 * k, r.minimum(ul[:, 0], ll[:, 0]) - 0.04 * k)
        x = x0 + (x1 - x0)[:, None] * torch.linspace(0, 1, 6, device=a.device, dtype=a.dtype)
        origin = torch.stack((x, torch.zeros_like(x) + 4 * k), -1)
        fi, fo, fd, fv = self._measure(origin, const([0.0, 1.0])[:, None].expand_as(origin), outer, inner, False)
        face = r.minimum(ll[:, 1:2] - (x - ll[:, 0:1]) * 4 / 3, fo[..., 1])
        opening = r.ramp(0.7, 1.0, lab[:, None]) * r.ramp(0.0, 0.1 * k, x - ll[:, 0:1])
        yin = mix(face, fo[..., 1] - 3.5 * k, opening)
        weight = r.gate(x - ll[:, 0:1])
        fi = mix(fi, torch.stack((x, yin), -1), weight[..., None])
        fd = mix(fd, fo[..., 1] - yin, weight)
        pin, pout = torch.cat((ph[0], fan[0], fi), 1), torch.cat((ph[1], fan[1], fo), 1)
        d, visibility = torch.cat((ph[2], fan[2], fd), 1), torch.cat((ph[3], fan[3], fv), 1)
        mid = (pin + pout) * 0.5
        index = torch.arange(49, device=a.device)
        region = torch.where(index < 19, 1, torch.where(index < 43, 2, 3))  # Fixed ray regions.
        tip_c = r.select_min(r.project(roof, tip_pts)[2])
        body_c = r.select_min(r.project(roof, arc)[2])
        half = 0.1 * k * (2 - r.ramp(-0.06 * k, 0.0, r.minimum(tip_c, body_c)))
        s, pr_dist, _ = r.project(roof, pout)
        on_roof = 1 - r.ramp(0.15 * k, 0.25 * k, pr_dist)
        dc = r.signed_gap(outline, pout)
        search = r.gate(0.6 * k - (dc - half[:, None] - pr_dist))
        roof_len = r.cumulative(roof)[:, -1:]
        for fraction in (-1.0, -0.5, 0.5, 1.0):
            q = r.point_along(roof, r.clamp(s + fraction * half[:, None], 0.0, roof_len))
            dc = mix(dc, r.minimum(dc, r.signed_gap(outline, q)), search)
        contact = (1 - r.ramp(0.15 * k, 0.6 * k, dc)) * on_roof
        cap = (1 - r.ramp(0.45 * k, 0.6 * k, dc)) * on_roof
        dp = r.positive(dc)
        d = mix(mix(d, r.minimum(d, dp), cap), dp, contact)
        edge_x = r.minimum(ul[:, 0:1], ll[:, 0:1])
        lab_open = r.ramp(0.0, 0.5, lab[:, None])
        ap_eff = ap[:, None] + 0.3 * k * lab_open / r.maximum(1 - lab_open, 1e-12)
        lip_gap = r.positive(ap_eff + r.positive(edge_x - mid[..., 0]).square() / (1.2 * k))
        lip_weight = (region == 3) * r.gate(mid[..., 0] - self.palate[-1, 0]) * r.gate(1 - lab_open)
        d = mix(d, r.minimum(d, lip_gap), lip_weight)
        # Positive tails ensure the fractional power and its backward are finite
        # even after deep contact. These are true forward derivatives, not STEs.
        # This is a roundoff/invalid-branch guard, not a second contact
        # relaxation. Using the contact width again would create an artificial
        # width/2 airway floor and drown the deep-contact reopening derivative.
        d = positive(d, r.width * r.tau * 1e-3)
        dental = 2 * d.pow(1.4)
        ahead = r.ramp(0.0, 0.4 * k, mid[..., 0] - tip[:, 0:1])
        oral = (2.6 - ahead) * d.pow(1.4)
        lip_u = r.clamp((mid[..., 0] - self.palate[-1, 0]) / (0.45 * k), 0.0, 1.0)
        lip_w = (1.1 + (self.lip_width - 1.1) * spread) * k
        front_area = mix(dental, math.pi / 4 * d * lip_w[:, None], lip_u)
        area = torch.where(region == 1, dental, torch.where(region == 2, oral, front_area))
        pairs, first, last = adjacency(visibility)
        # Rust's sequential spike filter, extended to uncertain ray adjacency.
        # Conditional preceding/succeeding distributions have no division by
        # visibility and no 0/0 when a ray disappears.
        filtered = []
        for i in range(49):
            head, tail = visibility[:, :i], visibility[:, i + 1 :]
            left_weight = (
                head
                * torch.cat(((1 - head[:, 1:]).flip(-1).cumprod(-1).flip(-1), torch.ones_like(visibility[:, :1])), -1)
                if i
                else head
            )
            right_weight = (
                tail * torch.cat((torch.ones_like(visibility[:, :1]), (1 - tail[:, :-1]).cumprod(-1)), -1)
                if i < 48
                else tail
            )
            left = (left_weight * torch.stack(filtered, -1)).sum(-1) if i else torch.zeros_like(area[:, 0])
            right = (right_weight * area[:, i + 1 :]).sum(-1)
            limit = 1.35 * r.maximum(left, right)
            mass = left_weight.sum(-1) * right_weight.sum(-1)
            filtered.append(mix(area[:, i], r.minimum(area[:, i], limit), mass))
        area = torch.stack(filtered, -1)
        lateral = (
            (region >= 2) * r.gate(1.4 * k - r.norm(mid - tip[:, None])) * strict_opening(a[:, 15:16], r.width, 1.0)
        )
        area = mix(area, r.maximum(area, a[:, 15:16] * 0.35 * k * k), lateral)
        distances = r.norm(mid[:, :, None, :] - mid[:, None, :, :])
        increment = (pairs * distances).sum(-2)
        grid_s = increment.cumsum(-1) + 2.4 * k * self.pharynx
        lip_s = (last * grid_s).sum(-1)
        oral_visibility = visibility[:, 19:]
        oral_first = oral_visibility * torch.cat(
            (torch.ones_like(oral_visibility[:, :1]), (1 - oral_visibility[:, :-1]).cumprod(-1)), -1
        )
        ph_len = (oral_first * grid_s[:, 19:]).sum(-1)
        front_v = visibility[:, 43:]
        front_first = front_v * torch.cat((torch.ones_like(front_v[:, :1]), (1 - front_v[:, :-1]).cumprod(-1)), -1)
        teeth_s = (front_first * grid_s[:, 43:]).sum(-1) + (1 - front_v).prod(-1) * lip_s
        return grid_s, area, visibility, ph_len, lip_s, teeth_s

    def resample(
        self,
        a: Tensor,
        grid_s: Tensor,
        area: Tensor,
        visibility: Tensor,
        start: Tensor,
        end: Tensor,
        n: int,
    ) -> tuple[Tensor, Tensor]:
        """Dense consecutive-ray interpolation and inclusive section sealing."""
        r, k = self.r, self.k
        step = (end - start) / n
        index = torch.arange(n, device=a.device, dtype=a.dtype)
        offsets = (torch.arange(8, device=a.device, dtype=a.dtype) + 0.5) / 8
        query = (start[:, None, None] + step[:, None, None] * (index[:, None] + offsets)).flatten(1)
        pairs, first, last = adjacency(visibility)
        # Static pair wiring, independent of articulator values.
        i, j = torch.triu_indices(49, 49, offset=1, device=a.device)
        left_s, right_s = grid_s[:, i], grid_s[:, j]
        u = r.clamp((query[..., None] - left_s[:, None]) / r.maximum(right_s - left_s, 1e-9)[:, None], 0.0, 1.0)
        weight = pairs[:, i, j][:, None] * r.gate(query[..., None] - left_s[:, None])
        weight = weight * r.gate(right_s[:, None] - query[..., None])
        values = mix(area[:, i][:, None], area[:, j][:, None], u)
        left = first[:, None] * r.gate(grid_s[:, None] - query[..., None])
        right = last[:, None] * r.gate(query[..., None] - grid_s[:, None])
        total = weight.sum(-1) + left.sum(-1) + right.sum(-1)
        interp = ((weight * values).sum(-1) + ((left + right) * area[:, None]).sum(-1)) / total
        epi_len = 2.4 * k * self.pharynx
        u = query / epi_len
        vent = 0.95 * (1 - 0.75 * r.clamp(a[:, 14:15], 0.0, 1.0))
        top = mix(1.9, 0.35, r.clamp(a[:, 13:14], 0.0, 1.0))

        def epi(x: Tensor | float) -> Tensor:
            x = x if isinstance(x, Tensor) else a.new_full((), x)
            return mix(mix(1.3, vent, r.gate(x - 0.3)), mix(1.2, top, (x - 0.5) / 0.5), r.gate(x - 0.5))

        ep = epi(r.clamp(u, 0.0, 1.0))
        for centre in (0.3, 0.5):
            membership = r.gate(u - (centre - 0.03)) * r.gate(centre + 0.03 - u)
            ep = mix(ep, mix(epi(centre - 0.03), epi(centre + 0.03), (u - (centre - 0.03)) / 0.06), membership)
        ep = ep * k * k
        first_area = (first * area).sum(-1, keepdim=True)
        ep = mix(
            ep,
            mix(ep, first_area, r.clamp((query - 0.9 * epi_len) / (0.1 * epi_len), 0.0, 1.0)),
            r.gate(query - 0.9 * epi_len),
        )
        samples = mix(ep, interp, r.gate(query - epi_len)).reshape(-1, n, 8)
        # The epilarynx's extrapolated inactive formula may become negative at a
        # finite temperature; a smooth positive part keeps harmonic means valid.
        samples = positive(samples, r.tau.square() * 1e-12)
        harmonic = 8 / (1 / samples).sum(-1)
        lo = start[:, None] + step[:, None] * index
        hi = start[:, None] + step[:, None] * (index + 1) + (index == n - 1) * 1e-6
        # An outward vanishing bias includes a point tied to BOTH endpoints.
        bias = r.width.sqrt()
        membership = visibility[:, None] * r.gate(grid_s[:, None] - lo[..., None] + bias)
        membership = membership * r.gate(hi[..., None] - grid_s[:, None] + bias)
        absent_cost = 1 + area.sum(-1, keepdim=True)
        point_cost = mix(absent_cost[:, None], area[:, None], membership)
        minimum = r.select_min(torch.cat((samples, point_cost), -1))
        seal_width = r.tau * 2e-5
        # One minimum gate instead of a product of many closed gates: a long
        # closure gets algebraic reopening tails, not an exponentially small
        # product. Positive bias gives Rust's inclusive <=1e-5 closure tie.
        gate = torch.sigmoid(2 * torch.asinh((minimum - 1e-5) / seal_width) + r.tau.log())
        return harmonic * gate, r.minimum(minimum, harmonic) * gate

    def __call__(self, artic: Tensor, lo: Tensor, hi: Tensor) -> dict[str, Tensor]:
        shape = artic.shape[:-1]
        a = self.r.control_clamp(artic, lo, hi).reshape(-1, 17)
        grid, area, visibility, ph_len, lip_s, teeth_s = self.compute(a)
        ph, ph_min = self.resample(a, grid, area, visibility, ph_len * 0, ph_len, self.n_ph)
        oral, oral_min = self.resample(a, grid, area, visibility, ph_len, lip_s, self.n_or)
        result = {
            "ph": ph,
            "orl": oral,
            "ph_min": ph_min,
            "or_min": oral_min,
            "teeth_s": teeth_s,
            "ph_len": ph_len,
            "or_len": lip_s - ph_len,
            "velum": self.r.positive(a[:, 11]) * self.k * self.k,
        }
        return {key: value.reshape(shape + value.shape[1:]) for key, value in result.items()}
