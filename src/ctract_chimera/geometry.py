"""Static HumanVocalTract port (CHIMERA anatomy/human.rs::compute and ::resample).

Hard mode follows Rust. Fit mode uses continuous visibility, ordering, contact
and resampling weights; see geometry_smooth.py and ARCHITECTURE.md.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .anatomy import ARTIC_MAX, ARTIC_MIN, Anatomy
from .geometry_primitives import (
    bezier,
    catmull,
    crossings,
    cumulative,
    gather,
    length,
    mix,
    point_along,
    project,
    signed_gap,
    smooth,
)
from .topology import SectionCounts, build_human


class ExactGeometry(nn.Module):
    """Raw (...,17) anatomy → renderer geometry, in cm and cm²; no learned weights.

    Supports float32/64 on CPU/CUDA. Hard mode preserves discrete topology;
    fit mode differentiates a continuous extension of the same geometry laws.
    """

    def __init__(
        self, anatomy: Anatomy = Anatomy(), internal_fs: float = 96000.0, counts: SectionCounts | None = None
    ) -> None:
        super().__init__()
        self.anatomy = anatomy
        self.k, self.pharynx = anatomy.scale, anatomy.pharynx
        self.radius, self.lip_width = anatomy.tongue_radius, anatomy.lip_width
        topology = build_human(internal_fs, anatomy, counts)
        self.n_ph, self.n_or = topology.n_ph, topology.n_or
        self.register_buffer("lo", torch.tensor(ARTIC_MIN, dtype=torch.float64))
        self.register_buffer("hi", torch.tensor(ARTIC_MAX, dtype=torch.float64))
        wall = [
            (-1.0, -1.5),
            (-1.1, 1.5),
            (-1.2, 3.6),
            (-1.3, 5.6),
            (-1.38, 7.5),
            (-1.33, 8.9),
            (-1.05, 10.1),
            (-0.45, 10.9),
            (0.4, 11.3),
        ]
        palate = [
            (1.0, 9.6),
            (1.8, 9.92),
            (2.7, 10.18),
            (3.6, 10.3),
            (4.4, 10.25),
            (5.1, 10.05),
            (5.7, 9.72),
            (6.15, 9.36),
            (6.5, 9.05),
            (6.78, 8.7),
            (6.98, 8.3),
            (7.12, 7.95),
        ]
        self.register_buffer("wall_landmarks", torch.tensor(wall, dtype=torch.float64))
        self.register_buffer("palate_landmarks", torch.tensor(palate, dtype=torch.float64))
        self.register_buffer("wall", catmull(self._template(torch.tensor(wall, dtype=torch.float64)), 4))
        self.register_buffer("palate", catmull(self._template(torch.tensor(palate, dtype=torch.float64)), 3))

    def _template(self, xy: Tensor) -> Tensor:
        x, y = xy.unbind(-1)
        return torch.stack(
            (
                torch.where(x > 0.5, 0.5 + (x - 0.5) * 0.86, x) * self.k,
                torch.where(y < 7, 7 - (7 - y) * self.pharynx, y) * self.k,
            ),
            -1,
        )

    def frame(
        self, artic: Tensor, mode: str = "hard", temperature: float | Tensor = 1.0, anatomy: Tensor | None = None
    ) -> dict[str, Tensor]:
        """Geometry plus true sub-section minima and incisor arclength (cm).

        Minima drive CHIMERA's nonlinear jet losses; teeth_s drives obstacle
        noise. They must not be substituted by harmonic section means or a
        guessed fixed distance from the lips.
        """
        return self._evaluate(artic, mode, temperature, anatomy, True)

    def forward(
        self, artic: Tensor, mode: str = "hard", temperature: float | Tensor = 1.0, anatomy: Tensor | None = None
    ) -> dict[str, Tensor]:
        """Articulators → section areas/lengths and velum port."""
        return self._evaluate(artic, mode, temperature, anatomy, False)

    def _evaluate(
        self,
        artic: Tensor,
        mode: str = "hard",
        temperature: float | Tensor = 1.0,
        anatomy: Tensor | None = None,
        details: bool = False,
    ) -> dict[str, Tensor]:
        if artic.dtype not in (torch.float32, torch.float64):
            raise TypeError("geometry requires float32 or float64")
        if artic.shape[-1] != 17:
            raise ValueError("geometry requires 17 articulators")
        if mode not in ("hard", "fit"):
            raise ValueError("geometry mode must be hard or fit")
        if not bool(torch.isfinite(artic).all()):
            raise ValueError("non-finite articulators")
        if not isinstance(temperature, Tensor) and not math.isfinite(temperature):
            raise ValueError("non-finite temperature")
        tau = (
            temperature.to(device=artic.device, dtype=artic.dtype)
            if isinstance(temperature, Tensor)
            else artic.new_full((), temperature)
        )
        if not bool(torch.isfinite(tau).all() & (tau > 0).all()):
            raise ValueError("temperature must be finite and positive")
        if anatomy is not None:
            if anatomy.shape != (4,) or not bool(torch.isfinite(anatomy).all() & (anatomy > 0).all()):
                raise ValueError("continuous anatomy must be four finite positive scalars")
            if mode != "fit":
                raise ValueError("tensor anatomy requires fit mode; use Anatomy configuration for the hard oracle")
        if mode == "fit":
            from .geometry_smooth import ContinuousGeometry

            # Geometry has near-parallel intersections and differences of clamp
            # tails. Evaluate them in fp64 even for fp32 controls, then cast the
            # public outputs back. These casts carry exact autograd edges; they
            # avoid losing small reopening sensitivities to fp32 cancellation.
            result = ContinuousGeometry(
                self, tau.to(torch.float64), None if anatomy is None else anatomy.to(torch.float64)
            )(artic.to(torch.float64), self.lo.to(torch.float64), self.hi.to(torch.float64))
            keys = result if details else ("ph", "orl", "ph_len", "or_len", "velum")
            return {key: result[key].to(artic.dtype) for key in keys}
        shape = artic.shape[:-1]
        a = torch.maximum(self.lo, torch.minimum(self.hi, artic)).reshape(-1, 17)
        grid_s, areas, split, teeth_s = self._compute(a)
        ph_len = grid_s.gather(-1, split[:, None]).squeeze(-1)
        or_len = grid_s[:, -1] - ph_len
        ph, ph_min = self._resample(a, grid_s, areas, ph_len * 0, ph_len, self.n_ph)
        oral, oral_min = self._resample(a, grid_s, areas, ph_len, grid_s[:, -1], self.n_or)
        result = {
            "ph": ph,
            "orl": oral,
            "ph_len": ph_len,
            "or_len": or_len,
            "velum": self._positive(a[:, 11]) * self.k * self.k,
        }
        if details:
            result.update(ph_min=ph_min, or_min=oral_min, teeth_s=teeth_s)
        result = {key: value.reshape(shape + value.shape[1:]) for key, value in result.items()}
        return result

    @staticmethod
    def _positive(x: Tensor) -> Tensor:
        return x.clamp_min(0)

    def _contours(self, a: Tensor) -> tuple[Tensor, ...]:
        k, batch = self.k, a.shape[0]
        vals = a.unbind(-1)
        jaw, tbx, tby, ttx, tty, curl, root, ap, prot, spread, lab, vel, lar = vals[:13]

        def p(x, y):
            return self._template(torch.stack((torch.zeros_like(jaw) + x, torch.zeros_like(jaw) + y), -1))

        def v(x, y):
            return torch.stack((torch.zeros_like(jaw) + x, torch.zeros_like(jaw) + y), -1)

        condyle = p(-2.5, 10.5)
        phi = -jaw.clamp(-0.2, 3.0) * k / length(p(6.95, 7.75) - condyle)

        def j(x: float, y: float) -> Tensor:
            dx, dy = (p(x, y) - condyle).unbind(-1)
            return condyle + v(dx * phi.cos() - dy * phi.sin(), dx * phi.sin() + dy * phi.cos())

        vo = (vel / 1.2).clamp(0, 1)
        ang = (194 + 46 * vo) * math.pi / 180
        u = torch.linspace(0, 1, 11, device=a.device, dtype=a.dtype)[None, :]
        flap = p(1.0, 9.75)[:, None, :] + 2.8 * k * u[..., None] * v(ang.cos(), ang.sin())[:, None, :]
        sag = (u * math.pi).sin() * k * (0.3 * vo[:, None] - 0.12)
        flap = flap + torch.stack((torch.zeros_like(sag), -sag), -1)
        flap = flap + ((0.48 - 0.2 * u) * k)[..., None] * v(-ang.sin(), ang.cos())[:, None, :]
        flap = mix(flap, self.palate[0], (1 - u / 0.35).clamp_min(0).square()[..., None])
        wall = self.wall.expand(batch, -1, -1)
        uv_y = flap[:, -1, 1] - 0.3 * k
        idx = (wall[..., 1] < uv_y[:, None]).sum(-1).clamp(1, wall.shape[1] - 1)
        wa, wb = gather(wall, (idx - 1)[:, None]).squeeze(1), gather(wall, idx[:, None]).squeeze(1)
        edge = mix(wa, wb, ((uv_y - wa[:, 1]) / (wb[:, 1] - wa[:, 1]).clamp_min(1e-9)).clamp(0, 1)[:, None])
        wall = torch.where(
            (torch.arange(wall.shape[1], device=a.device)[None, :] < idx[:, None])[..., None], wall, edge[:, None, :]
        )
        prot, spread, lab, ap = prot.clamp(-0.2, 1.6) * k, spread.clamp(0, 1), lab.clamp(0, 1), ap * k
        mid_y = (7.85 - 0.3 * jaw) * k
        fx = p(8.15 + 0.12 * (1 - spread), 0.0)[:, 0] + prot * 0.95
        ul = v(fx - 0.15 * k, mid_y + ap * 0.35 + lab * 0.35 * k)
        ll = v(fx - 0.25 * k, mid_y - ap * 0.65)
        inc = self.palate[-1]
        ll = mix(ll, v(inc[0] + 0.05 * k, inc[1] - ap.clamp_min(-0.1)), lab[:, None])
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
        hr = length(vallecula - pb) / 3
        rt = bezier([vallecula, vallecula + v(0.08 * k, hr), pb - tanb * hr[:, None], pb], 9)
        tip = p(ttx, tty)
        delta = tip - b
        dt = length(delta)
        # Rust norm(0) uses denominator 1. A tiny clamp preserves its zero value
        # but introduces a 1e30 derivative that can overflow fp32 optimizer moments.
        normalizer = torch.where(dt > 0, dt, torch.ones_like(dt))
        tip = torch.where((dt < r * 1.05)[:, None], b + delta / normalizer[:, None] * (r * 1.05), tip)
        delta = tip - b
        dt = length(delta)
        # Rust's zero-vector fallback has no tangent. Guard acos even in inactive branches.
        thf = torch.atan2(delta[:, 1], torch.where(dt > 0, delta[:, 0], 1.0))
        thf = thf + torch.where(dt > r * 1.0001, torch.acos((r / dt.clamp_min(r * 1.0001)).clamp(-1, 1)), 0.0)
        thf = thb - torch.remainder(thb - thf, 2 * math.pi)
        theta = thb + (thf - thb)[:, None] * torch.linspace(0, 1, 32, device=a.device, dtype=a.dtype)
        arc = b[:, None, :] + r * torch.stack((theta.cos(), theta.sin()), -1)
        pf = arc[:, -1]
        tanf = v(thf.sin(), -thf.cos())
        bl = length(pf - tip)
        curl = curl.clamp(0, 1)
        ctrl = pf + tanf * (bl * 0.5)[:, None] + v(0.0, (0.18 + 0.5 * curl) * k)
        blade = bezier([pf, ctrl, tip], 7)

        def dip(points: Tensor) -> Tensor:
            u = ((points[..., 0] - b[:, None, 0]) / (tip[:, 0] - b[:, 0]).clamp_min(1e-6)[:, None]).clamp(0, 1)
            # pow derivative at zero needs protection even though the endpoint dip is zero.
            w = torch.where((u > 0) & (u < 1), (math.pi * u.clamp_min(1e-12).pow(0.8)).sin().square(), 0.0)
            return points + torch.stack((torch.zeros_like(w), -a[:, 16:17].clamp_min(0) * 0.9 * k * w), -1)

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
        above = back[..., 1] > yroot[:, None]
        first = torch.where(above, torch.arange(back.shape[1], device=a.device), back.shape[1]).min(-1).values
        prev = (first - 1).clamp(0, back.shape[1] - 1)
        nxt = first.clamp_max(back.shape[1] - 1)
        ba, bb = gather(back, prev[:, None]).squeeze(1), gather(back, nxt[:, None]).squeeze(1)
        start = mix(ba, bb, ((yroot - ba[:, 1]) / (bb[:, 1] - ba[:, 1]).clamp_min(1e-9)).clamp(0, 1)[:, None])
        # When no root/arc point rises above the epiglottis, Rust omits the back entirely.
        start = torch.where((first == back.shape[1])[:, None], epi_tip, start)
        start = torch.where((first == 0)[:, None], epi_tip, start)
        back = torch.where(
            (torch.arange(back.shape[1], device=a.device)[None, :] >= (first + (first >= 9))[:, None])[..., None],
            back,
            start[:, None, :],
        )
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
        k = self.k
        t, valid, _ = crossings(origin, direction, outer)
        valid = valid & (t >= 0) & (t <= (8 if fan else 9) * k)
        tout = torch.where(valid, t, 1e6).min(-1).values
        exists = tout < 1e6
        tout = torch.where(exists, tout, 0.0)
        t, valid, exits = crossings(origin, direction, inner)
        valid = valid & (t >= 0) & (t <= tout[..., None] + 1.5 * k)
        if fan:
            ts, idx = torch.where(valid, t, 1e6).sort(dim=-1, stable=True)
            ex = exits.gather(-1, idx)
            is_exit = (ts[..., 0] < 1e6) & ex[..., 0]
            tb, vb, _ = crossings(origin, -direction, inner)
            tb = torch.where(vb & (tb >= 0) & (tb <= 2.5 * k), tb, 2.5 * k).min(-1).values
            tin = torch.where(is_exit, ts[..., 0], -tb)
            # Rust greedily consumes entry/exit pairs, including non-alternating self crossings.
            skip = is_exit
            tissue = torch.zeros_like(tin)
            for i in range(ts.shape[-1]):
                entry = (~skip) & (~ex[..., i]) & (ts[..., i] < tout)
                end = ts[..., i + 1] if i + 1 < ts.shape[-1] else tout + 1.5 * k
                tissue = tissue + torch.where(entry, torch.minimum(end, tout) - ts[..., i], 0.0)
                skip = entry
            tin = tin + tissue
        else:
            tin = torch.where(valid & (t < tout[..., None] + 1.5 * k), t, -1e6).max(-1).values
            tin = torch.where(tin > -1e6, tin, (tout - 3.5 * k).clamp_min(0))
        pin = origin + direction * torch.minimum(tin, tout)[..., None]
        pout = origin + direction * tout[..., None]
        return pin, pout, (tout - tin).clamp_min(0), exists

    def _compute(self, a: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        k, batch = self.k, a.shape[0]
        outer, inner, roof, outline, arc, tip_pts, tip, ul, ll, lab, ap, spread = self._contours(a)

        def const(x: list[float]) -> Tensor:
            return torch.stack([torch.full_like(a[:, 0], value) for value in x], -1)

        y = mix(
            (a[:, 12] * k + 7 * (1 - self.pharynx) * k + 2.4 * k * self.pharynx + 0.12 * k)[:, None],
            const([(7 - 0.05 * self.pharynx) * k]),
            torch.arange(12, device=a.device, dtype=a.dtype) / 12,
        )
        origin = torch.stack((torch.full_like(y, 4.6 * k), y), -1)
        directions = const([-1.0, 0.0])[:, None, :].expand_as(origin)
        ph = self._measure(origin, directions, outer, inner, False)
        u = torch.linspace(0, 1, 31, device=a.device, dtype=a.dtype)
        ang = (180 - 144 * u) * math.pi / 180
        w = ((u - 0.45) / 0.55).clamp_min(0)
        c1 = self._template(const([2.6, 6.95]))[:, None, :]
        c2 = self._template(const([5.0, 6.85]))[:, None, :]
        origin = mix(c1, c2, (w * w * (3 - 2 * w))[None, :, None])
        directions = torch.stack((ang.cos(), ang.sin()), -1).expand_as(origin)
        fan = self._measure(origin, directions, outer, inner, True)
        x0 = self.palate[-1, 0] + 0.03 * k
        x1 = torch.maximum(x0 + 0.3 * k, torch.minimum(ul[:, 0], ll[:, 0]) - 0.04 * k)
        x = x0 + (x1 - x0)[:, None] * torch.linspace(0, 1, 6, device=a.device, dtype=a.dtype)
        origin = torch.stack((x, torch.full_like(x, 4 * k)), -1)
        front = self._measure(origin, const([0.0, 1.0])[:, None].expand_as(origin), outer, inner, False)
        fi, fo, fd, fv = front
        face = torch.minimum(ll[:, 1:2] - (x - ll[:, 0:1]) * 4 / 3, fo[..., 1])
        opening = smooth(0.7, 1.0, lab[:, None]) * smooth(0.0, 0.1 * k, x - ll[:, 0:1])
        yin = mix(face, fo[..., 1] - 3.5 * k, opening)
        fi = torch.where((x > ll[:, 0:1])[..., None], torch.stack((x, yin), -1), fi)
        fd = torch.where(x > ll[:, 0:1], fo[..., 1] - yin, fd)
        pin = torch.cat((ph[0], fan[0], fi), 1)
        pout = torch.cat((ph[1], fan[1], fo), 1)
        d = torch.cat((ph[2], fan[2], fd), 1)
        valid = torch.cat((ph[3], fan[3], fv), 1)
        index = torch.arange(49, device=a.device).expand(batch, -1)
        # Compact omitted rays with a fixed-size stable permutation. Invalid trailing samples
        # are masked in every reduction, including resampling and centreline accumulation.
        order = torch.where(valid, index, index + 49).argsort(dim=-1, stable=True)
        pin, pout = gather(pin, order), gather(pout, order)
        d, valid = d.gather(-1, order), valid.gather(-1, order)
        original = index.gather(-1, order)
        region = torch.where(original < 19, 1, torch.where(original < 43, 2, 3))
        split = torch.where(valid & (original >= 19), index, 49).min(-1).values
        mid = (pin + pout) * 0.5  # Rust keeps ray midpoint even when contact changes d.
        tip_c = project(roof, tip_pts)[2].min(-1).values
        body_c = project(roof, arc)[2].min(-1).values
        half = 0.1 * k * (2 - smooth(-0.06 * k, 0.0, torch.minimum(tip_c, body_c)))
        s, pr_dist, _ = project(roof, pout)
        on_roof = 1 - smooth(0.15 * k, 0.25 * k, pr_dist)
        dc = signed_gap(outline, pout)
        search = dc - half[:, None] - pr_dist <= 0.6 * k
        roof_len = cumulative(roof)[:, -1:]
        for fr in (-1.0, -0.5, 0.5, 1.0):
            q = point_along(roof, torch.minimum((s + fr * half[:, None]).clamp_min(0), roof_len))
            dc = torch.where(search, torch.minimum(dc, signed_gap(outline, q)), dc)
        w = (1 - smooth(0.15 * k, 0.6 * k, dc)) * on_roof
        cap = (1 - smooth(0.45 * k, 0.6 * k, dc)) * on_roof
        dp = self._positive(dc)
        dm = mix(d, torch.minimum(d, dp), cap)
        d = mix(dm, dp, w)
        edge_x = torch.minimum(ul[:, 0:1], ll[:, 0:1])
        lab_open = smooth(0.0, 0.5, lab[:, None])
        ap_eff = ap[:, None] + 0.3 * k * lab_open / (1 - lab_open).clamp_min(1e-12)
        lip_gap = self._positive(ap_eff + (edge_x - mid[..., 0]).clamp_min(0).square() / (1.2 * k))
        d = torch.where(
            (region == 3) & (mid[..., 0] >= self.palate[-1, 0]) & (lab_open < 1), torch.minimum(d, lip_gap), d
        )
        dental = 2 * d.pow(1.4)
        ahead = smooth(0.0, 0.4 * k, mid[..., 0] - tip[:, 0:1])
        oral = (2.6 - ahead) * d.pow(1.4)
        lip_u = ((mid[..., 0] - self.palate[-1, 0]) / (0.45 * k)).clamp(0, 1)
        lip_w = (1.1 + (self.lip_width - 1.1) * spread) * k
        front_area = mix(dental, math.pi / 4 * d * lip_w[:, None], lip_u)
        area = torch.where(region == 1, dental, torch.where(region == 2, oral, front_area))
        # Sequential, not parallel: Rust updates each left neighbour before the next clamp.
        filtered = [area[:, 0]]
        for i in range(1, 48):
            lim = 1.35 * torch.maximum(filtered[-1], area[:, i + 1])
            filtered.append(torch.where(valid[:, i + 1], torch.minimum(area[:, i], lim), area[:, i]))
        area = torch.stack(filtered + [area[:, -1]], -1)
        lateral = (region >= 2) & (length(mid - tip[:, None]) < 1.4 * k) & (a[:, 15:16] > 0)
        area = torch.where(lateral, torch.maximum(area, a[:, 15:16] * 0.35 * k * k), area)
        distances = length(mid[:, 1:] - mid[:, :-1])
        distances = torch.where(valid[:, 1:], distances, 0.0)
        s = torch.cat((torch.zeros_like(distances[:, :1]), distances.cumsum(-1)), -1) + 2.4 * k * self.pharynx
        # Padded points have huge arc positions and nonclosed areas, so no section contains them.
        s = torch.where(valid, s, 1e6)
        area = torch.where(valid, area, 1.0)
        last = valid.sum(-1) - 1
        # Return final real lip_s in the last slot as well; resample masks padding separately.
        self_last = s.gather(-1, last[:, None])
        teeth_index = torch.where(valid & (original >= 43), index, 49).min(-1).values
        teeth_s = torch.where(teeth_index < 49, s.gather(-1, teeth_index.clamp_max(48)[:, None])[:, 0], self_last[:, 0])
        return torch.cat((s, self_last), -1), torch.cat((area, torch.ones_like(self_last)), -1), split, teeth_s

    def _resample(
        self, a: Tensor, grid_s: Tensor, area: Tensor, start: Tensor, end: Tensor, n: int
    ) -> tuple[Tensor, Tensor]:
        # The final slot carries lip_s, not a grid point.
        lip_s, grid_s, area = grid_s[:, -1:], grid_s[:, :-1], area[:, :-1]
        valid = grid_s < 1e6
        step = (end - start) / n
        pos = torch.arange(n, device=a.device, dtype=a.dtype)
        offsets = (torch.arange(8, device=a.device, dtype=a.dtype) + 0.5) / 8
        s = start[:, None, None] + step[:, None, None] * (pos[:, None] + offsets)
        query = s.flatten(1)
        idx = ((grid_s[:, None, :] <= query[..., None]) & valid[:, None, :]).sum(-1) - 1
        count = valid.sum(-1)
        idx = torch.minimum(idx.clamp_min(0), (count - 2)[:, None])
        s0, s1 = grid_s.gather(-1, idx), grid_s.gather(-1, idx + 1)
        v0, v1 = area.gather(-1, idx), area.gather(-1, idx + 1)
        interp = mix(v0, v1, (query - s0) / (s1 - s0).clamp_min(1e-9))
        interp = torch.where(query <= grid_s[:, :1], area[:, :1], interp)
        last_area = area.gather(-1, (count - 1)[:, None])
        interp = torch.where(query >= lip_s, last_area, interp)
        epi_len = 2.4 * self.k * self.pharynx
        u = query / epi_len
        vent = 0.95 * (1 - 0.75 * a[:, 14:15].clamp(0, 1))
        top = mix(1.9, 0.35, a[:, 13:14].clamp(0, 1))

        def epi(x: Tensor | float) -> Tensor:
            x = x if isinstance(x, Tensor) else a.new_full((), x)
            return torch.where(x < 0.3, 1.3, torch.where(x < 0.5, vent, mix(1.2, top, (x - 0.5) / 0.5)))

        ep = epi(u)
        for u0 in (0.3, 0.5):
            ep = torch.where((u - u0).abs() < 0.03, mix(epi(u0 - 0.03), epi(u0 + 0.03), (u - (u0 - 0.03)) / 0.06), ep)
        ep = ep * self.k * self.k
        ep = torch.where(query > 0.9 * epi_len, mix(ep, area[:, :1], (query - 0.9 * epi_len) / (0.1 * epi_len)), ep)
        samples = torch.where(query <= epi_len, ep, interp).reshape(-1, n, 8)
        a0 = start[:, None] + step[:, None] * pos
        a1 = start[:, None] + step[:, None] * (pos + 1) + (pos == n - 1) * 1e-6
        inside = valid[:, None, :] & (grid_s[:, None, :] >= a0[..., None]) & (grid_s[:, None, :] <= a1[..., None])
        sealed = (samples <= 1e-5).any(-1) | (inside & (area[:, None, :] <= 1e-5)).any(-1)
        # Rust accumulates these eight reciprocals left-to-right. A parallel
        # reduction can move a harmonic mean one ULP below an otherwise equal
        # lateral-channel minimum, changing the discrete frication location.
        reciprocal = 1 / samples.clamp_min(1e-12)
        inv = torch.zeros_like(reciprocal[..., 0])
        for sample in range(8):
            inv = inv + reciprocal[..., sample]
        harmonic = 8 / inv
        minimum = torch.minimum(samples.min(-1).values, torch.where(inside, area[:, None], 1e6).min(-1).values)
        return torch.where(sealed, 0.0, harmonic), torch.where(sealed, 0.0, torch.minimum(minimum, harmonic))
