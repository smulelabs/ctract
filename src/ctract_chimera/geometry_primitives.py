"""Polyline primitives for CHIMERA anatomy/human.rs.

bezier/catmull port anatomy/geom.rs::{bezier2,bezier3,catmull_rom};
crossings ports ray_crossings; project/point_along port Polyline projection
and arc sampling. signed_gap ports human.rs's tongue-outline containment.
These are exact piecewise primitives, not smooth topology relaxations.
"""

from __future__ import annotations

import torch
from torch import Tensor


def length(x: Tensor) -> Tensor:
    # vector_norm has a finite zero subgradient, unlike sqrt(sum(x*x)).
    return torch.linalg.vector_norm(x, dim=-1)


def mix(a: Tensor, b: Tensor, t: Tensor | float) -> Tensor:
    return a + (b - a) * t


def smooth(lo: float, hi: float, x: Tensor) -> Tensor:
    u = ((x - lo) / (hi - lo)).clamp(0, 1)
    return u * u * (3 - 2 * u)


def bezier(points: list[Tensor], count: int) -> Tensor:
    t = torch.linspace(0, 1, count, device=points[0].device, dtype=points[0].dtype)[..., None]
    work = [p[..., None, :] for p in points]
    while len(work) > 1:
        work = [mix(a, b, t) for a, b in zip(work[:-1], work[1:], strict=False)]
    return work[0]


def catmull(points: Tensor, per: int) -> Tensor:
    out = []
    for i in range(points.shape[-2] - 1):
        p0, p1, p2, p3 = [points[..., j, :] for j in (max(0, i - 1), i, i + 1, min(i + 2, points.shape[-2] - 1))]
        for k in range(per):
            t = k / per
            out.append(
                0.5
                * (
                    2 * p1
                    + (-p0 + p2) * t
                    + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                    + (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t
                )
            )
    return torch.stack(out + [points[..., -1, :]], -2)


def gather(points: Tensor, index: Tensor) -> Tensor:
    return points.gather(-2, index[..., None].expand(*index.shape, points.shape[-1]))


def cumulative(points: Tensor) -> Tensor:
    d = length(points[..., 1:, :] - points[..., :-1, :])
    return torch.cat((torch.zeros_like(d[..., :1]), d.cumsum(-1)), -1)


def crossings(o: Tensor, direction: Tensor, wall: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """All ray/segment intersections, keeping the Rust positive-side vertex convention."""
    a, b = wall[..., None, :-1, :], wall[..., None, 1:, :]
    o, direction = o[..., :, None, :], direction[..., :, None, :]
    e = b - a

    def cross(x, y):
        return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]

    sa, sb = cross(direction, a - o) >= 0, cross(direction, b - o) >= 0
    den = cross(direction, e)
    ok = (sa != sb) & (den.abs() >= 1e-15)
    t = cross(a - o, e) / torch.where(ok, den, torch.ones_like(den))
    return t, ok, den < 0


def project(wall: Tensor, points: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Closest segment projection: arc position, distance, signed distance."""
    a, e = wall[..., None, :-1, :], (wall[..., 1:, :] - wall[..., :-1, :])[..., None, :, :]
    delta = points[..., :, None, :] - a
    l2 = (e * e).sum(-1)
    t = ((delta * e).sum(-1) / l2.clamp_min(1e-12)).clamp(0, 1)
    d = length(delta - t[..., None] * e)
    d = torch.where(l2 >= 1e-12, d, torch.full_like(d, 1e6))
    best, idx = d.min(-1)
    arc = cumulative(wall)[..., None, :-1] + t * length(e)
    s = arc.gather(-1, idx[..., None]).squeeze(-1)
    cross = e[..., 0] * delta[..., 1] - e[..., 1] * delta[..., 0]
    sign = torch.where(cross < 0, 1.0, -1.0).gather(-1, idx[..., None]).squeeze(-1)
    return s, best, sign * best


def point_along(wall: Tensor, s: Tensor) -> Tensor:
    cum = cumulative(wall)
    # Rust uses < rather than <= and ignores all but the final boundary.
    idx = (cum[..., None, 1:-1] < s[..., None]).sum(-1)
    a, b = gather(wall, idx), gather(wall, idx + 1)
    c0, c1 = cum.gather(-1, idx), cum.gather(-1, idx + 1)
    u = torch.where(c1 - c0 > 1e-9, (s - c0) / (c1 - c0).clamp_min(1e-9), 0.0).clamp(0, 1)
    return mix(a, b, u[..., None])


def signed_gap(wall: Tensor, points: Tensor) -> Tensor:
    a, b = wall[..., None, :-1, :], wall[..., None, 1:, :]
    p = points[..., :, None, :]
    e = b - a
    l2 = (e * e).sum(-1)
    t = ((p - a) * e).sum(-1) / l2.clamp_min(1e-12)
    t = torch.where(l2 > 1e-12, t.clamp(0, 1), 0.0)
    d = length(p - a - t[..., None] * e).min(-1).values
    dy = e[..., 1]
    cross = ((a[..., 1] > p[..., 1]) != (b[..., 1] > p[..., 1])) & (
        p[..., 0] < a[..., 0] + (p[..., 1] - a[..., 1]) * e[..., 0] / torch.where(dy != 0, dy, 1.0)
    )
    inside = cross.sum(-1).remainder(2).bool()
    return torch.where(inside, -d, d)
