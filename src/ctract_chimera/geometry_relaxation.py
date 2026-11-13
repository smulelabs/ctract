"""Continuous polyline topology for the HumanVocalTract port.

The hard geometry is kept separately as an oracle. Every value-dependent choice
here is a probability or a weighted sum: no selected indices, sort, or custom
backward. Static tensor indices refer only to polyline vertices or a sorting
network. Widths are in cm unless explicitly dimensionless.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .geometry_primitives import mix
from .relaxation import positive


class Relaxation:
    """Smooth algebraic tails, all converging pointwise off decision ties.

    A fixed positive temperature has bounded local derivatives. Bounds cannot
    remain uniform as temperature→0 at discontinuities of the Rust map. See the
    derivation document for that distinction and the treatment of inclusive ties.
    """

    def __init__(self, tau: Tensor) -> None:
        self.tau = tau
        self.width = tau * 0.002

    def gate(self, x: Tensor, width: Tensor | None = None) -> Tensor:
        """Symmetric soft x>0, with algebraic tails (and exact midpoint at 0)."""
        w = self.width if width is None else width
        return torch.sigmoid(2 * torch.asinh(x / w))

    def positive(self, x: Tensor) -> Tensor:
        return positive(x, self.width)

    def clamp(self, x: Tensor, lo: float | Tensor, hi: float | Tensor) -> Tensor:
        """Smooth clamp; integral of two algebraic-tailed step functions."""
        return lo + self.positive(x - lo) - self.positive(x - hi)

    def control_clamp(self, x: Tensor, lo: Tensor, hi: Tensor) -> Tensor:
        """Endpoint-preserving smooth control projection.

        Ordinary soft clipping maps a true zero lateral-channel control to a
        positive opening of width/2. Preserve the endpoint values instead;
        outside the control domain this extension can overshoot by O(width),
        and still tends to the Rust clamp. Interval-interpolation clamps keep
        their bounded version because their convex weights must stay in [0,1].
        """
        edge = self.positive(torch.zeros_like(x))
        lower = lo + edge - self.positive(lo - hi)
        upper = lo + self.positive(hi - lo) - edge
        return lo + (hi - lo) * (self.clamp(x, lo, hi) - lower) / (upper - lower)

    def maximum(self, x: Tensor, y: Tensor | float) -> Tensor:
        return y + self.positive(x - y)

    def minimum(self, x: Tensor, y: Tensor | float) -> Tensor:
        # Unlike x-positive(x-y), the convex selector cannot create negative
        # distances from two nonnegative arguments. Both have the same hard limit.
        w = self.gate(y - x)
        return w * x + (1 - w) * y

    def norm(self, x: Tensor) -> Tensor:
        return (x.square().sum(-1) + self.width.square() * 1e-6).sqrt()

    def ramp(self, lo: float | Tensor, hi: float | Tensor, x: Tensor) -> Tensor:
        u = self.clamp((x - lo) / (hi - lo), 0.0, 1.0)
        return u.square() * (3 - 2 * u)

    def select_min(self, values: Tensor, dim: int = -1) -> Tensor:
        """Continuous minimum as a convex average (no negative undershoot)."""
        return (torch.softmax(-values / self.width, dim=dim) * values).sum(dim)

    def inverse(self, x: Tensor) -> Tensor:
        """Tikhonov inverse: parallel/degenerate segments remain finite.

        The narrower 1e-3 spatial scale regularises ray denominators without
        broadening visible contacts. For every nonzero x it tends to 1/x.
        """
        return x / (x.square() + (self.width * 1e-3).square())

    def cumulative(self, points: Tensor) -> Tensor:
        d = self.norm(points[..., 1:, :] - points[..., :-1, :])
        return torch.cat((torch.zeros_like(d[..., :1]), d.cumsum(-1)), -1)

    def interval_weights(self, knots: Tensor, query: Tensor) -> Tensor:
        """Partition of unity over adjacent intervals, including end holds.

        The first/last interval owns the infinite end outside the polyline.
        Difference of monotone CDFs keeps weights nonnegative and telescopes
        to one. This replaces integer interpolation-bin selection.
        """
        cdf = self.gate(query[..., None] - knots[..., None, 1:-1])
        return torch.cat((1 - cdf[..., :1], cdf[..., :-1] - cdf[..., 1:], cdf[..., -1:]), -1)

    def point_along(self, wall: Tensor, query: Tensor) -> Tensor:
        knots = self.cumulative(wall)
        delta = knots[..., 1:] - knots[..., :-1]
        u = self.clamp((query[..., None] - knots[..., None, :-1]) / self.maximum(delta, 1e-9)[..., None, :], 0.0, 1.0)
        points = mix(wall[..., None, :-1, :], wall[..., None, 1:, :], u[..., None])
        return (self.interval_weights(knots, query)[..., None] * points).sum(-2)

    def crossings(self, origin: Tensor, direction: Tensor, wall: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Relax geom.rs::ray_crossings, including endpoint-side convention.

        Side XOR is its Bernoulli extension. Parallel-ray division uses a
        vanishing Tikhonov regulariser; the extra visibility factor suppresses
        degenerate segments. The exact hard threshold remains in the limit.
        """
        a, b = wall[..., None, :-1, :], wall[..., None, 1:, :]
        o, d = origin[..., :, None, :], direction[..., :, None, :]
        edge = b - a

        def cross(x: Tensor, y: Tensor) -> Tensor:
            return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]

        den = cross(d, edge)
        # Log visibility is essential: a finite positive probability alone can
        # underflow before the weighted intersection selection sees it. Side
        # comparisons narrow as tau²; selection narrows only as tau. Therefore
        # invisible lower-coordinate candidates CANNOT dominate the hard limit.
        width = self.width * self.tau
        side_a = cross(d, a - o) / width - self.tau.log()
        side_b = cross(d, b - o) / width - self.tau.log()
        log_sigmoid = torch.nn.functional.logsigmoid
        log_visibility = torch.logaddexp(
            log_sigmoid(side_a) + log_sigmoid(-side_b), log_sigmoid(-side_a) + log_sigmoid(side_b)
        )
        absolute = torch.hypot(den, torch.full_like(den, 1e-30))
        log_visibility = log_visibility + log_sigmoid((absolute - 1e-15) / (self.tau.square() * 1e-17))
        t = cross(a - o, edge) * self.inverse(den)
        return t, log_visibility, self.gate(-den)

    def project(self, wall: Tensor, points: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Smooth closest-segment projection and signed oriented distance."""
        a = wall[..., None, :-1, :]
        edge = (wall[..., 1:, :] - wall[..., :-1, :])[..., None, :, :]
        delta = points[..., :, None, :] - a
        l2 = edge.square().sum(-1)
        u = self.clamp((delta * edge).sum(-1) / self.maximum(l2, 1e-12), 0.0, 1.0)
        dist = self.norm(delta - u[..., None] * edge)
        visibility = self.gate(l2 - 1e-12, self.width.square() * 1e-3)
        # A finite absent-segment cost, large relative to every cranial gap.
        cost = dist + 100 * (1 - visibility)
        weights = torch.softmax(-cost / self.width, -1)
        arc = self.cumulative(wall)[..., None, :-1] + u * self.norm(edge)
        cross = edge[..., 0] * delta[..., 1] - edge[..., 1] * delta[..., 0]
        sign = 1 - 2 * self.gate(cross)
        return (weights * arc).sum(-1), (weights * dist).sum(-1), (weights * sign * dist).sum(-1)

    def signed_gap(self, wall: Tensor, points: Tensor) -> Tensor:
        """Continuous even/odd containment, preserving self-intersection parity.

        E[(-1)^number_of_crossings] = product(1-2*p_crossing). Unlike replacing
        parity with a winding sign, this has the SAME hard limit on self-crossing
        outlines. Distance is a smooth closest-segment distance.
        """
        a, b = wall[..., None, :-1, :], wall[..., None, 1:, :]
        p = points[..., :, None, :]
        edge = b - a
        l2 = edge.square().sum(-1)
        u = self.clamp(((p - a) * edge).sum(-1) / self.maximum(l2, 1e-12), 0.0, 1.0)
        distance = self.select_min(self.norm(p - a - u[..., None] * edge))
        first, second = self.gate(a[..., 1] - p[..., 1]), self.gate(b[..., 1] - p[..., 1])
        crossing = first * (1 - second) + second * (1 - first)
        intercept = a[..., 0] + (p[..., 1] - a[..., 1]) * edge[..., 0] * self.inverse(edge[..., 1])
        crossing = crossing * self.gate(intercept - p[..., 0])
        return distance * (1 - 2 * crossing).prod(-1)

    def sort(self, values: Tensor, payload: Tensor) -> tuple[Tensor, Tensor]:
        """Fixed bitonic compare/exchange network, with smooth convex exchanges.

        The comparator's auxiliary payload follows the SAME probability as its
        coordinate. Static network wiring is independent of the inputs. Padding
        is greater than the bounded absent-intersection cost used by the caller.
        """
        count = values.shape[-1]
        n = 1 << (count - 1).bit_length()
        values = torch.nn.functional.pad(values, (0, n - count), value=1000.0)
        payload = torch.nn.functional.pad(payload, (0, n - count))
        index = torch.arange(n, device=values.device)
        size = 2
        while size <= n:
            stride = size // 2
            while stride:
                other = index ^ stride
                v, p = values.index_select(-1, other), payload.index_select(-1, other)
                choose_low = ((index & size) == 0) == ((index & stride) == 0)
                low = self.gate(v - values)
                weight = torch.where(choose_low, low, 1 - low)
                values, payload = mix(v, values, weight), mix(p, payload, weight)
                stride //= 2
            size *= 2
        return values[..., :count], payload[..., :count]


def adjacency(visibility: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Bernoulli consecutive-visible-pair, first, and last probabilities.

    Pair[i,j] = v_i*v_j*product(1-v_k, i<k<j). This is the continuous
    extension of missing-ray compaction WITHOUT sorting or integer indices.
    Only 49 fixed rays are involved. Explicit products avoid 0/0 at hard masks.
    """
    n = visibility.shape[-1]
    pairs = []
    for i in range(n):
        tail = visibility[..., i + 1 :]
        exclusive = torch.cat((torch.ones_like(visibility[..., :1]), (1 - tail).cumprod(-1)), -1)[..., :-1]
        row = visibility[..., i : i + 1] * tail * exclusive
        pairs.append(torch.nn.functional.pad(row, (i + 1, 0)))
    pair = torch.stack(pairs, -2)
    ones = torch.ones_like(visibility[..., :1])
    first = visibility * torch.cat((ones, (1 - visibility[..., :-1]).cumprod(-1)), -1)
    last = visibility * torch.cat(((1 - visibility[..., 1:]).flip(-1).cumprod(-1).flip(-1), ones), -1)
    return pair, first, last
