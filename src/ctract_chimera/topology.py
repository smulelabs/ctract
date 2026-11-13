"""Static topology of CHIMERA's human airway (crates/chimera/src/voice/human_airway.rs).

The network is rebuilt here from the same constants, in the same node/branch order, so the
element arrays line up index for index with `AirwayNetwork::linear_elements()`. Only the
pharynx and oral tubes (areas, lengths) and the velum port area change at control rate; every
other element is fixed for a body.

Verified against the Rust dump in tests/test_vs_rust.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .anatomy import Anatomy

# crates/chimera/src/physics.rs (CGS)
RHO = 1.14e-3
C_SOUND = 35000.0
MU = 1.86e-4
RHO_C2 = RHO * C_SOUND * C_SOUND
WALL_SOFT = (1.5, 1600.0, 3.0e5)
WALL_TRACHEA = (0.6, 1000.0, 1.0e6)
WALL_NASAL = (1.0, 3000.0, 1.0e6)

# AirwayNetwork defaults
CLOSED_AREA = 2e-4
MIN_NODE_AREA = 0.02
VISCOUS_OMEGA = 2.0 * math.pi * 700.0
VISC = math.sqrt(RHO * MU * VISCOUS_OMEGA / 2.0)

SERIES, PORT = 0, 1


def js_round(x: float) -> int:
    """JavaScript Math.round (half up), which the Rust port reproduces."""
    return int(math.floor(x + 0.5))


@dataclass(frozen=True)
class SectionCounts:
    """One discrete CHIMERA graph, used as a component of a continuous atlas."""

    pharynx: int
    oral: int
    trachea: int
    nasal: int


@dataclass
class Tube:
    first: int
    n: int

    def node(self, i: int) -> int:
        return self.first + i

    @property
    def last(self) -> int:
        return self.first + self.n - 1


@dataclass
class Topology:
    fs: float
    k: float
    n_nodes: int = 0
    # per node (fixed parts)
    area0: list[float] = field(default_factory=list)
    dx0: list[float] = field(default_factory=list)
    has_wall: list[bool] = field(default_factory=list)
    wall_m: list[float] = field(default_factory=list)
    wall_r: list[float] = field(default_factory=list)
    wall_k: list[float] = field(default_factory=list)
    wall_perim: list[float] = field(default_factory=list)
    wall_rad: list[float] = field(default_factory=list)
    visc_f: list[float] = field(default_factory=list)
    fixed: list[bool] = field(default_factory=list)
    # per branch
    a: list[int] = field(default_factory=list)
    b: list[int] = field(default_factory=list)
    kind: list[int] = field(default_factory=list)
    port_area0: list[float] = field(default_factory=list)
    port_len: list[float] = field(default_factory=list)
    port_slit: list[float] = field(default_factory=list)
    port_half: list[bool] = field(default_factory=list)
    port_extra_r: list[float] = field(default_factory=list)
    rad_nodes: list[int] = field(default_factory=list)

    # --- builder (mirrors AirwayNetwork) ---
    def add_tube(
        self,
        n: int,
        length: float,
        area: float | list[float] | tuple[float, ...],
        wall: tuple[float, float, float] | None = None,
        perimeter: float = 1.0,
        viscous: float = 1.0,
        radiating: float = 0.0,
    ) -> Tube:
        """Port AirwayNetwork::add_tube, preserving Rust node/branch order."""
        t = Tube(self.n_nodes, n)
        dx = length / n
        for i in range(n):
            self.area0.append(area[i] if isinstance(area, list | tuple) else area)
            self.dx0.append(dx)
            self.has_wall.append(wall is not None)
            m, r, kk = wall if wall is not None else (0.0, 0.0, 0.0)
            self.wall_m.append(m)
            self.wall_r.append(r)
            self.wall_k.append(kk)
            self.wall_perim.append(perimeter * 2.0 * math.sqrt(math.pi))
            self.wall_rad.append(radiating)
            self.visc_f.append(viscous)
            self.fixed.append(False)
        self.n_nodes += n
        for i in range(n - 1):
            self._branch(t.node(i), t.node(i + 1), SERIES)
        return t

    def _branch(
        self,
        a: int,
        b: int,
        kind: int,
        area: float = 0.0,
        length: float = 0.0,
        slit: float = 0.0,
        half: bool = True,
        extra_r: float = 0.0,
    ) -> int:
        self.a.append(a)
        self.b.append(b)
        self.kind.append(kind)
        self.port_area0.append(area)
        self.port_len.append(length)
        self.port_slit.append(slit)
        self.port_half.append(half)
        self.port_extra_r.append(extra_r)
        return len(self.a) - 1

    def connect(self, a: int, b: int) -> int:
        return self._branch(a, b, SERIES)

    def add_port(
        self, a: int, b: int, area: float, length: float, slit: float = 0.0, half: bool = True, extra_r: float = 0.0
    ) -> int:
        return self._branch(a, b, PORT, area, length, slit, half, extra_r)


@dataclass
class HumanTopology:
    topo: Topology
    trachea: Tube
    pharynx: Tube
    oral: Tube
    nasal: Tube
    fossa: Tube
    lungs: int
    glottis_port: int
    velum_port: int
    n_ph: int
    n_or: int

    @property
    def inject(self) -> int:
        return self.pharynx.first


def build_human(
    fs: float = 96000.0,
    anatomy: Anatomy = Anatomy(),
    counts: SectionCounts | None = None,
) -> HumanTopology:
    """Construct HumanAirway::new's static graph for explicit anatomy.

    The internal rate sets node counts through CHIMERA's CFL discretisation.
    No audio rate conversion or pitch-dependent configuration is performed.
    """
    if not math.isfinite(fs) or fs <= 0:
        raise ValueError("internal sample rate must be finite and positive")
    k, pharynx_f = anatomy.scale, anatomy.pharynx
    t = Topology(fs=fs, k=k)
    dx_min = (C_SOUND / fs) / 0.93
    ph_min_len = 8.7 * k * (0.55 + 0.45 * pharynx_f) - 1.6 * k
    or_min_len = 8.3 * k - 1.0 * k
    n_ph = int(max(8.0, math.floor(ph_min_len / dx_min)))
    n_or = int(max(8.0, math.floor(or_min_len / dx_min)))
    if counts is not None:
        n_ph, n_or = counts.pharynx, counts.oral
    tr_len = 12.0 * k
    trachea = t.add_tube(
        counts.trachea if counts else int(max(6.0, math.floor(tr_len / max(dx_min, 0.9 * k)))),
        tr_len,
        2.5 * k * k,
        WALL_TRACHEA,
    )
    pharynx = t.add_tube(n_ph, 8.8 * k, 3.0 * k * k, WALL_SOFT, perimeter=1.25, radiating=0.25)
    oral = t.add_tube(n_or, 8.3 * k, 3.0 * k * k, WALL_SOFT, perimeter=1.5, radiating=0.25)
    na_len = 11.4 * k
    n_na = int(max(8.0, math.floor(na_len / max(dx_min, 0.5 * k))))
    if counts is not None:
        n_na = counts.nasal
    na_areas = []
    for i in range(n_na):
        u = (i + 0.5) / n_na
        a = 2.6 if u < 0.18 else (2.0 - 0.6 * (u - 0.18) if u < 0.75 else (1.4 if u < 0.9 else 1.0))
        na_areas.append(a * k * k)
    nasal = t.add_tube(n_na, na_len, na_areas, WALL_NASAL, perimeter=3.2, viscous=2.5)
    fossa = t.add_tube(3, 2.0 * k, 0.9 * k * k, WALL_SOFT, perimeter=1.3)
    lungs = t.add_tube(1, 1.0, 100.0).first
    t.fixed[lungs] = True
    tr_a = 2.5 * k * k
    t.add_port(lungs, trachea.first, tr_a, 1.0, extra_r=(0.5 * RHO * C_SOUND) / tr_a)
    glottis_port = t.add_port(trachea.last, pharynx.first, 0.0, 0.3 * k, slit=1.4 * k)
    t.connect(pharynx.last, oral.first)
    velum_port = t.add_port(pharynx.last, nasal.first, 0.0, 1.0 * k)
    fossa_node = pharynx.node(int(min(n_ph - 1, js_round((2.6 / 8.8) * n_ph))))
    t.connect(fossa_node, fossa.first)
    vol = 14.0 * k**3
    s_len = vol ** (1.0 / 3.0)
    sinus = t.add_tube(1, s_len, vol / s_len, None, perimeter=1.2).first
    t.add_port(nasal.node(js_round(n_na * 0.45)), sinus, 0.12 * k * k, 0.5 * k, half=False)
    t.rad_nodes = [oral.last, nasal.last]
    return HumanTopology(t, trachea, pharynx, oral, nasal, fossa, lungs, glottis_port, velum_port, n_ph, n_or)


@dataclass(frozen=True)
class TreePlan:
    """Static BFS tree from the ideal-flow injection, independent of geometry."""

    order: tuple[int, ...]
    parent: tuple[int, ...]
    parent_branch: tuple[int, ...]


def plan_tree(h: HumanTopology) -> TreePlan:
    """Plan frequency.rs's leaf elimination with the glottis cut.

    All potentially open edges are present, including the nasal port. Data
    closures are handled by continuous/hard conductance in the solve itself.
    """
    t = h.topo
    adj: list[list[int]] = [[] for _ in range(t.n_nodes)]
    for j, (a, b) in enumerate(zip(t.a, t.b, strict=True)):
        if j != h.glottis_port:
            adj[a].append(j)
            adj[b].append(j)
    root = h.inject
    order, parent, pbranch = [root], {root: -1}, {}
    q = 0
    while q < len(order):
        m = order[q]
        for j in adj[m]:
            k = t.b[j] if t.a[j] == m else t.a[j]
            if k in parent:
                continue
            parent[k] = m
            pbranch[k] = j
            order.append(k)
        q += 1
    return TreePlan(tuple(order), tuple(max(parent[n], 0) for n in order), tuple(pbranch.get(n, 0) for n in order))
