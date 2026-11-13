# Design and derivation

This document derives the hard reference and positive-temperature differentiable
extensions, including their limits. Rust references are files in CHIMERA's
[`crates/chimera/src/`](https://github.com/jamesondh/chimera-vocal-synth/tree/eed4a5ff2695/crates/chimera/src) at commit `eed4a5ff2695`.

## 1. Anatomy and articulators

`anatomy/human.rs::HumanBody` defines `scale`, `pharynx`, `tongue_radius`,
`lip_width`. They are independent positive Python configuration values here,
not preset labels and not inferred from f0. `HumanAirway::new` determines tube
counts from the shortest expected lengths and the CFL length
`dx_min = (35000/internal_fs)/0.93` cm. These counts use floor/max and therefore
change discontinuously with anatomy. `AnatomicalRenderer` and `AnatomicalSynth`
represent these changing counts using a static finite atlas.
`AnatomyDomain` specifies scale/pharynx bounds, and monotone endpoint counts
bound an exhaustive Cartesian set of possible graphs. It is not a set of body
presets. For a pre-floor count coordinate q, CDF differences at integers form a
partition of unity. Each CDF is an inclusive algebraic-tailed step, so an exact
integer selects Rust's floor(integer) bin in the cold limit. Joint graph weights
are products of the four partitions; every component is evaluated, with no
fitted-value-dependent selection. The output is a mixture of graph audio, not
interpolation between differently sized parameter arrays.

Within each component, the four tensor anatomy values reconstruct all contour
landmarks. Overall scale also updates all fixed areas by k² and lengths by k,
port slit widths by k and fixed port resistances by k^-2. The pressure-node
lung volume and its 1 cm neck length remain unscaled, exactly as in Rust.
Pharynx/oral lengths come from geometry. Material wall constants remain the same.
A custom-anatomy Rust fixture checks these physical scalings independently of
geometry; continuous anatomy and count-boundary audio have fp64 gradchecks.

An unbounded scale domain requires an unbounded number of graph nodes and
cannot be represented by one finite dense computation. The explicit finite
domain is therefore part of the model's discretisation contract, and inputs
outside it are rejected, never silently clipped. Researchers can enlarge the
domain and accept its additional computational cost. Radius and lip width do
not affect node counts, so they need no finite atlas bounds beyond positivity.

`ExactGeometry` takes instantaneous explicit-tip articulators. CHIMERA's live
`HumanVoice` instead includes command tracking, resting-tip replacement, contact
projection and slew limiting. Those are different contracts. The geometry port
corresponds to `HumanAirway::apply(a,0)` after clamping `ARTIC_SPECS` bounds.

## 2. Midsagittal geometry and section areas

`anatomy/human.rs::compute` transforms the canonical head frame, constructs the
posterior wall and palate (Catmull-Rom), velum flap, jaw-carried teeth/floor,
circular tongue body, Bézier root/blade/lips, and lateral channels. Horizontal,
fan and vertical rays measure wall gaps. Oral sections follow the `alpha*d^1.4`
area law; lip sections blend to an elliptical slit `pi*d*width/4`. Contact caps
replace ray gaps near the roof, and the sequential spike filter caps an isolated
area at `1.35*max(neighbours)`. Centreline distances use the original midpoints,
before contact correction, as in Rust.

`HumanVocalTract::resample` uses eight midpoint samples per airway section and
their harmonic mean `8/sum(1/A)`. A sampled area or contained geometry-grid area
at or below `1e-5 cm²` seals the section completely. The epilarynx/ventricular
profile is evaluated analytically ahead of the grid.

Hard mode preserves this discrete algorithm. Floating-point agreement is tested
with tolerances, not bitwise. Fit mode lives in `geometry_smooth.py` and
`geometry_relaxation.py` and is a continuous extension of the same contour laws:

| Rust decision | Continuous forward extension |
| --- | --- |
| Clamps and contact gaps | Algebraic-tailed smooth positive part |
| First wall/root point | First-success Bernoulli distribution; weighted clipping boundary |
| Tongue circle and tangent | Regularised normalization and continuously gated tangent |
| Wrapped tangent angle | Analytic Abel sum of the sawtooth, with positive atan2 denominator |
| Ray crossing visibility | Log-probability of endpoint-side XOR and distance/degeneracy predicates |
| Min/max crossing | Weighted coordinate with visibility penalty; no selected index |
| Fan crossing order | Fixed bitonic comparator network, payloads following the same convex exchanges |
| Greedy tissue pairing | Bernoulli extension of Rust's skip/consume recurrence |
| Missing rays | P(i,j consecutive)=v_i*v_j*product(1-v_k, i<k<j), no compaction |
| Pharynx split/lip end | First/last-success weighted arclength |
| Closest segment | Soft nearest-segment projection, weighted arc and oriented distance |
| Polygon parity | Exact Bernoulli identity E[(-1)^crossings]=product(1-2*p) |
| Roof sampling/interpolation bins | Monotone CDF interval partition of unity |
| Lateral/contact/lip handover | Smooth spatial membership and minimum/maximum |
| Spike filter | Sequential recurrence over conditional previous/next-visible weights |
| Harmonic section resampling | Dense fixed consecutive-ray pairs and continuous interval ownership |
| Point closure membership | Outward sqrt(width) bias, giving inclusive ownership at both endpoints |
| Section sealing | One soft minimum/opening gate with algebraic tails; not a product of closed gates |

Crossing visibility sharpens as tau² while selection sharpens as tau. Log
visibility is retained before selection: an invisible lower-coordinate candidate
cannot win merely because its small probability decays slower than a softmax.
The geometry equations, ray count (49), and eight samples per airway section
remain CHIMERA's. Static tensor indexing and comparator wiring never depend on
fitted parameters. There are no custom backward rules or stopped gradients.

For positive tau the formulas are differentiable through contacts, visibility,
ordering and resampling; zero-distance norms and parallel denominators have
vanishing regularisers. Temperature is validated, not silently floored. fp32
inputs use fp64 internally to preserve the differences of small clamp tails.
Cold-limit tests cover the 96 synthetic oracle poses, and operator gradchecks
include interpolation/ordering ties. The analytic pointwise limit is the Rust
map on nondegenerate decision cells. Inclusive ray-side and section-boundary
ties have vanishing directional biases; coincident/self-intersection topology
needs the explicitly implemented Rust ordering, not a generic winding heuristic.
The test corpus is evidence, not an exhaustive proof for arbitrary degenerate
self-intersecting polylines.

Two universal guarantees one might ask for cannot
hold literally. If smooth f_tau tends to a hard jump of size J, the mean-value
theorem forces some derivative near that jump to grow at least J/(2*epsilon)
as its transition width epsilon tends to zero. Thus no temperature-independent
gradient bound is possible there. Also a physically inactive parameter (e.g. a
noise cutoff at zero gain) has exactly zero audio derivative; faithful models
cannot promise nonzero derivatives for all parameters at all states. Tests
instead require finite, nonzero per-articulator sensitivities on explicit
contact/closure sweeps at tau=.05 (including compressed lips and closed velum),
with a bound of 1e7 in the stated units, and verify the cold forward limit
separately. No arbitrary straight-through slope is added to evade these facts.

## 3. Linear airway elements

`voice/human_airway.rs::new` builds lungs, trachea, glottal port, pharynx, oral,
nasal, piriform fossae and maxillary sinus. The glottal port is cut for the
ideal-flow source. The reachable graph is a tree rooted at pharynx.first.
Topology construction happens once, independently of audio/geometry values.

CGS constants from `physics.rs` are `rho=1.14e-3 g/cm³`, `c=35000 cm/s`,
`mu=1.86e-4 g/(cm*s)`. For a tube branch a→b,

```
L = rho/2 * (dx_a/A_a + dx_b/A_b)
R = visc/2 * sum(perimeter_factor * dx * sqrt(A)/A² * viscous_factor)
C_node = max(A, .02) * dx / (rho*c²)
```

`visc = sqrt(rho*mu*(2*pi*700)/2)`. Ports add neck inertance and optional
half-section terms. Slit resistance is `12*mu*length*slit_width²/A³`;
a duct uses the circular perimeter law. Yielding walls have impedance
`Z_wall = r_wall + j*(omega*m_wall - k_wall/omega)` per wall area.
Radiation is the Flanagan parallel R–L termination, as in `frequency.rs`.

Hard opening is `min(A_a,A_b)>=2e-4 cm²` (or the port area). Soft opening uses
`sigmoid(2*asinh((A-threshold)/w) - log(w/threshold))`, with smooth positive areas,
area floors and end-area minimum. Its algebraic tails avoid exponential
extinction in moderately deep closures. The logarithmic bias gives the
inclusive open tie in the zero-width limit. Here `w` is **cm²**, not the
geometry annealing scalar. Outside threshold ties the gate converges pointwise
to the Rust choice; finite precision still limits how small a tail can be.

## 4. Tree elimination and arbitrary source placement

Let Y_k be each node's compliance + wall + radiation shunt; Z_j a branch's
resistance + j*omega*inertance; O_j its opening factor. The global nodal operator
is symmetric (not Hermitian), with branch conductance O_j/Z_j. If g is the
radiating-admittance vector, reciprocity gives radiated flow of injection J as
`g^T A^-1 J = phi^T J`, where `A phi = g`.

Eliminate each leaf subtree k into parent p:

```
t = O / (O + Z*Y_k)
Y_p += t*Y_k
S_p += t*S_k
```

Solve the root `phi_root=S_root/Y_root`, then back-substitute:

```
phi_k = (Z*S_k + O*phi_parent) / (O + Z*Y_k)
```

A node flow source gives `H_k=phi_k`; a series pressure source in a→b gives
`D_j=(phi_b-phi_a)*O_j/Z_j`. The glottal cut and unreachable subglottal branches
have exactly zero dipole transfer. `transfer` returns the root's H, the same
quantity as `frequency.rs::transfer_function`. `responses` exposes all H and D,
so continuous pressure-source placement is just a weighted sum. No constriction
argmin or nasal on/off decision occurs inside the solve. All nasal nodes are
always visited; a shut port attenuates them algebraically.

## 5. What is and is not the glottal source

`sources/glottis.rs::shape` is a raised-cosine rise followed by a blend of
raised-cosine/cosine closure. The normalized area is zero after the open phase.
This is implemented directly, including the skew clamp. The caller supplies
phase; no hidden oscillator or detached phase accumulator exists here.

The full `ParametricGlottis::next` is ported separately in `glottis.py`,
including all 17 GlottisParams fields and fixed Rust random realisations.
`CoupledSynth.simulate` runs that source against the previous network pressures
and advances the nonlinear momentum, wall and pressure states in Rust order.
The analytic quadratic root and triangular node solve carry autograd through
the coupling. The frozen-block `forward` readout convolves the resulting glottal
flow; it does not pretend area and flow are interchangeable. See
[COUPLING.md](COUPLING.md) for the derivation, exact short-case Rust results,
model boundaries and cycle/transition validation.

## 6. Noise and bursts

The complete noise and burst port is documented in [NOISE.md](NOISE.md).
`Turbulence` uses true sub-section minima/incisor position, continuous pairwise
rank localisation, inclusive jet/teeth rounding partitions and flow-direction
routing. Registration and filter `(src,fwd)` identity preserve Rust's shared
white-stream draw order in the hard limit; all fixed draws are supplied before
simulation. The three gains and six cutoffs carry gradients through the causal
HP/LP/LP states. Reynolds excess uses the exact 1800 threshold and 12000-Re
saturation, with algebraic-tailed relaxations at positive temperature.

Pressure-release bursts emerge from the nonlinear recurrent pressure/momentum
state, and moving-wall `-dV/dt` is squeeze-routed and two-pole filtered in Rust
node order. The convolutional readout additionally propagates explicit noise
pressure and moving-wall node-flow histories. No independent burst envelope
or fixed tooth position is added.
The faithful nonlinear time-domain output remains available through simulate().

## 7. Convolution and fitting

Each fixed-clock source block samples geometry at its midpoint. Its discrete H/D response is
corrected for the internal derivative clock and inverse transformed; a centred
`ir_length` window retains the finite-band negative-time tails. Linear
FFT convolution uses zero padding of at least `block_size+ir_length-1`. Each
block tail is overlap-added, the requested length is selected, then the fixed IR offset is removed. The factorised output-rate difference
implements the internal-rate derivative exactly on the frequency grid. Fixed
clock interpolation indices do not depend on any fitted value.

For stationary geometry this is one LTI convolution up to response-grid aliasing
and IR truncation; block partition invariance is tested. For moving geometry,
wave tails retain excitation-time geometry, an explicit approximation. The
continuous-frequency transfer API differs from `AirwayNetwork::step`'s finite-step
integration; the renderer uses the stepper's exact z-domain linearisation instead.
Audio oracle tests state separate waveform and magnitude bounds, accounting for
finite-band projection and IR truncation rather than claiming exact equality.

`examples/fit_synthetic.py` recovers jaw, lip aperture, larynx height and source
flow gain from committed audio rendered by Rust's actual time stepper. The known
24-tone ideal-flow excitation spans 250–6000 Hz with a smooth finite envelope.
With a Hann-windowed DFT, the loss is squared magnitude error plus 0.2 times
squared complex-spectrum error, normalised by target spectral power. The small
phase term helps resolve correlated articulators. L-BFGS variable-metric descent
with strong-Wolfe line search uses ordinary autograd gradients; the log-gain
coordinate enforces positive flow gain. Starting at [.85, 1.05, .2, exp(.1)], it
recovers [.600685, .700431, .0000409, .999675] for targets [.6, .7, 0, 1], reducing
loss from .458475 to 6.014e-8. This is local multi-parameter recovery, not an
identifiability claim for arbitrary recordings.

## Renderer timing

The renderer uses `discrete_responses` (exact z-domain linearisation of the Rust
stepper) rather than a generic jω/sine warp. It factors the internal-rate
derivative into an output-rate difference and a smooth frequency correction,
retaining exact DC rejection. A centred IR keeps the negative-time tails of
the finite-band projection; its fixed offset is removed after overlap-add.
[STATIONARY.md](STATIONARY.md) derives the 26.04 µs delay an output-rate derivative would add and reports
raw/aligned errors, IR sweeps and the native-rate causal check. The continuous
transfer API serves CHIMERA's frequency-domain oracle.

Deep-contact tests also check log-area sensitivity, not just a tiny nonzero raw
derivative. Redundant positive-part guards use a much narrower vanishing width
than actual contact relaxation, and input controls use endpoint-preserving
smooth projection so a true zero lateral control is not changed to width/2.
The local positive-gap derivative always opens a gap; the full articulated
geometry remains nonconvex and is not promised globally monotone in every
articulator. No artificial straight-through reopening direction is injected.
