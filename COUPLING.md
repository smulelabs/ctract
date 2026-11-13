# Pressure-coupled source port

`glottis.py` ports all 17 fields of `sources/glottis.rs::GlottisParams` and
`ParametricGlottis::next`: pressure and voicing smoothers, static/closure/effort
smoothing, skew/sharpness/OQ, drift and vibrato on the 32-sample source clock,
fry, jitter/shimmer, diplophonia and the per-cycle logistic chaos map.
`source_random.py` realises Rust's mulberry32, bounded four-uniform Gaussian and
three-band drift before simulation. Callers supply those fixed realisations;
backward never redraws them. Constant per-clip controls match the Rust contract
used in the oracle; general live command dynamics are not implied.

The smooth path uses unwrapped phase, continuous cycle-bin weights over the
supplied finite cycle domain, and an analytic Abel-summed periodic phase. The
hard path uses exact phase bins. Every cycle's logistic state is an ordinary
tensor recurrence. No cycle parameters or phase accumulators are detached.
Cycle capacity is a simulation-domain bound: an insufficient fixed realisation raises rather than silently dropping
cycles. Cycle-boundary fp64 gradchecks, pressure-threshold sweeps, a cold source limit,
and nonzero sensitivities over several physical cycles supplement the
short oracle check. Strict chaos reset/activation ties retain Rust's limit.
Chaotic logistic-map sensitivities can grow exponentially with cycle count;
a uniform useful bound over arbitrary chaos/duration is not a faithful promise.

`dynamics.py` ports the branch/wall/node updates of `AirwayNetwork::step`,
including subglottal pressure feedback. At each sample:

1. `ParametricGlottis` reads the previous tracheal minus pharyngeal pressure and
   produces glottal **area**, cm².
2. The glottal port gets that area. Its inertance, slit resistance and directional
   Bernoulli coefficients are reconstructed from the actual airway geometry.
3. The momentum equation `K*abs(U)*U + (L/dt+R)*U = dp + momentum/dt` is solved
   with Rust's cancellation-safe analytic quadratic root. Smooth mode blends
   directional coefficients and relaxes contact/opening predicates.
4. Yielding-wall velocity/displacement advance on the staggered clock. Node
   pressures and parallel R–L radiation loads use Rust's Crank–Nicolson update.
   The junction compliance floor and opening/closing momentum transfer are kept.
5. Sealed-cell pressure following is a triangular solve reproducing Rust's
   in-place node order, rather than silently replacing it by a simultaneous
   update. Differentiation passes through that triangular solve and every
   recurrent tensor state; there is no custom or stopped backward.

`ExactGeometry.frame()` supplies actual sub-section minima and incisor arclength,
not harmonic-mean substitutes or a guessed tooth position. Moving-wall squeeze routing and its two-pole volume source are ported,
alongside pressure-release bursts; see [NOISE.md](NOISE.md).

`CoupledSynth.simulate()` returns the time-domain audio and the interactive
port's glottal flow. `CoupledSynth.forward()` applies the frozen-block linear
convolution renderer to that pressure-coupled flow. These are scientifically
distinct outputs: the full stepper includes supraglottal nonlinear losses;
the convolution readout is the documented linear frozen-geometry approximation.
The readout still carries gradients through the source's pressure feedback.

On the committed 384-sample synthetic Rust case at 96 kHz (450 Hz source, nonzero
fry/diplophonia/chaos and fixed Rust random realisations), measured hard-mode
errors are:

- Source-only area maximum absolute error: `3.13e-16 cm²`.
- Coupled glottal flow relative L2 error: `5.09e-14`.
- Coupled time-domain audio relative L2 error: `8.70e-14`.

No gain or latency was fitted. This is a short direct `ParametricGlottis` +
`HumanAirway` oracle, not a claim about full `HumanVoice` command dynamics,
decimation or output conditioning. fp64 gradchecks cover the source recurrence
and articulation/source/pressure through coupling and convolution; every input
has a nonzero finite sensitivity in those test cases. Cycle/threshold transitions and several-cycle sensitivities are also tested.
Moving-wall injection and automatic noise are validated separately in NOISE.md.
