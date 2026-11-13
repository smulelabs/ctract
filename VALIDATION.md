# Validation

The reference is pinned to CHIMERA `eed4a5ff2695`. Every audio fixture is public
synthetic input rendered by CHIMERA's Rust crate at that commit, never a recording or a
Python-rendered fitting target. Results below are CPU/fp64 unless stated.

## Coverage

| Area | Implemented and checked |
| --- | --- |
| Geometry topology/contact/resampling | Continuous ray visibility, ordering, greedy pairing, missing-ray adjacency, closure membership and dense resampling; 96-pose cold limit; fp64 primitive/full-geometry gradchecks; six contact sweeps checking all 17 sensitivities; deep-lip log-area reopening and fp32-control tests |
| Anatomy/discretisation | Four continuous HumanBody fields, all fixed material-element scalings, exhaustive finite graph atlas; custom Rust anatomy oracle; inclusive count-bin limit, count-boundary sensitivities and prescribed-flow/complete-synth gradchecks |
| Pressure-coupled glottis | All 17 source controls, fixed Rust jitter/shimmer/drift, cycle/chaos recurrence; analytic Bernoulli and triangular sealed-node solves; Rust source/coupled oracle, cycle-boundary gradchecks, pressure sweeps and several-cycle sensitivities |
| Noise/bursts | True sub-section minima, soft rank/localisation and jet/incisor bins, all three families, shared continuous white cursor, filter identity changes, Reynolds law, pressure release and moving-wall squeeze flow; static/moving Rust oracles and full moving-readout gradcheck |
| Stationary audio | Prescribed-flow Rust multisine: derivative-clock correction, discrete response and centred IR; bandwidth/boundary and IR sweeps. Separate native-rate branch-pressure impulse versus the ported time stepper, full clip with no alignment; see STATIONARY.md |
| Complete pipeline/fitting | fp64 complete source/anatomy/graph/noise/convolution gradcheck on 32 samples; finite nonzero gradients for all 17 articulators, four anatomy fields, 17 source fields, lung pressure, three noise gains and six cutoffs; four-parameter Rust-target recovery |

The end-to-end test includes multiple graph components, with no fitted-value
selection or component pruning. The separate moving-readout test includes wall
motion. Finite differences perturb noise bands in units of 10 kHz, then apply
the exact chain rule to the public Hz inputs: an extremely small Hz perturbation
would otherwise fall below fp64 resolution after the high-amplitude recurrent
solve. This scaling changes neither the model nor the derivatives.

## Measured numerical results

| Check | Result |
| --- | ---: |
| Hard geometry, maximum absolute discrepancy across 96 cases | 6.22e-14 cm / cm² |
| Complex linear transfer, maximum pointwise relative discrepancy | 1.07e-13 |
| Standalone hard glottal area, maximum absolute discrepancy | 3.13e-16 cm² |
| Coupled hard glottal flow, relative L2 discrepancy | 5.09e-14 |
| Coupled hard time-domain waveform, relative L2 discrepancy | 8.70e-14 |
| Three-family noise time-domain waveform, relative L2 discrepancy | 5.65e-14 |
| Moving closure/release time-domain waveform, relative L2 discrepancy | 5.94e-14 |
| Changing noise registration time-domain waveform, relative L2 discrepancy | 3.57e-14 |
| Fixed stationary prescribed-flow renderer, 16 kHz waveform / Hann spectral error | 0.277620% / 0.0416961% |
| Fixed stationary prescribed-flow renderer, interior aligned waveform / best advance | 0.0408750% / 0 µs |
| Native-rate prescribed-flow waveform versus Rust, long IR, no alignment | 2.853e-8 relative |
| Native-rate branch-pressure impulse versus ported time stepper, full clip, no alignment | 4.9842e-12 relative |

These are observed errors on the committed synthetic cases, not universal
error bounds. Tests use explicit tolerances (e.g. hard fp64 geometry
`atol=rtol=1e-10`, native waveform `<1e-6` relative). General bitwise agreement
is unproven. Hard harmonic sums preserve Rust operation order because a
one-ULP shift can change a tied frication location even when area error is tiny.

The branch-pressure regression uses neutral geometry, 1024 samples, a closed
glottis, zero lung pressure and a 1e-6 dyn/cm² impulse on the last internal oral
branch at 96 kHz. With IR length 32768 and response FFT 65536, accounting for the
current-source/previous-node clock distinction reduces unaligned relative error
from **1.22405979969 to 4.9842e-12**; the test requires **<1e-10** over the entire
clip. No delay/gain fitting or cropping is permitted. This tests the
small-signal pressure-source convolution separately from the Rust multisine's
prescribed-flow path. The noise/burst Rust numbers above concern `.simulate()`
time-domain output; they do not establish exact noisy/moving convolution.

The four-parameter fitting run uses the actual Rust time-domain target and
ordinary autograd L-BFGS descent, taking about seven seconds on the development
CPU with two PyTorch threads. Loss falls from **0.458475 to 6.014e-8**:

| Parameter | Target | Recovered |
| --- | ---: | ---: |
| Jaw | 0.6 | 0.600685 |
| Lip aperture | 0.7 | 0.700431 |
| Larynx height | 0 | 0.0000409 |
| Source-flow gain | 1 | 0.999675 |

This demonstrates local recovery of several correlated controls, not global
identifiability or a general-purpose speech fitter.

## Known limits

1. **Nonzero derivatives everywhere.** A noise cutoff at zero noise gain, a
   vibrato rate at zero vibrato depth, or pressure before causal propagation
   can have exactly zero audio sensitivity. Adding an artificial slope would
   cease to differentiate the physical forward model. Tests activate the
   relevant paths and also sweep the implemented contact/cycle/count boundaries.
2. **Uniform bounds while cooling to hard jumps.** For a jump J over shrinking
   width epsilon, the mean-value theorem forces a derivative at least
   J/(2 epsilon) somewhere. The declared positive-temperature contact sweeps
   instead require finite/nonzero sensitivities bounded by 1e7 in physical
   control units. Deep-lip tests additionally bound log-area sensitivity away
   from zero. Genuine logistic chaos can amplify sensitivities over long runs.
3. **Finite computation for unbounded anatomy.** CHIMERA's node count is
   unbounded with scale. A finite explicit atlas domain is necessary; expanding
   it evaluates more graphs. Every candidate is evaluated and its complete
   audio is mixed, converging to the floor-selected Rust graph as tau→0.
4. **A differentiable hard discrete map at its discontinuities.** The hard
   reference is separate. Positive-temperature laws are smooth relaxations;
   their analytic limit matches nondegenerate hard decision cells and the
   declared tie biases. Arbitrary degenerate/self-intersecting contours have
   no exhaustive convergence proof. Floating-point transcendentals also prevent
   a general cross-runtime bitwise claim.
5. **Exact nonlinear motion from frozen convolution.** The nonlinear recurrence
   remains available through `CoupledSynth.simulate` and
   `AnatomicalSynth.time_domain`. Convolution retains the linear/frozen-tail,
   finite-bandwidth and finite-IR approximations described in ARCHITECTURE.md.

These limits are explicit scientific contracts, not straight-through gradient
exceptions. There is no detach, stopped gradient or custom backward in the
positive-temperature forward pipeline.

## Reproduction

From the package directory, `python -m pytest -m 'not cuda'` runs **54 CPU tests**.
The optional CUDA test runs with `CTRACT_TEST_CUDA=1 python -m pytest -m cuda`.
The wheel and sdist both include the two MIT notices.

Not yet covered: broader contour/adversarial numerical validation and listening
evaluation.
