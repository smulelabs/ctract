# ctract-chimera

Compute gradients of a physical vocal-tract synthesizer's audio with respect to its
articulators, anatomy and voice-source controls, so that they can be fitted to a
target sound.

A self-contained differentiable port of [CHIMERA](https://github.com/jamesondh/chimera-vocal-synth)
(MIT, Jameson Hodge), pinned to commit [`eed4a5ff2695`](https://github.com/jamesondh/chimera-vocal-synth/tree/eed4a5ff2695). CHIMERA's
reference implementation is written in Rust; "Rust" in these docs and the code
refers to it. It exposes CHIMERA's own anatomy,
articulators, pressure-coupled glottis, branching airway, turbulence and wall
motion, together with a frozen-block convolutional renderer. Positive-temperature
paths differentiate their actual forward equations, including topology changes;
there are no stopped gradients. Runtime dependencies are Python and PyTorch.

## Install and quickstart

Python 3.11+, PyTorch 2.4+. CPU is sufficient. Install an appropriate PyTorch
wheel, then, from this directory:

```sh
python -m pip install -e '.[test]'
python -m pytest -m 'not cuda'
python examples/fit_synthetic.py
```

This short example differentiates through the full source/airway recurrence,
continuous anatomy/discretisation and convolutional readout:

```python
import torch
from ctract_chimera import (
    AnatomicalSynth, DEFAULT_GLOTTIS, NEUTRAL, NoiseInputs, SourceRandomness,
)

model = AnatomicalSynth(block_size=32, ir_length=64, response_fft=128)
pose = torch.tensor([[NEUTRAL]], dtype=torch.float64, requires_grad=True)
# HumanBody fields: scale, pharynx, tongue_radius (cm), lip_width (cm).
body = torch.tensor([1.0, 1.0, 2.25, 4.6], dtype=torch.float64, requires_grad=True)
source = torch.tensor([DEFAULT_GLOTTIS], dtype=torch.float64, requires_grad=True)
pressure = torch.tensor([10.0], dtype=torch.float64, requires_grad=True)  # cmH2O
randomness = SourceRandomness.chimera(32, 96000.0, cycles=4)
noise = NoiseInputs.chimera(32)  # Fixed white realisation; gains/cutoffs are tensors.
audio = model(pose, body, source, pressure, randomness, samples=32, frame_size=1, noise=noise)
audio.square().mean().backward()
# model.time_domain(...) returns the nonlinear reference instead of convolution.
```

The tiny IR above is for a fast gradient example; use longer IRs for audio.
`AnatomicalSynth` steps the coupled source and airway sample by sample, so it is
slow: with default settings, forward and backward for 10 ms of 96 kHz audio take
about 37 s on an 8-core CPU and about 90 s on an A100.
Modules and generated fixed random inputs default to float64. Move modules,
input tensors, `SourceRandomness` and `NoiseInputs` together with `.to(...)` for
float32/CUDA. Geometry uses float64 internally even with float32 controls to
preserve deep-contact tails. Nothing selects a GPU. For optional GPU tests,
set `CTRACT_TEST_CUDA=1` and run `pytest -m cuda`.

## API, layout and units

Lengths are **cm**, areas **cm²**, time **s**, frequency **Hz**, volume velocity
**cm³/s**, series pressure **dyn/cm²**. The source's lung-pressure input is
**cmH2O**, explicitly converted internally. Output is the unscaled derivative
of total radiated flow, **cm³/s²**, rather than calibrated microphone pressure.

| File / interface | Physical role and CHIMERA source |
| --- | --- |
| `anatomy.py`: `Anatomy`, `ARTIC_NAMES`, `NEUTRAL` | `anatomy/human.rs::HumanBody`; `control/articulators.rs::ARTIC_SPECS` |
| `geometry.py`, `geometry_smooth.py`, `geometry_relaxation.py` | `HumanVocalTract::{compute,resample}` and `anatomy/geom.rs`; hard and continuous contour/contact/resampling paths |
| `topology.py`: `build_human` | `voice/human_airway.rs::HumanAirway::new`, fixed graph structure |
| `acoustics.py`: `HumanAcoustics` | Exact linear elements and reciprocal tree elimination; continuous frequency and Rust-stepper z-domain responses |
| `glottis.py`, `source_random.py` | All 17 `GlottisParams`, `ParametricGlottis::next`, fixed Rust random realisations; `source.py` exposes its pulse primitive separately |
| `dynamics.py`: `AirwayDynamics` | Nonlinear momentum/pressure/wall/radiation updates, analytic Bernoulli solve and moving-wall squeeze flow |
| `turbulence.py`: `NoiseInputs`, `Turbulence`; `noise.py` | `HumanAirway::update_noise`, `AirwayNetwork::compute_noise`, shared white stream and filters |
| `renderer.py`: `Renderer` | Frozen-block convolution of prescribed flow, branch pressure and node flow sources |
| `coupled.py`: `CoupledSynth` | Full interactive source/airway recurrence, plus convolutional readout |
| `discretisation.py`: `AnatomicalSynth`, `AnatomicalRenderer`, `AnatomyDomain` | Continuous anatomy and exhaustive finite graph mixture across changing section counts |

`src/ctract_chimera/` contains the library; `tests/` contains synthetic Rust
fixtures and CPU/optional GPU tests; `examples/` contains fitting and oracle
regeneration/diagnosis; `oracle/` is an optional Rust executable. The derivation
is in [ARCHITECTURE.md](ARCHITECTURE.md), with detailed
[source coupling](COUPLING.md), [noise/bursts](NOISE.md),
[stationary diagnosis](STATIONARY.md) and [validation](VALIDATION.md).

Geometry accepts `(...,17)` instantaneous explicit-tip articulators in
`ARTIC_NAMES` order and returns section areas, lengths and velum area. `.frame()`
also returns actual sub-section minima and incisor arclength for noise.
Rendering uses `(B,L)` source flow and `(B,T,...)` geometry; controls are spaced
by `frame_size` samples. `T=1` is stationary. `CoupledSynth` takes `(B,T,17)`
articulators, constant per-clip `(B,17)` source parameters, `(B,)` pressure and
fixed source/noise realisations. `.simulate()` returns nonlinear audio and its
connected state histories; `forward()` renders these histories by convolution.

The four tensor anatomy values update contour landmarks and **all** fixed airway
areas, lengths, ports and resistances with Rust's scaling. `AnatomicalSynth`
evaluates every graph in a finite explicit `AnatomyDomain` and mixes complete
audio, avoiding incompatible node-state interpolation. Its count-bin weights
converge to CHIMERA's floor discretisation, including integer boundaries.
`AnatomicalRenderer` provides the same anatomy mechanism for prescribed flow.
Wider domains increase cost; out-of-domain anatomy raises rather than clipping.

## Exactness, differentiation and model boundaries

- Hard geometry matches 96 synthetic Rust cases to `atol=rtol=1e-10` in fp64
  (`3e-3` in fp32). General **bitwise equality is not established**: Rust/PyTorch
  transcendental evaluation differs. Harmonic means retain Rust's left-to-right
  summation because even a one-ULP change can alter discrete noise localisation.
- Hard linear elements match at relative tolerance `1e-12` on open branches;
  four complex transfer cases match at `rtol=1e-9, atol=1e-10`. Coupled source,
  noise and moving-closure time-domain audio agree with Rust to about `1e-13`
  relative L2, without alignment or fitted gain. See the detailed reports.
- Positive geometry temperature is dimensionless; it smooths contacts,
  crossings, ordering, missing rays, adjacency, closures and resampling.
  Source/noise/count relaxations use the same dimensionless annealing scalar.
  The lower-level acoustics temperature is an **area width in cm²**; combined
  synthesizers convert it as `tau * 2e-4`. `None` selects hard predicates in
  the lower-level acoustic and coupled-reference APIs.
- Cold geometry convergence is tested on all 96 Rust poses and custom anatomy.
  The analytic limit is the Rust map on nondegenerate decision cells; declared
  inclusive/exclusive ties use vanishing biases. Arbitrary degenerate,
  self-intersecting contours are not covered by an exhaustive limit proof.
- At positive temperature the actual forward law carries gradients end to end.
  Tests require finite, nonzero per-input sensitivities in active cases,
  contact/closure sweeps and count transitions, plus fp64 module and complete
  source/anatomy/noise-to-audio gradchecks. A globally nonzero derivative is
  physically impossible for inactive controls; a uniform derivative bound as
  temperature tends to zero is impossible at a hard jump. These limits and
  deep-contact log-sensitivity tests are explained in ARCHITECTURE.md.
- The nonlinear time-domain reference retains pressure feedback, losses,
  turbulence and wall motion. Convolution is its **linear, frozen-geometry
  readout**, with finite response bandwidth and IR truncation. It is not an
  exact replacement for moving nonlinear dynamics. On the stationary
  benchmark, 16 kHz waveform error is **0.278%**, spectral error **0.0417%**,
  aligned interior waveform error **0.0409%**, best delay **zero**. At native
  rate a sufficiently long IR gives **2.853e-8** relative waveform error.
- Inputs are instantaneous geometry/control ramps and constant source controls.
  Full `HumanVoice` command tracking, resting-tip substitution, live contact
  projection, physiological slew limits, output conditioning and decimation
  are outside this API. Fixed random/cycle capacity is explicit and validated;
  insufficient capacity raises instead of silently truncating simulation.

## Synthetic fitting and tests

`examples/fit_synthetic.py` recovers jaw, lip aperture, larynx height and source
flow gain from a 24-tone target rendered by Rust's actual time stepper. It uses
a Hann-windowed spectral loss (magnitude plus a small complex-spectrum term)
and L-BFGS gradient descent with line search. The documented CPU run reduces loss
from **0.458475 to 6.014e-8** and recovers each control within **0.001**.

From this directory, build the optional oracle (needs a Rust toolchain; Cargo
fetches the pinned CHIMERA commit), then regenerate stationary and fitting audio:

```sh
cargo build --release --manifest-path oracle/Cargo.toml
python examples/regenerate.py
python examples/regenerate_fit.py
python examples/diagnose_stationary.py
python -m pytest -m 'not cuda'
```

The oracle is not a runtime dependency. Its JSON stdin/stdout
interface also supports custom anatomy, coupled sources and moving gestures.
Fixture provenance and exact commands are retained alongside the generators.
Named bodies in the original synthetic geometry fixture are test labels only;
the library does not expose them as presets. See VALIDATION.md for test scope.

## Licence and attribution

This package is MIT; see [LICENSE](LICENSE). CHIMERA-derived algorithms and
constants are attributed to Jameson Hodge, with the upstream MIT notice in
[LICENSE-CHIMERA](LICENSE-CHIMERA). Ported routines cite corresponding Rust
files/functions.

The convolutional renderer follows the frozen-block approach associated with
[VocalTrax](https://github.com/PapayaResearch/vocaltrax), using CHIMERA's branching
lumped network. No external VocalTrax source files are bundled.
