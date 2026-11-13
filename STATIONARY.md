# Renderer timing: derivative clock and impulse window

Reproduce on the committed public synthetic Rust multisine:

```sh
python examples/diagnose_stationary.py
```

The source in the multisine comparisons below is **prescribed volume velocity**,
identical in both models. No glottal source is used. Rust integrates at 96 kHz and its
output is point-sampled at 16 kHz; there is no fitted gain or resampler latency.
Those numbers validate the prescribed-flow path; the separate branch-pressure
regression at the end checks the pressure-source clock.

## Differentiating on the internal clock

Rust computes `96000*(U[n]-U[n-1])` **before** downsampling. Differencing
**after** downsampling, `16000*(U[m]-U[m-1])`, would add half a sample of phase
delay at the output rate, an extra delay of

`(1/16000 - 1/96000)/2 = 26.0416667 microseconds`.

With output-rate differencing, offline fractional alignment measures **26.25
microseconds** on a 0.3125 µs search grid, and alignment alone reduces the
waveform error from **30.5567%** to **1.76783%**: the error is a clock offset,
not a different glottis or an arbitrary transport-latency offset.

The correction factors `D_internal = D_output * K(f)`, where

```
K(f) = exp(j*pi*f*(1/F_output - 1/F_internal))
       * sinc(f/F_internal) / sinc(f/F_output)
```

(`sinc(x)=sin(pi*x)/(pi*x)`). K multiplies the flow response before convolution;
the final output-rate difference retains an exact DC zero. Directly truncating
a pressure IR instead leaks DC, so moving the difference inside the IR is not
sufficient.

## Discrete response and finite-band impulse support

A continuous jω model, or a sine warp alone, also misses the integrator's mixed
clocks. `HumanAcoustics.discrete_responses` uses the exact z-domain linear
response of `AirwayNetwork::step`: backward-Euler branch resistance/momentum,
staggered walls, Crank–Nicolson radiation, and the junction compliance floor.
The continuous `transfer` API serves Rust's frequency-domain oracle.
Both use the same differentiable tree elimination; no time/phase is fitted.

Projecting an internal-rate response onto the 16 kHz frequency interval has
negative-time tails. The renderer keeps them with a
centred finite impulse window and removes its fixed indexing offset after
overlap-add. This is not a fitted latency. Frozen-block partition invariance
and gradients remain tested. Motion still has the documented frozen-tail
approximation; it plays no role in this stationary case.

| Model | Waveform error | Hann spectral-magnitude error | Best advance | Interior aligned waveform error |
| --- | ---: | ---: | ---: | ---: |
| Continuous response, output-rate difference | 30.5567% | 1.73034% | 26.25 µs | 1.76783% |
| Native-rate derivative, continuous response | 1.37427% | 1.32741% | 0.3125 µs | 1.36604% |
| Discrete response, centred IR (as implemented) | **0.277620%** | **0.0416961%** | **0 µs** | **0.0408750%** |

The first waveform column excludes the first 256 samples. Alignment uses samples `[256:-256]` to exclude **both** finite-clip
boundaries; it fits delay only, never gain. Spectral error uses the full clip
with a Hann window. The implemented renderer's interior error is attained at **zero delay**;
its reduction relative to the first column is boundary exclusion, not hidden
latency compensation.

## What remains inherent, and what is finite-IR error

At 16 kHz, continuing only the **known synthetic formula** beyond the scoring
interval lowers unaligned waveform error to **0.005129%** and spectral error to
**0.003125%**. This identifies finite-band/source-boundary effects. The library
does not invent future source samples; for a strictly causal comparison, render
at the native rate and then downsample explicitly.

Native 96 kHz rendering with a 32768-sample IR and 65536-point response grid has
**2.853e-8 relative waveform error (0.000002853%)**, over the **entire clip**, with
no alignment or gain. A shorter 16384-sample native IR gives 0.041272% error;
reducing source amplitude by 10×/100× leaves that short-IR error unchanged, while
extending the IR removes it. It is truncation, not nonlinear source behaviour.
The remaining long-IR discrepancy is consistent with the tiny Bernoulli losses
absent from the linear transfer, plus numerical precision; the much larger
source-clock and discarded-tail errors do not arise with the native-rate
derivative and the centred window.

Regression bounds are <0.5% original-interval waveform, <0.1% spectral and
interior waveform, and <1e-6 relative native-rate full-clip error.

## Branch-pressure timing: impulse check

The multisine comparison does not exercise the branch-pressure path, which has
its own one-sample timing. `AirwayNetwork::step` uses previous-step **node
pressures**, but current-step **series pressure sources**. With
`z1 = exp(-j*2*pi*f/F_internal)` and `d = F_internal*(1-z1)`, its linear
momentum equation is

```
(R + L*d)*U = z1*(P_a - P_b) + S
```

The nodal solve therefore needs `Z_effective = (R + L*d)/z1`. Reusing that
impedance directly for the dipole response would delay S as well, so
`discrete_responses` returns its solved dipoles **divided by z1**, leaving
the node response, nodal timing and continuous-frequency API unchanged.

`test_native_pressure_impulse_matches_time_stepper_without_alignment` compares
the renderer directly with `AirwayDynamics.step(temperature=None)` at 96 kHz,
neutral hard geometry, closed glottis and zero lung pressure. The only input is
a **1e-6 dyn/cm² impulse on the last internal oral branch**. It scores all 1024
samples with a 32768-sample IR and 65536-point response grid:

- Without the division by z1: **1.22405979969 relative L2 waveform error**.
- With it (as implemented): **4.9841853e-12 relative L2 waveform error**.
- Regression bound: **<1e-10**, with **no alignment, cropping or fitted gain**.

This is native-rate, small-signal pressure-source coverage against the ported
time-domain solver, whose nonlinear source/noise cases are separately checked
against Rust. It is not a claim that arbitrary moving/noisy convolution matches
the full nonlinear Rust stepper. The linear and frozen-tail limitations remain.
