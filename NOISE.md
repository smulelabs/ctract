# Continuous CHIMERA turbulence and pressure-release bursts

`Turbulence` ports `HumanAirway::update_noise` and
`AirwayNetwork::compute_noise`. It consumes **actual sub-section minima** and
`HumanFrame.teeth_s` from `ExactGeometry.frame()`. No harmonic-area substitute,
fixed tooth offset or additional burst envelope is used.

## Localisation and placement

The hard path scans the narrowest **strictly positive** minimum, preserving
Rust's first-index tie rule. The smooth path uses continuous visibility and
algebraic pairwise ranks with a vanishing lexicographic bias. Cooling selects
the same minimum on nondegenerate inputs; finite-temperature ranks keep useful
sensitivities when an ordinary area-softmax would underflow. Integer jet offsets
(`max(1,round(.9*scale/dx))`) and incisor rounding use inclusive CDF-bin weights.
Every candidate's actual Rust pharynx/oral seam exclusions and branch clamps
are represented. The pressure-source signs blend continuously through zero
flow, tending to forward/upstream routing in the hard limit.

Rust's area/distance gates, three gains and LP/HP bands are retained. Reynolds
growth saturates at the same `12000²-1800²` denominator; temperature relaxes its
threshold and the strict `area>closed_area` predicate. The three gains and six
cutoffs are explicit trainable tensors. Causal HP→LP→LP recurrences start at
rest and remain connected to gains, localisation and flow throughout backward.

## Registration, filter identity and the random stream

`NoiseInputs` supplies a fixed mulberry32 white stream from Rust seed 12345,
with Rust's `bipolar()*1.732` scaling. This is **one shared stream**, not one
independent stream per noise family. The draw cursor advances by continuous
registration mass; each family samples the same fixed stream through continuous
bin weights. Its cold limit consumes exactly one draw per registered source,
in Rust order. An L-sample clip needs at most 3L fixed white samples.

At each control update, filter state is retained only with continuous probability
of the same `(src,fwd)` identity, exactly Rust's matching rule in the hard limit.
Changing/absent sources reset it; backward sources are not incorrectly included
in the identity key. This matters when frication appears/disappears during a
closure. Finite-temperature states are a deterministic continuous extension,
not a claim to exactly marginalise all stochastic registration histories.

## Physical burst release and wall displacement

The recurrent airway retains pressure, momentum and yielding-wall state during
closures and releases. Moving geometry injects `-dV/dt`, with Rust's squeeze
routing: a sealed node keeps a smoothstep fraction and shares the remainder
with eligible neighbours, or absorbs it laterally when none are open. Earlier
neighbours use updated geometry; later neighbours use old geometry, while
endpoint target minima also determine eligibility. A two-pole 60 Hz filter
follows this mass-conserving routing. Node/radiation closure relaxations and
opening/closing momentum transfer are retained.

The control convention matches `apply(target,0)` + `begin_block(frame_size)`:
the initial frame is snapped, then sample n advances from the previous frame
by `(n+1)/frame_size` toward the next target. Source registration uses that
next control target, as `HumanAirway` does before the area ramp. This is an
instantaneous-area/control-ramp contract, not `HumanVoice` command tracking or
physiological slew limiting.

`CoupledSynth.simulate()` returns the nonlinear time-domain burst/noise output.
Its convolutional `forward()` routes the explicit series-pressure history and
moving-wall node-flow history through the same branching transfer, alongside
the interactive glottal flow. The readout retains the documented linear and
frozen-geometry approximation; simulation is the faithful nonlinear reference.

## Measured validation

Public synthetic Rust oracle cases at 96 kHz give relative waveform errors:

- All three registered noise families, including fixed white draw order:
  **5.65e-14**.
- Moving lip closure/release, squeeze flow peak **118 cm³/s**:
  **5.94e-14**.
- Moving gesture with three→one→three registered sources and identity resets:
  **3.57e-14**.

These use no alignment, fitted gain, real recordings or noise resampling.
Hard true minima/incisor positions and source identities/gains are checked
separately. fp64 gradchecks cover localisation, all gains/cutoffs and the full
moving articulation/source/pressure/anatomy/noise→convolution path. Nonzero
sensitivities are asserted at nondegenerate positive-temperature test cases.
Universal nonzero sensitivity is physically impossible, e.g. a cutoff at zero
noise gain or a lung-pressure change before its wave can reach the glottis.

Hard harmonic means deliberately use Rust's eight-term left-to-right sum: a
parallel reduction moved one lateral-channel mean by one ULP and selected a
different frication branch. This is why source identity tests supplement
area-tolerance tests. General bitwise geometry equality is still not claimed;
the numerical contract remains the documented Rust tolerances and the stated
nondegenerate cold limit.
