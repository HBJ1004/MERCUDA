# MERCUDA technical notes

These notes describe the force equations, numerical changes, CUDA implementation,
and build options. For ordinary use and a first-run example, see the
[user guide](../README_MERCUDA.md). The [MERCURY6 manual](../README_MERCURY6.md)
describes the inherited methods and file formats.

## Changes from MERCURY6

| Area | MERCUDA behavior |
| --- | --- |
| Integration methods | Keeps the BS, BS2, RADAU, MVS and HYBRID recurrences, with a CPU or CUDA backend. No new integrator is substituted. |
| CPU performance | Removes quadratic work for massless bodies in initialization and energy calculations, speeds up name checking, and fuses PN into the gravity pass. This helps large particle counts. Small planetary systems can run slower than original MERCURY6 on CPU, because work arrays are sized at run time. |
| Body capacity | Allocates from input counts in all three executables instead of the original fixed 2,000-body limit. Memory, output encoding and disk space still limit large runs. |
| Integration corrections | Direction-aware adaptive scheduling, BS2 error-norm correction, signed BS stages, RADAU velocity-dependent prediction, event and impact timing corrections, and a local-clock step check for HYBRID close encounters. These apply on CPU as well as GPU. |
| Relativity (PN) | Honors the existing input switch and uses a central-mass Cartesian Schwarzschild 1PN acceleration. Original MERCURY6's PN routine was a placeholder. |
| Radiation pressure / PR | Radiation pressure and PR drag for massless bodies, based on [Burns, Lamy & Soter (1979)](https://doi.org/10.1016/0019-1035(79)90050-2), [Liou, Zook & Jackson (1995)](https://doi.org/10.1006/icar.1995.1120), and [Klačka et al. (2012)](https://doi.org/10.1111/j.1365-2966.2012.20321.x). Original MERCURY6's PR routine was a placeholder. |
| Non-gravitational coefficients | A1/A2/A3 keep the cometary law. The separate `yar` input defaults to zero and specifies an inverse-square transverse Yarkovsky acceleration, including for massive bodies. |
| Diagnostics and dumps | Reports the execution backend and workload and timing counters. Dumps keep force parameters and identify the force model. |
| Postprocessors | `element6` selects rows correctly at large Julian dates. `close6` has a validated reader with corrected orbital elements. See [Postprocessing](#postprocessing). |
| Input checks | A few malformed inputs that MERCURY6 silently tolerated now stop with an error. See [Input compatibility](#input-compatibility). |

**MERCUDA CPU is not identical to original MERCURY6.** Even with every
additional force disabled, corrections to scheduling and error control can
change the accepted steps. CPU and CUDA use the same force models, but a
different order of floating-point evaluation and reduction can produce small
trajectory differences. Compare convergence and physical outputs rather than
expecting bitwise agreement.

## Build options

Install GNU Make, gfortran, and a C++ compiler (for example g++). For CUDA, also
install an NVIDIA driver and a CUDA toolkit containing `nvcc` and the static CUDA
runtime. A driver or GPU alone is not enough. Validation used CUDA 12.6 and
gfortran 13 on Linux/WSL. Other versions have not been tested.

From the package directory:

```sh
make
```

The Makefile looks for `nvcc` on PATH, under `/usr/local/cuda`, and under
`~/.local/opt/cuda-*`. If it finds one, it builds CUDA support. Otherwise it builds
CPU support without a CUDA dependency. To force a CPU-only build:

```sh
make cpu
```

For a toolkit in another location:

```sh
make NVCC=/path/to/cuda/bin/nvcc
```

CUDA builds for the local GPU architecture by default. To build for a specific
architecture, use for example `make CUDA_ARCH=sm_89` (an RTX 4070). This is a
build option, not a program flag. If the CUDA runtime library is not found,
supply `CUDA_LIB=/path/to/libcudart_static.a`, or use `make cpu`. Before changing
the toolkit, architecture or compiler flags, run `make clean-build`. It keeps
simulation inputs, outputs and dumps.

## Algorithm support

| Selector | CUDA work | Restrictions |
| --- | --- | --- |
| BS | Midpoint stages, polynomial extrapolation, shared error reduction | Supports all built-in forces, including velocity-dependent and dissipative terms. Massive and semi-active small bodies are supported. |
| BS2 | Conservative position recurrence and extrapolation | Gravity and oblateness only. PN, PR, `yar` and A1/A2/A3 are rejected, as on CPU. |
| RADAU | Stage prediction, divided differences, persistent coefficients, error reduction | Supports all built-in forces, including velocity-dependent and dissipative terms. Uses the original shared adaptive controller. |
| MVS | Jacobi transforms, Kepler drifts, kicks, output corrector | Small bodies must be massless, as on CPU. |
| HYBRID | Democratic-heliocentric drift and kick map, encounter selection, compact BS2 stages | Original changeover function and shared encounter timestep. Collisions are resolved in Fortran. |
| TEST | MVS stepping with identity boundary transforms | MERCURY6 diagnostic selector, not a separate production integrator. |
| Close/wide binary | Unavailable | This distribution has no CPU drivers for these methods, so there is nothing to port yet. |
| Custom `mfo_user` | CPU only | Arbitrary Fortran cannot run inside a CUDA kernel. Each custom force would need a matching device implementation and its own tests. |

Every implemented production algorithm has a CUDA path, and each port keeps its
original recurrence and controller. Whether the GPU is faster depends on the
workload. The binary methods are missing implementations rather than a GPU
limitation.

MVS keeps the original Kepler solver and output corrector. Corrected output uses
scratch arrays and does not modify the live integration state. HYBRID keeps the
regular system on the GPU, restores only encounter members after the tentative
Kepler drift, and integrates that small subsystem in a separate CUDA BS2
workspace. The encounter endpoints return to Fortran after each substep for the
original event and merger handling. This transfer and launch overhead can make
small encounter groups slower on a GPU. Arbitrary collision logic and file I/O
stay on the CPU. If a pair merger and a central impact occur in the same step,
the redone step starts from a saved state that already includes the merged
body's position and momentum.

Indirect gravity, momentum and the oblateness reaction use parallel block
reductions over the massive bodies. Jacobi transforms and the MVS prefix
recurrence are still serial over the big bodies, so systems with many big bodies
can be limited by these kernels. Massless particles do not lengthen those loops.
If automatic CUDA initialization fails, `info.out` records the CPU fallback.

The GPU keeps the BS midpoint stages, extrapolation table, error reductions and
encounter screening between accepted steps. The CPU handles scheduling, files,
synchronization of different input epochs, and collision and ejection
resolution. State is transferred for output, dumps, periodic checks, actual
collisions and HYBRID encounter substeps. The original shared adaptive timestep
and tolerance test are kept, so one difficult orbit can limit the whole
ensemble. Periodic Hill-radius updates send the new radii without resetting the
predictor or kick history. Full state uploads happen only at initialization and
after actual state changes.

Further optimization would target launch overhead, encounter transfers and the
remaining serial big-body loops. Changes to precision or to the integration
method would need separate accuracy studies.

Body capacity is counted from the input before arrays are allocated in all three
executables. GPU workspace depends on the algorithm, with extra storage for
extrapolation or predictor coefficients and encounter records. CPU encounter
capacity is sized to the possible interacting pairs, so host memory still
depends on the number of massive bodies. GPU event storage grows when needed.
The original output encoding limits the body count to roughly 11.2 million, but
this has not been tested. `element6` and `close6` need enough disk space for
their output files, so select a subset of bodies for large ensembles.

## Forces and body input

Radiation pressure and PR drag use `b=<beta>` on a body's parameter line. Beta
is dimensionless and defaults to zero. Bodies with nonzero mass receive no PR
acceleration. With heliocentric distance r, radial velocity
`v_r = (r_vector · v)/r`, and `GM = k^2` (the Gaussian solar value), the
acceleration is

```
v_t,k = v_k (1 - x_k/r)        (k = x, y, z, component-wise)
 a_PR = (GM beta/r^2) [(1 - 2 (1+sw) v_r/c) r_hat - (1+sw) v_t/c]
```

with solar-wind factor `sw = 0.3` and `c = 173.1 AU/day`. Written in this
radial and transverse form, the standard formula of
[Burns, Lamy & Soter (1979)](https://doi.org/10.1016/0019-1035(79)90050-2) has
`v_t = v - v_r r_hat`. MERCUDA uses the component-wise expression above instead.
CPU (`mfo_pr`) and CUDA evaluate the same expression.

Use the existing `include relativity in integration = yes` setting for solar
Schwarzschild 1PN (PN). With heliocentric position **r**, velocity **v**, and
`mu = G Mcentral`, the additional acceleration is

```
a_1PN = mu/(c^2 r^3) [(4 mu/r - v^2) r_vector + 4 (r_vector · v) v_vector]
```

Here `c = 299792458 m/s`, converted to AU/day using the package's AU. This is a
central-mass approximation, not the full relativistic N-body equations. Planetary
PN cross terms, solar spin and higher PN orders are omitted. The equation is the
test-body limit (eta=0) of [Will (2014), equation 79](https://doi.org/10.12942/lrr-2014-4),
with G and c restored. [Tamayo et al. (2020), Appendix B](https://doi.org/10.1093/mnras/stz2870)
discusses this approximation and more complete alternatives.

PN is selected outside the GPU particle kernel through separately compiled
versions. On CPU, PN is fused into the final gravity pass. With PN off there is
no PN arithmetic and no extra force pass. With PN on the extra arithmetic has a
cost, which the benchmarks measure.

`yar` defaults to zero and can be placed on a body's ordinary parameter row in
**either `big.in` or `small.in`**, for example:

```
ASTEROID m=1.0d-15 r=1.0d0 d=2.5d0 yar=-3.0d-14
  ... existing position/elements, velocity and spin rows ...
```

`yar` is in **AU/day² at 1 AU**, with

```
v_transverse = v - (r_vector · v)/r^2 * r_vector
 a_Yarkovsky = yar * (1 AU/r)^2 * v_transverse/|v_transverse|
```

Positive `yar` accelerates along the orbital motion and negative `yar` against
it. It acts on massive and massless bodies and has no cometary distance cutoff.
A nonzero `yar` with an undefined transverse direction is rejected. This is the
commonly fitted transverse model, not a thermophysical model of spin and heat
transport. See [Farnocchia et al. (2013)](https://doi.org/10.1016/j.icarus.2013.02.004).
The coefficient called A2 in Farnocchia et al. is named `yar` in the input files
to distinguish it from MERCURY6's cometary A2. The two distance laws differ, so a
coefficient fitted for one cannot be reused for the other.

A1, A2 and A3 keep the original Marsden cometary law:

```
q = r/(2.808 AU)
g(r) = 0.111262 q^(-2.15) [1 + q^5.093]^(-4.6142)
a_comet = g(r) [A1 r_hat + A2 t_hat + A3 n_hat]
```

Here `t_hat` is the normalized transverse velocity and `n_hat` is the normalized
orbital angular momentum. The inherited cometary cutoff applies unless
`r^2 < 88 AU^2` or any of `|A1|`, `|A2|`, `|A3|` exceeds `1e-7 AU/day²`.
Yarkovsky has no such cutoff. Both transverse terms can be used together, and
their accelerations are added. An undefined direction is rejected only when a
nonzero coefficient needs it.

PN, `yar` and PR require `BS` or `RADAU`, and other algorithms reject them.
A1/A2/A3 work with BS, RADAU, MVS and HYBRID. BS2 rejects them, as in MERCURY6.

The internal non-gravitational array has five components per body:
`[A1, A2, A3, beta, yar]`. CUDA uploads all five. The four-component Fortran PR
routine receives the `ngf(1:4,:)` section, so beta keeps its original position.
External callers of the CUDA upload API must supply the five-component array.

## Parabolic orbits

Cometary input with `e=1` uses Barker's equation. An orbit with exactly zero
energy uses its current distance for the encounter scale, like other unbound
orbits. The conversion from Cartesian coordinates to elements uses the matching
Barker mean anomaly. For an exact parabola, `element6` reports the semimajor
axis and aphelion as `Infinity`, while the Cartesian coordinates stay finite.
These quantities have no finite value for a parabola.

## Backward integration and restarts

Set the stop time earlier than the start time in `param.in`, as before. BS, BS2
and RADAU use direction-aware step clipping for synchronization, preparation,
output and final epochs. The output interval also caps the steps used for
preparation and synchronization. The BS2 velocity-error norm corrects an
inherited cross-component typo (`d(5)*d(2)` becomes `d(5)*d(5)`), so the same
tolerance can choose different steps and give different errors from original
MERCURY6's BS2. BS substage force times are signed correctly, and encounter
checks use the accepted step. RADAU predictors include velocity-dependent forces
and are reset after externally imposed step changes. A reset carries no stale
prediction error into the next sequence, so frequent output does not reduce
RADAU accuracy.

MVS and HYBRID keep their fixed production timestep and write output on that
grid. An output interval smaller than the timestep cannot create intermediate
states. Requested epochs that have passed are advanced so that later output
continues. BS is used to reach an off-grid start epoch before a fixed-step run
begins. The encounter interpolation and impact timing corrections apply to CPU
and CUDA. HYBRID close-encounter BS2 substeps check their progress against the
local encounter clock on both backends, so a large Julian epoch does not reject
substeps that can be represented. The central-impact time is still a two-body
estimate, and an exactly radial impact uses a linear crossing estimate within
the accepted step.

Conservative trajectories integrated backward and then forward again are checked
numerically, not bit for bit. PR and Yarkovsky depend on velocity, so they do
**not** follow the same velocity-reversal comparison with unchanged coefficients.
Integration with negative time steps follows the specified equations backward.
Drag then undoes its forward evolution and can amplify numerical errors. BS and
RADAU support this on CPU and CUDA. Use an earlier stop time with the same
physical velocities, beta and all non-gravitational coefficients. Do not negate
these parameters to obtain a backward run.

For bound orbits with positive beta, the PR contribution usually decreases the
semimajor axis forward in time, so its secular trend is traced outward into the
past. The fitted Yarkovsky term drifts forward in time with the sign of `yar`
([Farnocchia et al. 2013, equations 1–5](https://doi.org/10.1016/j.icarus.2013.02.004)).
Integrating those equations toward earlier times traces positive-`yar` drift
inward and negative-`yar` drift outward. These are the contributions of the added
forces. The total orbit need not change monotonically when planetary
perturbations or encounters are present. Recovering a trajectory under fixed
coefficients is mathematically possible, but it does not establish the actual
past values of beta, spin or thermal properties.

Dumps keep A2, `yar` and beta at full precision and record the force model.
Restarts read the dynamics from the dump files, as MERCURY6 always has. Only
`execution backend` can be changed in the ordinary `param.in`, which lets a run
switch between CPU and CUDA when it is continued. Dumps written by original
MERCURY6 keep the cometary meaning of A2. They are rejected only if relativity
is enabled, because original MERCURY6 had no working relativity model to
continue. The energy report in `info.out` is Newtonian and should not be read as
a conservation error when extra forces are enabled.

## Input compatibility

A few inputs that original MERCURY6 silently tolerated now stop with a clear
error, rather than risk an altered run:

- A `files.in` line must contain only the filename, without trailing text.
- Hyperbolic Asteroidal input (`e > 1`) requires a negative semimajor axis.
- `message.in` must not end with a blank record. Use the supplied file.
- `ndump` and `nfun` in `param.in` must be positive.

`close.in` keeps the original reader's tolerance. Only the first letter of the
time-unit (`d`/`y`) and relative-time (`y`/`n`) answers is read, and a selection
line uses its first word, truncated to 25 characters. Answers starting with any
other letter are rejected.

## Postprocessing

### Element output selection

`element6` writes a row when the time since the previous row reaches the
minimum output interval. As in the original program, the comparison allows 0.1%
slack. This keeps rows scheduled one interval apart when the output times
written by `mercury6` drift slightly, for example with a 0.1-day interval at a
large Julian date. It also allows for the precision of the compressed
timestamps, whose seven base-224 digits resolve only about a nanoday at modern
Julian dates.

### Close-encounter output

`close6` is a CPU postprocessor for the original MERCURY6 encounter format. Its
validated reader is in `mercury_close.f90`, and `close6.for` keeps the original
header format and conversion helpers. Integration and encounter recording are
separate from this reader.

The encounter writer normalizes velocities using the central mass only. The
reader follows that convention, then uses the sum of central and body mass for
the reported osculating Keplerian elements. This corrects a mass-dependent
velocity error in the original reader. The semimajor axis comes from the energy,
and eccentricity and inclination come from orbital invariants, without computing
unused orbital angles. A radial orbit can have a finite semimajor axis and
`e=1`, with an undefined inclination. `Infinity` marks zero energy at working
precision. `NaN` marks an undefined inclination, including angular momentum that
cannot be distinguished from zero within floating-point product bounds. These
are reporting conventions and do not change the integrated force model.

Ordinary rows keep the original column order and fixed decimal formats. Distances
have 8 decimal places in AU, semimajor axes 4 in AU, eccentricities 6, and
inclinations 3 in degrees. Days use 5 decimal places and relative years use 7.
A numeric field that would overflow switches to `ES17.8E3`, which makes that row
wider. Scripts should accept whitespace-separated fields, `Infinity` and `NaN`.
Absolute years are Julian or Gregorian calendar dates, with the transition on
1582-10-15 and astronomical year numbering. Calendar output accepts decoded
times within ±1e12 Julian days. Larger values can use days or relative years.

All files are checked before any output is opened. Headers may change the active
body population, codes and masses, and each encounter must reference two active
bodies. Storage grows as needed, with hashed name lookup. Output files are
processed in batches of at most 256. Duplicate selections are merged. Duplicate
names or codes within a header, and collisions between output filenames, are
errors. A selected body with no encounters gets a file with only its header.

With several input files, rows follow file order and the recorded encounter
order. Records are not sorted or deduplicated. Relative time uses the first
header's epoch in each file, whereas the original reader used the first file's
origin for all files. Later headers within a file do not change its origin.
Existing outputs are skipped with a warning, as in the original program.

A zero stored velocity fraction can be valid for a fast encounter, because the
encoding cannot represent the speed. Such an event is kept, its orbital elements
are reported as `NaN`, and a warning is printed. Body names may contain
non-ASCII bytes within the original 25-byte limit. Malformed records are still
rejected.

See section 5 of the [MERCURY6 manual](../README_MERCURY6.md) for the input
format and the HYBRID recording limitation.

## Validation

`make test` runs isolated regression tests written with the Python standard
library. It never runs the user's input files. The tests cover analytic force
values, PR accelerations checked against a frozen reference routine, CPU and CUDA
force and trajectory comparisons, reverse integration, round trips, off-grid
preparation, relativistic precession, secular Yarkovsky drift, CPU and CUDA
restart switching for all five production algorithms, postprocessing,
collisions (including a HYBRID merger and central impact in one step),
ejections, and 4,100 simultaneous encounter records. GPU tests are skipped when
CUDA is unavailable. `make test-debug` runs the same tests with Fortran bounds
and runtime checks in a separate build directory.
[Testing MERCUDA](validation.md) describes the full validation campaign, its
independent references and its acceptance bounds.

## Further reading

See the [benchmark report](benchmarks.md) for runtime and accuracy measurements,
and the [reference list](references.md) for the integrator and force-model
literature. Source comments cite the equations where they are implemented.
