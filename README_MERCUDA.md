# MERCUDA

MERCUDA is a derivative of John E. Chambers' MERCURY6 with CUDA backends,
extended forces, and corrections shared by CPU and GPU integration. It retains
the original integrators, Makefile workflow, three executables, and `.in` files.
Having a GPU alone does **not** enable it: build CUDA support and select a backend
in `param.in`.

This guide covers MERCUDA-specific behavior. For ordinary input formats,
coordinate choices, timestep/tolerance selection, output extraction, and the
standard restart workflow, use the [inherited MERCURY6 README](README_MERCURY6.md)
and [original manual (`mercury6.man`)](mercury6.man), especially sections 2–7.
Those historical documents describe MERCURY6; the differences below take
precedence where they disagree, including compilation, PN, A2, and restarts.

## Differences from MERCURY6, including CPU runs

| Area | MERCUDA behavior |
| --- | --- |
| Integration methods | Retains BS, BS2, RADAU, MVS and HYBRID recurrences, with a CPU or CUDA backend; does not substitute a new integrator. |
| CPU performance | Removes unnecessary work for massless bodies in initialization/energy calculations, improves name checking, and fuses PN into the gravity pass. A speedup over original MERCURY6 includes these CPU improvements. |
| Body capacity | Allocates from input counts in all three executables instead of the original fixed 2,000-body limit. Memory, output encoding and disk space still limit large runs. |
| Integration corrections | Direction-aware adaptive scheduling, BS2 error-norm correction, signed BS stages, RADAU velocity-dependent prediction, and event/impact timing corrections apply on CPU too. |
| Relativity | Honors the existing input switch and uses a central-mass Cartesian Schwarzschild 1PN acceleration, replacing the supplied Marion-based prescription. Original unmodified MERCURY6's PN routine was a placeholder. |
| Radiation pressure / PR | Preserves the supplied prescription, based on [Burns, Lamy & Soter (1979)](https://doi.org/10.1016/0019-1035(79)90050-2), [Liou, Zook & Jackson (1995)](https://doi.org/10.1006/icar.1995.1120), and [Klačka et al. (2012)](https://doi.org/10.1111/j.1365-2966.2012.20321.x). Original unmodified MERCURY6's PR routine was a placeholder. |
| Non-gravitational coefficients | A1/A3 retain the cometary law. A2 defaults to zero and now specifies an inverse-square transverse Yarkovsky acceleration, including for massive bodies. Old cometary A2 values have a different meaning. |
| Diagnostics and dumps | Reports the execution backend and workload/timing counters; dumps preserve force parameters and identify the force model. |

**MERCUDA CPU is not identical to either original MERCURY6 or the supplied
MERCURY6 (+PN, PR).** Even with every additional force disabled, corrections to
scheduling and error control can change the accepted steps. CPU and CUDA use the
same MERCUDA force models, but floating-point evaluation/reduction order can
produce small trajectory differences. Compare convergence and physical outputs,
rather than expecting bitwise agreement.

## Step-by-step migration and use

### 1. Build the executables

Install GNU Make, gfortran, and a C++ compiler (for example g++). For CUDA also
install an NVIDIA driver and CUDA toolkit containing `nvcc` and the static CUDA
runtime. A driver or GPU alone is insufficient. CUDA 12.6 and gfortran 13 were
validated; other versions have not been tested. Linux/WSL was used for validation.

From the package directory:

```sh
make
```

The Makefile discovers `nvcc` on PATH, under `/usr/local/cuda`, or under
`~/.local/opt/cuda-*`. If found, it builds CUDA support; otherwise it builds CPU
support without a CUDA dependency. To force CPU-only compilation:

```sh
make cpu
```

For a toolkit outside the discovered locations:

```sh
make NVCC=/path/to/cuda/bin/nvcc
```

CUDA defaults to the local GPU architecture. A build for a specific architecture
can use `make CUDA_ARCH=sm_89` (RTX 4070 example); this is a build option, not an
executable flag. If runtime-library discovery fails, supply
`CUDA_LIB=/path/to/libcudart_static.a`, or use `make cpu`.
When changing toolkit, architecture or compiler flags, first run
`make clean-build`; it preserves simulation inputs, outputs and dumps.

### 2. Prepare the usual input files

Use your existing MERCURY6 inputs. For a new working directory with no inputs:

```sh
make gen-in
```

This copies missing `.in` files from the supplied samples and never overwrites
existing inputs. Edit them as usual; sample integration intervals are examples,
not a short smoke test. For the formats and filename configuration refer to
[manual sections 2–3](mercury6.man). Review old A2 values and legacy dumps using
[this guide's force and restart notes](#forces-and-body-input).

### 3. Select the execution backend

Append one optional setting **after all existing settings** in `param.in`:

```text
 execution backend = cuda
```

| Value | Behavior |
| --- | --- |
| `cpu` | Default if omitted. Uses MERCUDA's CPU implementation, even in a CUDA-enabled build. |
| `cuda` | Explicit GPU request. Stops with an error if the build, device, initialization or selected configuration cannot support it. |
| `auto` | Uses CUDA for supported configurations with an available device and at least 4,096 small bodies; otherwise uses CPU. Initial CUDA allocation failure can also fall back to CPU. |

`auto` is a heuristic, not a performance guarantee. Short runs and small
ensembles can be faster on CPU. Both CPU and GPU memory must accommodate the
selected method; the same configuration and tolerance apply to either backend.

### 4. Select the algorithm and optional forces

Keep the original algorithm setting in `param.in`, for example:

```text
 algorithm (MVS, BS, BS2, RADAU, HYBRID etc) = bs
```

All five production methods have CUDA implementations. **PN, PR and nonzero A2
require BS or RADAU on both backends.** MVS small bodies must be massless.
Custom `mfo_user` forces require CPU. Consult the
[algorithm support table](#algorithm-support) for the complete assessment.

For PN, edit the existing setting rather than appending a second copy:

```text
 include relativity in integration = yes
```

For Yarkovsky drift, add `A2` on the body's ordinary non-vector parameter row
in `big.in` or `small.in`; omit it or set it to zero to disable the term:

```text
ASTEROID m=1.0d-15 r=1.0d0 d=2.5d0 A2=-3.0d-14
```

Keep that body's position/elements, velocity and spin rows in their existing
format. A2 units are AU/day² at 1 AU. For the preserved radiation/PR term use
`b=<beta>` on the same parameter row of a **massless** body, for example:

```text
DUST m=0.d0 b=1.0d-4 A2=1.0d-12
```

Beta is dimensionless and defaults to zero. The supplied PR prescription skips
massive bodies; A2 does not. Detailed equations and their limitations are
[below](#forces-and-body-input). For pure Newtonian comparisons use relativity
`no` and zero A1/A2/A3/beta (and matching J2/J4/J6 settings).

### 5. Run and inspect the selected backend

```sh
./mercury6
```

There are no new executable flags. Read `info.out` for `Execution backend: CPU`
or `Execution backend: CUDA` and the force-model messages. An automatic fallback
is recorded there; inspect the final backend message if initialization failed.
Compilation with CUDA does not establish which backend actually ran.

Postprocess with the same executables and input files:

```sh
./element6
./close6
```

Use [manual sections 4–5](mercury6.man) for element and encounter extraction.
For large particle ensembles, select only the bodies you need to avoid creating
unnecessary `.aei`/`.clo` files. I/O and postprocessing remain on CPU.

### 6. Continue, switch devices, or start a fresh calculation

Follow [manual sections 6–7](mercury6.man) for continuing/extending a run.
MERCUDA still reads dynamics from dump files when restarting: changing PN, A2,
or the integration interval only in ordinary `.in` files does not change the
saved dynamics. The exception is `execution backend` in ordinary `param.in`,
which can override the saved backend for CPU-to-CUDA or CUDA-to-CPU continuation.

Legacy dumps with enabled PN or nonzero A2 are incompatible with the revised
models and are rejected. Start a new calculation from initial conditions when
adopting those models; do not relabel an old dump as force model 1.
Legacy gravity-only or PR-only dumps remain supported.

For a fresh run, archive any results/dumps you need before removing them.
`make rm-gen` removes `.aei`, `.clo`, `.out`, `.dmp` and `.tmp` files while keeping
inputs and executables. `make clean-build` removes compiled files only.
**`make clean` retains the inherited destructive cleanup: it also removes `.in`
files and simulation outputs/dumps.** `make rm-in` removes inputs and
`make unbuild` removes executables only.

## Algorithm support

| Selector | CUDA work | Restrictions / assessment |
| --- | --- | --- |
| BS | Midpoint stages, polynomial extrapolation, shared error reduction | General forces; massive and semi-active small bodies supported |
| BS2 | Conservative position recurrence and extrapolation | Same conservative-force restrictions as CPU |
| RADAU | Stage prediction, divided differences, persistent coefficients, error reduction | General forces; original shared adaptive controller |
| MVS | Jacobi transforms, Kepler drifts, kicks, output corrector | Small bodies must be massless, as on CPU |
| HYBRID | Democratic-heliocentric drift/kick map, encounter selection, compact BS2 stages | Original changeover function and shared encounter timestep; collisions resolved in Fortran |
| TEST | MVS stepping with identity boundary transforms | Legacy diagnostic selector, not a separate production integrator |
| Close/wide binary | Unavailable | This distribution lacks their CPU drivers. A GPU port cannot be provided without first implementing and validating those methods. |
| Custom `mfo_user` | CPU only | Arbitrary Fortran cannot be invoked from a CUDA kernel. A matching device implementation and regression tests are required for each custom force. |

All implemented production algorithms have CUDA paths. No production method was
found intrinsically unsuitable for GPU execution; profitability depends on the
workload. The binary selectors are missing implementations, not evidence of a
fundamental GPU limitation.

MVS retains the original Kepler solver and output corrector. Corrected output
uses scratch arrays without modifying the live integration state. HYBRID keeps
the regular system resident, restores only encounter members after the tentative
Kepler drift, and integrates that compact subsystem in a separate CUDA BS2
workspace. Compact encounter endpoints return to Fortran each substep for the
original event and merger handling. This transfer and launch overhead can make
small encounter groups slower on a GPU. Arbitrary collision logic and file I/O
remain on the CPU.

The implementation sequence was shared force/mass semantics, BS2, RADAU, MVS,
then HYBRID. Each port retains its original recurrence and controller rather than
substituting a different integrator. Future optimization should focus on measured
launch overhead, encounter transfers, and large massive-body reductions; changes
to precision or the integration method need separate accuracy studies.

Indirect gravity, momentum and oblateness reaction use parallel block reductions
over the massive sources. Jacobi transforms and the MVS prefix recurrence remain
serial over the big bodies; systems with many big bodies can still be limited by
these kernels. Massless ensembles do not increase those serial loops.
HYBRID encounter BS2 checks timestep progress against its local encounter clock,
so a large Julian epoch does not reject otherwise representable encounter steps.
If automatic CUDA initialization fails, `info.out` records the CPU fallback.

The GPU retains the BS midpoint stages, extrapolation table, error reductions,
and encounter screening between accepted steps. The CPU retains scheduling,
files, synchronization of different input epochs, and collision/ejection
resolution. State transfers occur for output, dumps, periodic checks, actual collisions,
and compact HYBRID encounter substeps. The original shared adaptive timestep and tolerance test
are retained: one difficult orbit can limit the entire ensemble. Periodic Hill-radius
updates transfer the new radii without resetting predictor or kick history; full
state uploads are reserved for initialization and actual state changes.

Body capacity is counted from the input before allocating arrays. The original
2000-body limit is removed from all three executables. GPU workspace depends on the selected
algorithm, with additional storage for extrapolation or predictor coefficients
and encounter records. CPU encounter capacity is sized to the possible
interacting pairs, so host memory still depends on the number of massive bodies.
GPU event storage grows when necessary. The legacy output encoding limits the
body count to roughly 11.2 million; this is not a tested capacity claim.
`element6` and `close6` still require enough disk space for their selected output
files; selecting a subset is useful for large ensembles.

## Forces and body input

**PR is preserved.** The Fortran `mfo_pr` routine is unchanged from the supplied
version, including its component-wise transverse-velocity expression, solar-wind
factor 0.3, mass gating, and speed-of-light constant. CUDA implements that same
prescription. It has not been replaced with a different textbook PR formula.

Use the existing `include relativity in integration = yes` setting for solar
Schwarzschild 1PN. The parser now actually honors that setting. With heliocentric
position **r**, velocity **v**, and `mu = G Mcentral`, the additional acceleration is

```
a_1PN = mu/(c^2 r^3) [(4 mu/r - v^2) r_vector + 4 (r_vector · v) v_vector]
```

Here `c = 299792458 m/s` converted to AU/day using the package's AU. This is a
central-mass approximation, not the full relativistic N-body equations; planetary
PN cross terms, solar spin, and higher PN orders are omitted. The equation is the test-body limit (eta=0) of
[Will (2014), equation 79](https://doi.org/10.12942/lrr-2014-4), with G and c
restored. The approximation and more complete alternatives are discussed in [Tamayo et al. (2020), Appendix B](https://academic.oup.com/mnras/article/491/2/2885/5594029).

PN selection occurs outside the GPU particle kernel through separate compiled
specializations. CPU PN is fused into the final gravity pass. PN off has no PN
arithmetic or additional force pass. PN on still requires arithmetic, so its
runtime cost is measured rather than assumed to be zero.

A2 defaults to zero and can be placed on a body's ordinary parameter row in
**either `big.in` or `small.in`**, for example:

```
ASTEROID m=1.0d-15 r=1.0d0 d=2.5d0 A2=-3.0d-14
  ... existing position/elements, velocity and spin rows ...
```

A2 is in **AU/day² at 1 AU**, with

```
v_transverse = v - (r_vector · v)/r^2 * r_vector
 a_Yarkovsky = A2 * (1 AU/r)^2 * v_transverse/|v_transverse|
```

Positive A2 accelerates along orbital motion; negative A2 gives the opposite
sign. It acts on massive and massless bodies and has no cometary distance
cutoff. A nonzero A2 with an undefined transverse direction is rejected.
This is the commonly fitted transverse model, not a thermophysical model of
spin and heat transport; see [Farnocchia et al. (2013)](https://arxiv.org/abs/1212.4812).
A1 and A3 retain their cometary Marsden distance law. **A2 always means Yarkovsky
in this version**, replacing Mercury's old cometary transverse coefficient.
PN, A2, and PR require `BS` or `RADAU`; incompatible algorithms are rejected.

## Backward integration and restarts

Set stop time earlier than start time in `param.in`, as before. BS, BS2 and RADAU
now use direction-aware step clipping for synchronization, preparation, output,
and final epochs. The output interval also caps preparation/synchronization
steps. The BS2 velocity-error norm also corrects an inherited cross-component typo
(`d(5)*d(2)` becomes `d(5)*d(5)`). The same numerical tolerance can therefore
choose different steps and yield different errors from historical BS2.
BS substage force times are signed correctly, encounter checks use the
accepted step, and RADAU predictors include velocity-dependent forces and are
reset after externally imposed step changes.

MVS and HYBRID retain their fixed production timestep and output on that grid.
An output interval smaller than the internal timestep cannot create intermediate
states; passed requested epochs are advanced so later output continues. BS is
used to reach an off-grid initial start epoch before beginning a fixed-step run.
Encounter interpolation and impact timing corrections apply to CPU and CUDA.
The central-impact time remains a two-body estimate; exactly radial impacts use
a linear crossing estimate within the accepted step.

Conservative backward/forward-reversed trajectories are checked numerically,
not bit for bit. PR and Yarkovsky are velocity-dependent and do **not** obey the
same velocity-reversal comparison with unchanged coefficients. Signed-time
integration follows the supplied equations backward; drag then undoes its
forward evolution and can amplify numerical errors.

New dumps retain A2/beta precision and record `force model version = 1`.
Restarts read the dynamics from the dump files as Mercury traditionally does.
Only `execution backend` can be overridden in the ordinary `param.in`, allowing
CPU/CUDA restart changes. Legacy dumps with nonzero A2 or enabled PN are rejected
rather than silently interpreted under the changed model. Old gravitational or
PR-only dumps remain accepted. The ordinary energy report is Newtonian and
should not be interpreted as a conserved-energy error under these extra forces.

## Validation and timing

`make test` runs isolated standard-library Python regression tests; it never runs
the user's input files. Tests cover analytic force values, byte-for-byte PR
preservation, CPU/CUDA force and trajectory comparisons, reverse integration,
round trips, off-grid preparation, relativistic precession, secular A2 drift,
CPU/CUDA restart switching across all five production algorithms, postprocessing,
collisions (including a HYBRID merger and central impact in one step), ejections,
and 4100 simultaneous
encounter records. GPU tests skip when CUDA is unavailable. `make test-debug` runs the same tests
with Fortran bounds and runtime checks in a separate build directory.

## Benchmarks

See the [benchmark report](docs/benchmarks.md) for runtime and accuracy tables,
particle-count and duration plots, matched-force comparisons, measurement
settings and limitations. The local `benchmarks/` workspace remains gitignored;
only selected final figures are published with the documentation.

## References

Please cite [Chambers (1999)](https://doi.org/10.1046/j.1365-8711.1999.02379.x)
when publishing calculations based on MERCURY6/MERCUDA, and identify the MERCUDA
revision and force settings. The [reference list](docs/references.md) covers PN,
radiation/PR, Yarkovsky, cometary forces, oblateness and the IAS15 benchmark
reference. Formula comments in `mercury6_2.for` and `mercury_cuda.cu` identify the
models and distinguish the preserved PR prescription from the standard vector
formula. Inherited algorithm references remain in the
[original manual](mercury6.man) and source headers.
