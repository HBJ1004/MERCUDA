# MERCUDA — CUDA integration and extended forces for MERCURY6

This version keeps Chambers' integrators, the three ordinary executables, the
Makefile workflow, and `.in` configuration. The inherited manual is in [README.md](README.md). These notes describe
the changes to its build, force models, integration behavior, and GPU support.

```
make
./mercury6
./element6
./close6
```

`make` uses gfortran and a C++ compiler. If NVIDIA `nvcc` is available, it also
builds the CUDA backend. Otherwise it builds a CPU executable without a CUDA
dependency. `make cpu` explicitly builds CPU only. CUDA builds default to the
local GPU architecture; use `make CUDA_ARCH=sm_89` to target an RTX 4070 explicitly.
`NVCC=/path/to/nvcc` and `CUDA_LIB=/path/to/libcudart_static.a` are optional build
overrides. No new executable arguments are required. CUDA 12.6 and gfortran 13
were used for validation; other compiler/toolkit versions have not been tested.
On this machine the CUDA toolkit is installed at `~/.local/opt/cuda-12.6`,
which the Makefile discovers automatically.

Append this optional line after the existing settings in `param.in`:

```
 execution backend = cuda
```

The choices are `cpu` (default), `cuda`, and `auto`. CUDA supports the general
`BS`, conservative `BS2`, and `RADAU` algorithms, massive bodies in `big.in`, and massless or semi-active bodies in
`small.in`. Small bodies can perturb big bodies but never one another. It includes Newtonian gravity, central J2/J4/J6, solar 1PN, the
preserved PR prescription, and A1/A2/A3. A customized `mfo_user` requires CPU.
Explicit CUDA requests fail clearly for unsupported cases. `auto` chooses CUDA
only for supported cases with at least 4096 small bodies and an available device;
it falls back to CPU if initial device allocation fails. This threshold is a
heuristic, not a measured crossover for every system.

## Why the other algorithms currently use the CPU

BS, BS2, and RADAU have CUDA timesteppers in this version.
This is an implementation scope limit; the other algorithms can also be ported.
Accelerating gravity alone would still leave their integration stages on the CPU
and require state transfers during force evaluations.

A complete port must preserve each method's numerical operations: MVS needs
Kepler drifts, Jacobi transformations, and symplectic correctors; HYBRID needs both the symplectic path and encounter-driven BS switching.
Each implementation also needs CPU/GPU trajectory and encounter validation.

The GPU retains the BS midpoint stages, extrapolation table, error reductions,
and encounter screening between accepted steps. The CPU retains scheduling,
files, synchronization of different input epochs, and collision/ejection
resolution. State transfers occur for output, dumps, periodic checks, and
actual collisions. The original shared adaptive timestep and tolerance test
are retained: one difficult orbit can limit the entire ensemble.

Body capacity is counted from the input before allocating arrays. The original
2000-body limit is removed from all three executables. GPU workspace is about
760 bytes per body plus encounter records; one million bodies needs about
0.76 GB for these device arrays. CPU encounter capacity is sized to the possible
interacting pairs, so host memory still depends on the number of massive bodies.
GPU event storage grows when necessary. The legacy output encoding limits the
body count to roughly 11.2 million; this is not a tested capacity claim.
`element6` and `close6` still require enough disk space for their selected output
files; selecting a subset is useful for large ensembles.

## Forces and body input

**PR is preserved.** The Fortran `mfo_pr` routine is unchanged from the supplied
version, including its component-wise transverse-velocity expression, solar-wind
factor, mass gating, and speed-of-light constant. CUDA implements that same
prescription. It has not been replaced with a different textbook PR formula.

Use the existing `include relativity in integration = yes` setting for solar
Schwarzschild 1PN. The parser now actually honors that setting. With heliocentric
position **r**, velocity **v**, and `mu = G Mcentral`, the additional acceleration is

```
a_1PN = mu/(c^2 r^3) [(4 mu/r - v^2) r_vector + 4 (r_vector · v) v_vector]
```

Here `c = 299792458 m/s` converted to AU/day using the package's AU. This is a
central-mass approximation, not the full relativistic N-body equations; planetary
PN cross terms, solar spin, and higher PN orders are omitted. The approximation
and more complete alternatives are discussed in [Tamayo et al. (2020), Appendix B](https://academic.oup.com/mnras/article/491/2/2885/5594029).

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
steps. BS substage force times are signed correctly, encounter checks use the
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
restart files, postprocessing, collisions, ejections, and 4100 simultaneous
encounter records. GPU tests skip when CUDA is unavailable. `make test-debug` runs the same tests
with Fortran bounds and runtime checks in a separate build directory.

Benchmark scripts, reference builds, results, and figures are maintained locally
outside this package. GPU speed depends on the algorithm, particle population,
encounters, output cadence, and hardware; integration time and total runtime
measure different costs.

`make clean-build` removes compiled files and executables, preserving simulation
inputs and outputs. `make clean` retains its original purpose: it removes inputs,
simulation outputs, dumps, and executables, and now also removes the build directory.
`make rm-gen` removes generated `.aei`, `.clo`, `.out`, `.dmp`, and `.tmp` files;
`make rm-in` removes `.in` files. `make unbuild` removes only executables.
`make gen-in` copies missing sample inputs without overwriting files.
The original PR routine is retained as a small regression fixture.
