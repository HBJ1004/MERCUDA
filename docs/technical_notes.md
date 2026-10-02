# MERCUDA technical notes

These notes describe the force equations, numerical changes, CUDA implementation,
and build options. For ordinary use and a first-run example, see the
[user guide](../README_MERCUDA.md). The [MERCURY6 manual](../README_MERCURY6.md)
describes the inherited methods and file formats.

## Changes from MERCURY6

| Area | MERCUDA behavior |
| --- | --- |
| Integration methods | Retains BS, BS2, RADAU, MVS and HYBRID recurrences, with a CPU or CUDA backend; does not substitute a new integrator. |
| CPU performance | Removes unnecessary work for massless bodies in initialization/energy calculations, improves name checking, and fuses PN into the gravity pass. A speedup over original MERCURY6 includes these CPU improvements. |
| Body capacity | Allocates from input counts in all three executables instead of the original fixed 2,000-body limit. Memory, output encoding and disk space still limit large runs. |
| Integration corrections | Direction-aware adaptive scheduling, BS2 error-norm correction, signed BS stages, RADAU velocity-dependent prediction, event/impact timing corrections, and a local-clock step check for HYBRID close encounters apply on CPU too. |
| Relativity (PN) | Honors the existing input switch and uses a central-mass Cartesian Schwarzschild 1PN acceleration. Original MERCURY6's PN routine was a placeholder. |
| Radiation pressure / PR | Radiation pressure and PR drag for massless bodies, based on [Burns, Lamy & Soter (1979)](https://doi.org/10.1016/0019-1035(79)90050-2), [Liou, Zook & Jackson (1995)](https://doi.org/10.1006/icar.1995.1120), and [Klačka et al. (2012)](https://doi.org/10.1111/j.1365-2966.2012.20321.x). Original MERCURY6's PR routine was a placeholder. |
| Non-gravitational coefficients | A1/A2/A3 retain the cometary law. The separate `yar` input defaults to zero and specifies an inverse-square transverse Yarkovsky acceleration, including for massive bodies. |
| Diagnostics and dumps | Reports the execution backend and workload/timing counters; dumps preserve force parameters and identify the force model. |

**MERCUDA CPU is not identical to original MERCURY6.** Even with every
additional force disabled, corrections to
scheduling and error control can change the accepted steps. CPU and CUDA use the
same MERCUDA force models, but floating-point evaluation/reduction order can
produce small trajectory differences. Compare convergence and physical outputs,
rather than expecting bitwise agreement.

## Build options

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

## Algorithm support

| Selector | CUDA work | Restrictions / assessment |
| --- | --- | --- |
| BS | Midpoint stages, polynomial extrapolation, shared error reduction | Supports all built-in forces, including velocity-dependent and dissipative terms; massive and semi-active small bodies supported |
| BS2 | Conservative position recurrence and extrapolation | Gravity and oblateness only: rejects PN, PR, `yar` and A1/A2/A3, as on CPU |
| RADAU | Stage prediction, divided differences, persistent coefficients, error reduction | Supports all built-in forces, including velocity-dependent and dissipative terms; original shared adaptive controller |
| MVS | Jacobi transforms, Kepler drifts, kicks, output corrector | Small bodies must be massless, as on CPU |
| HYBRID | Democratic-heliocentric drift/kick map, encounter selection, compact BS2 stages | Original changeover function and shared encounter timestep; collisions resolved in Fortran |
| TEST | MVS stepping with identity boundary transforms | Legacy diagnostic selector, not a separate production integrator |
| Close/wide binary | Unavailable | This distribution lacks their CPU drivers. A GPU port cannot be provided without first implementing and validating those methods. |
| Custom `mfo_user` | CPU only | Arbitrary Fortran cannot be invoked from a CUDA kernel. A matching device implementation and regression tests are required for each custom force. |

All implemented production algorithms have CUDA paths, and each port keeps its
original recurrence and controller rather than substituting a different
integrator. Whether the GPU is faster depends on the workload. The binary
selectors are missing implementations, not a GPU limitation.

MVS retains the original Kepler solver and output corrector. Corrected output
uses scratch arrays without modifying the live integration state. HYBRID keeps
the regular system resident, restores only encounter members after the tentative
Kepler drift, and integrates that compact subsystem in a separate CUDA BS2
workspace. Compact encounter endpoints return to Fortran each substep for the
original event and merger handling. This transfer and launch overhead can make
small encounter groups slower on a GPU. Arbitrary collision logic and file I/O
remain on the CPU.

Further optimization would target launch overhead, encounter transfers and the
remaining serial big-body loops; changes to precision or the integration method
would need separate accuracy studies.

Indirect gravity, momentum and oblateness reaction use parallel block reductions
over the massive sources. Jacobi transforms and the MVS prefix recurrence remain
serial over the big bodies; systems with many big bodies can still be limited by
these kernels. Massless ensembles do not increase those serial loops.
If automatic CUDA initialization fails, `info.out` records the CPU fallback.

The GPU retains the BS midpoint stages, extrapolation table, error reductions,
and encounter screening between accepted steps. The CPU retains scheduling,
files, synchronization of different input epochs, and collision/ejection
resolution. State transfers occur for output, dumps, periodic checks, actual collisions,
and compact HYBRID encounter substeps. The original shared adaptive timestep and tolerance test
are retained: one difficult orbit can limit the entire ensemble. Periodic Hill-radius
updates transfer the new radii without resetting predictor or kick history; full
state uploads are reserved for initialization and actual state changes.

Body capacity is counted from the input before allocating arrays in all three
executables. GPU workspace depends on the selected
algorithm, with additional storage for extrapolation or predictor coefficients
and encounter records. CPU encounter capacity is sized to the possible
interacting pairs, so host memory still depends on the number of massive bodies.
GPU event storage grows when necessary. The legacy output encoding limits the
body count to roughly 11.2 million; this is not a tested capacity claim.
`element6` and `close6` still require enough disk space for their selected output
files; selecting a subset is useful for large ensembles.

## Forces and body input

Radiation pressure and PR drag use `b=<beta>` on a body's parameter line. Beta
is dimensionless and defaults to zero; bodies with nonzero mass receive no PR
acceleration. With heliocentric distance r, radial velocity
`v_r = (r_vector · v)/r`, and `GM = k^2` (the Gaussian solar value), the
acceleration is

```
v_t,k = v_k (1 - x_k/r)        (k = x, y, z; component-wise)
 a_PR = (GM beta/r^2) [(1 - 2 (1+sw) v_r/c) r_hat - (1+sw) v_t/c]
```

with solar-wind factor `sw = 0.3` and `c = 173.1 AU/day`. Written in this
radial/transverse form, the standard formula of
[Burns, Lamy & Soter (1979)](https://doi.org/10.1016/0019-1035(79)90050-2) has
`v_t = v - v_r r_hat`; MERCUDA uses the component-wise expression above instead.
CPU (`mfo_pr`) and CUDA evaluate the same expression.

Use the existing `include relativity in integration = yes` setting for solar
Schwarzschild 1PN (PN). With heliocentric
position **r**, velocity **v**, and `mu = G Mcentral`, the additional acceleration is

```
a_1PN = mu/(c^2 r^3) [(4 mu/r - v^2) r_vector + 4 (r_vector · v) v_vector]
```

Here `c = 299792458 m/s` converted to AU/day using the package's AU. This is a
central-mass approximation, not the full relativistic N-body equations; planetary
PN cross terms, solar spin, and higher PN orders are omitted. The equation is the test-body limit (eta=0) of
[Will (2014), equation 79](https://doi.org/10.12942/lrr-2014-4), with G and c
restored. The approximation and more complete alternatives are discussed in [Tamayo et al. (2020), Appendix B](https://doi.org/10.1093/mnras/stz2870).

PN selection occurs outside the GPU particle kernel through separate compiled
specializations. CPU PN is fused into the final gravity pass. PN off has no PN
arithmetic or additional force pass. PN on still requires arithmetic, so its
runtime cost is measured rather than assumed to be zero.

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

Positive `yar` accelerates along orbital motion; negative `yar` gives the opposite
sign. It acts on massive and massless bodies and has no cometary distance
cutoff. A nonzero `yar` with an undefined transverse direction is rejected.
This is the commonly fitted transverse model, not a thermophysical model of
spin and heat transport; see [Farnocchia et al. (2013)](https://doi.org/10.1016/j.icarus.2013.02.004).
The coefficient called A2 in Farnocchia et al. is named `yar` in the input files
to distinguish it from MERCURY6's cometary A2. The two distance laws differ, so a
coefficient fitted for one cannot be reused for the other. Initial input files
from earlier MERCUDA versions that used A2 for Yarkovsky must be edited to use
`yar`; restart dumps migrate automatically (see
[restarts](#backward-integration-and-restarts)).

A1, A2 and A3 retain the original Marsden cometary law:

```
q = r/(2.808 AU)
g(r) = 0.111262 q^(-2.15) [1 + q^5.093]^(-4.6142)
a_comet = g(r) [A1 r_hat + A2 t_hat + A3 n_hat]
```

Here `t_hat` is the normalized transverse velocity and `n_hat` is the normalized
orbital angular momentum. The inherited cometary cutoff applies unless
`r^2 < 88 AU^2` or any of `|A1|`, `|A2|`, `|A3|` exceeds `1e-7 AU/day²`.
Yarkovsky has no such cutoff. Both transverse terms can be enabled together;
their accelerations are added. Undefined directions are rejected only when the
corresponding nonzero coefficient requires them.
PN, `yar`, and PR require `BS` or `RADAU`; other algorithms reject them.
A1/A2/A3 work with BS, RADAU, MVS and HYBRID; BS2 rejects them, as in MERCURY6.

The internal non-gravitational array has five components per body:
`[A1, A2, A3, beta, yar]`. CUDA uploads all five. The four-component Fortran PR
routine receives the `ngf(1:4,:)` section, so beta keeps its original position.
External callers of the CUDA upload API must supply the five-component array.

## Backward integration and restarts

Set stop time earlier than start time in `param.in`, as before. BS, BS2 and RADAU
use direction-aware step clipping for synchronization, preparation, output,
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
HYBRID close-encounter BS2 substeps check timestep progress against the local
encounter clock on both backends, so a large Julian epoch does not reject
representable substeps.
The central-impact time remains a two-body estimate; exactly radial impacts use
a linear crossing estimate within the accepted step.

Conservative backward/forward-reversed trajectories are checked numerically,
not bit for bit. PR and Yarkovsky are velocity-dependent and do **not** obey the
same velocity-reversal comparison with unchanged coefficients. Signed-time
integration follows the specified equations backward; drag then undoes its
forward evolution and can amplify numerical errors. BS and RADAU support this on
CPU and CUDA: use an earlier stop time with the same physical velocities, beta
and all non-gravitational coefficients. Do not negate these parameters to obtain a backward run.

For bound orbits with positive beta, the PR contribution usually decreases
semimajor axis forward in time, so its secular trend is traced outward into the
past. The fitted Yarkovsky term has forward secular drift with the sign of `yar`
([Farnocchia et al. 2013, equations 1–5](https://doi.org/10.1016/j.icarus.2013.02.004)); integrating
those equations toward earlier times traces positive-`yar` drift inward and
negative-`yar` drift outward. These describe the contributions of the added forces,
not a guarantee that the total orbit changes monotonically when planetary
perturbations or encounters are present. Recovering a trajectory under fixed
coefficients is mathematically possible; it does not establish the actual past
values of beta, spin, or thermal properties.

New dumps retain A2/yar/beta precision and record `force model version = 2`.
Restarts read the dynamics from the dump files as Mercury traditionally does.
Only `execution backend` can be overridden in the ordinary `param.in`, allowing
CPU/CUDA restart changes. Version-1 MERCUDA dumps automatically move their old
Yarkovsky A2 to `yar` and clear cometary A2. A version-1 dump that also contains
nonzero `yar` is rejected as ambiguous. Unversioned legacy dumps retain the
original cometary A2 interpretation; they are rejected only if PN is enabled,
because the PN equation changed. Old gravitational, cometary and PR-only dumps
remain accepted. The ordinary energy report is Newtonian and
should not be interpreted as a conserved-energy error under these extra forces.

## Validation

`make test` runs isolated standard-library Python regression tests; it never runs
the user's input files. Tests cover analytic force values, PR accelerations
checked against a frozen reference routine, CPU/CUDA force and trajectory comparisons, reverse integration,
round trips, off-grid preparation, relativistic precession, secular Yarkovsky drift,
CPU/CUDA restart switching across all five production algorithms, postprocessing,
collisions (including a HYBRID merger and central impact in one step), ejections,
and 4,100 simultaneous
encounter records. GPU tests skip when CUDA is unavailable. `make test-debug` runs the same tests
with Fortran bounds and runtime checks in a separate build directory.

## Further reading

See the [benchmark report](benchmarks.md) for runtime and accuracy measurements,
and the [reference list](references.md) for the integrator and force-model
literature. Source comments cite the equations at their implementations.

## Validation

See [Testing MERCUDA](validation.md) for the independent reference models,
acceptance bounds, configuration coverage and commands for reproducing checks.

## Parabolic orbits

Cometary input with `e=1` uses Barker's equation. An exactly zero-energy orbit
uses its current distance for the encounter scale, as other unbound orbits do.
The Cartesian-to-elements conversion uses the matching Barker mean anomaly.
For an exact parabola, `element6` reports the semi-major axis and aphelion as
`Infinity`; Cartesian coordinates remain finite. These quantities have no
finite value for a parabolic orbit.
