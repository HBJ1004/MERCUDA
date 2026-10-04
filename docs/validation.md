# Testing MERCUDA

The validation suite checks the behavior promised by the user guides. It checks
CPU and CUDA calculations against independent references, as well as checking
that unsupported combinations stop with an error. It does not prove that every
possible orbit or user-written force is correct.

## Routine checks

From the repository directory:

```sh
make test
make test-debug
```

These run the regression tests using the chosen build. A CPU-only build skips
GPU-specific checks. GitHub Actions runs the CPU regression suite in optimized
and debug builds.

## Full validation

The full campaign builds its own optimized and debug CPU and CUDA programs in
`build/validation/`. It does not overwrite the programs or simulation files in
the repository directory. It requires Python development dependencies:

```sh
python3 -m venv build/validation/venv
build/validation/venv/bin/pip install -r tests/validation/requirements.txt
make VALIDATION_PYTHON=build/validation/venv/bin/python test-full
```

A working NVIDIA GPU, CUDA toolkit and NVIDIA Compute Sanitizer are required
for complete GPU validation. Set `COMPUTE_SANITIZER=/path/to/compute-sanitizer`
if it is not on your path. Without GPU hardware, use:

```sh
make VALIDATION_PYTHON=build/validation/venv/bin/python test-full-cpu
```

To repeat only the memory, initialization, synchronization and race checks:

```sh
make VALIDATION_PYTHON=build/validation/venv/bin/python test-sanitize
```

On Windows/WSL, Compute Sanitizer may report that the WDDM debugger interface
cannot initialize. A blocked sanitizer check is recorded as **incomplete**, never
as a successful check. To enable the interface, follow
[NVIDIA's Windows instructions](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#windows-specific-behavior):

1. Open a Windows Command Prompt as administrator.
2. Enable NVIDIA's debugger interface:

   ```bat
   reg add "HKLM\SOFTWARE\NVIDIA Corporation\GPUDebugger" /v EnableInterface /t REG_DWORD /d 1 /f
   ```

3. Return to WSL and repeat `make test-sanitize` with the validation Python
   setting shown above. The report must confirm that the checks actually ran.

This is a Windows setting; the validation runner does not change it.

The full campaign can take hours. A nonzero exit status means the campaign
failed or is incomplete. A CPU-only pass certifies only the CPU scope. Debug
builds enable Fortran bounds checks and traps for invalid arithmetic, division
by zero and overflow.

## What is checked

| Area | Checks |
| --- | --- |
| Build and use | Separate build profiles, sample-input workflow, postprocessing and cleanup commands |
| Forces | All 512 activation combinations of PN, PR, Yarkovsky, cometary A1/A2/A3 and J2/J4/J6 |
| Algorithms | BS, BS2, RADAU, MVS and HYBRID, including rejection of unsupported force combinations |
| Accuracy | Analytic Kepler orbits and independent IAS15 trajectories; both time directions and multiple epochs |
| Convergence | Three tolerances or timesteps; reference calculations at two IAS15 tolerances |
| Epochs | Initial epochs before and after the start time, mixed small-body epochs, Julian-date epochs and output intervals |
| Restarts | Current and older force-model dumps, incompatible dumps, backend changes and input/dump precedence |
| Inputs and outputs | Cartesian, Asteroidal and Cometary inputs; postprocessing frames, precision, time units and body selection |
| Events | Close encounters, stopping on an encounter, central impacts, ejection and mergers |
| Capacity | Empty systems, thread/block boundaries, the auto-backend threshold, 100,000 particles and 257 massive bodies |
| Errors | Missing files, invalid settings, nonfinite values, invalid orbital elements and undefined force directions |
| Custom forces | Representative time-dependent and velocity-dependent CPU forces; CUDA rejection |
| Memory | Device lifecycle and encounter-buffer growth; host AddressSanitizer/UBSan; NVIDIA Compute Sanitizer on both the lifecycle worker and real integration programs |

The claim catalog is in [tests/validation/catalog.json](../tests/validation/catalog.json).
The exact fixtures and settings are recorded in the machine-readable report.
The force reference is coded separately from MERCUDA and evaluated with
80-digit arithmetic. It differentiates a Legendre potential for oblateness.
The trajectory reference uses [REBOUND IAS15](https://rebound.hanno-rein.de/integrators/ias15/)
with separately implemented extra-force callbacks. In `param.in`, J2/J4/J6 are
dimensionless; MERCUDA multiplies them by the central radius to the appropriate
power before evaluating accelerations.

The stable-orbit acceptance bound is `1e-8` for position divided by the
characteristic distance and velocity divided by the corresponding Kepler
speed. The encounter bound is `1e-6`. Force errors must be at most `1e-12` times
the sum of absolute contributing terms, plus `1e-30` in AU/day². Reference
uncertainty must be smaller than one tenth of the trajectory bound. Refinement
must improve an error by at least a factor of two unless it is already within
ten times the reference/roundoff floor. These bounds are fixed in the catalog;
they are test acceptance criteria, not a universal accuracy guarantee.

## Close6 validation

The required `close6` group checks the actual numeric values in `.clo` files,
not just whether the program finishes. Its reference uses an independent
integer base-224 decoder and 80-digit decimal orbital invariants. A small
fixture written by pinned original MERCURY6 checks compatibility with its writer.
The reference starts from the compressed state; compression error is separate
from postprocessor error. Encounter state fields use four base-224 digits even
when the integration output precision setting is low or high.

Coverage includes:

- Circular, eccentric, near-parabolic, hyperbolic, radial and zero-velocity
  states; different inclinations and masses; 256 fixed-seed random states.
- Exact parabolic and radial states tested directly through the element helper,
  including finite radial semimajor axes and undefined inclination.
- All four time formats, different input-file origins, negative and large dates,
  leap years and the Julian/Gregorian transition.
- Selection, duplicate and absent selections, names and filename collisions,
  existing outputs, line endings and complete records without a final newline.
- 1, 2 and 50 input files, changing populations and remapped codes; 255, 256, 257
  and 513 output files; old capacity boundaries and 100,000 input bodies with a
  small selected subset. Allocating the full encoding limit is not claimed.
- Invalid settings and records, stale body codes, and 256 fixed-seed mutations
  per build profile. These must return a controlled error before creating outputs.
- Files written by all five algorithms, both integration directions and all
  three precision settings, including force-enabled BS/RADAU, HYBRID restarts,
  mergers, central impacts and ejections. CPU/CUDA encounter states are compared
  separately.
- Numerical samples from the 4,100-encounter regression, with all row counts,
  identities and order checked. Host AddressSanitizer/UBSan instruments the
  actual reader, reporting helper and support code.

For ordinary numeric output, acceptance is half the last printed unit plus
64 floating-point units of the reference value. Scientific notation uses its
own printed resolution. Direct helper bounds are `1e-12 * max(1,e)` for
eccentricity, `1e-10` degrees for defined inclination in the direct test states,
and `1e-12` for inverse semimajor axis multiplied by radius. Inclination is marked undefined when all
angular-momentum components lie within a roundoff bound of 16 machine epsilons
times the sum of their absolute products; a resolved small angular momentum
remains defined. Near zero energy, inverse semimajor axis
avoids an ill-conditioned comparison of enormous semimajor axes. The report
records errors with their units. These checks do not promise that discarded
compression digits can be recovered.

`close6` can only decode encounters the integrator recorded. In particular,
MVS cannot resolve close encounters, and HYBRID records encounters handled by
its Bulirsch-Stoer subsystem. Read the
[usage guide](../README_MERCUDA.md#reading-close-encounters) and
[technical notes](technical_notes.md#close-encounter-output) for output behavior.

## Close6 validation on 4 October 2026

At revision [`ceaf09d`](https://github.com/HBJ1004/MERCUDA/commit/ceaf09dabe09840bac2535b90c11effeec037421),
the complete CPU campaign passed **13,773 cases**, with **zero failures and
zero incomplete checks**, using optimized and debug builds. This includes
**785 close6 checks** and the full scientific and capacity groups.

The combined CPU/CUDA campaign completed all **1,579 close6 checks**, using
optimized and debug builds on both backends. Its 20 NVIDIA device-sanitizer
checks and host encounter-buffer sanitizer check also passed. The hardware and
compiler versions were the same as recorded for the 3 October campaign below.

The combined campaign was stopped to avoid competing with the user's GPU work.
Its report contains **26,240 passed cases, zero failures and one interruption**;
the package-wide scientific and stress groups remain incomplete. This is a
completed close6 validation, not a new complete GPU validation of the package.
The separate CPU report is complete. Both reports record the clean source
revision and source hashes; documentation was updated after testing.

The tests exposed and fixed the legacy reader's mass-dependent velocity error,
undefined orbital quantities on radial and parabolic paths, and unsafe handling
of malformed records and capacity boundaries. The integrated force equations
were unchanged. Counts include expected input rejections and named checks of
multiple output rows.

## Validation completed on 3 October 2026

The complete CPU campaign passed **12,988 validation cases** at revision
[`3a6c32d`](https://github.com/HBJ1004/MERCUDA/commit/3a6c32d5ee331fbf6cba4a392fd73d7a67ca4133),
using optimized and debug builds. `make test` also passed all 52 regression tests.
The combined CPU/CUDA campaign passed **25,964 cases**, with **zero failures
and zero incomplete checks**. All 20 NVIDIA device-sanitizer checks passed for
BS, BS2, RADAU, MVS and HYBRID, covering both the lifecycle worker and integration
driver. No memory-access errors, device leaks, uninitialized-memory reads,
synchronization errors or shared-memory race hazards were detected in these
fixtures. The host AddressSanitizer/UBSan check of encounter-buffer packing also
passed. The GPU was an NVIDIA RTX 4070, with driver 591.86 and CUDA toolkit 12.6;
GNU Fortran 13.3 was used for all profiles.

The device checks were initially blocked by the Windows debugger interface.
After enabling it, they passed on 3 October at revision
[`0aa6bc4`](https://github.com/HBJ1004/MERCUDA/commit/0aa6bc47de88c9861cf605b19fbb5a80f23fb1c8).
The numerical source, fixtures and tools were verified unchanged before adding
the follow-up results to the combined report. Its provenance preserves the
original blocked checks and the follow-up campaign.

An environment restart interrupted the combined run. It continued from a
13,400-case checkpoint after verifying the numerical source, fixtures and tools;
this continuation is recorded in the report provenance.

Counts include expected input rejections. In the combined force matrix,
9,280 cases compare trajectories with an independent reference and 11,200 check
that unsupported combinations are rejected.

Tests exposed and fixed a CUDA encounter-buffer overread, exact-parabolic input
and element-conversion errors, singular initial configurations, missing encounter
records when stopping, postprocessing bounds errors, and misleading GPU fallback
logging. The supplied PR formula was preserved.

## Results and limits

Reports, CSV tables and Python plots are written to `results/validation/full/`
or `results/validation/cpu/`. Sanitizer-only results go to
`results/validation/sanitize/`. Results and build products are gitignored.
Simulation directories are temporary and removed after each test. The report
records source hashes, revision, tool versions, hardware, case settings,
measured errors and unavailable checks. Figure hashes and rendering provenance
are in `rendering.json`. The convergence figure shows the worst error at each
refinement level; coarse settings can exceed the final acceptance bound.
Development subsets are always marked
incomplete and cannot substitute for a full campaign.

Passing on one GPU does not certify every driver or GPU architecture. Arbitrary
custom forces, pathological singular orbits and long chaotic integrations
cannot be covered exhaustively. MVS still cannot resolve close encounters, and
symplectic methods require an adequate timestep for eccentric orbits. PR tests
check the preserved supplied prescription, including its component-wise
velocity expression; they do not establish equivalence to another PR model.
Binary-star drivers and fragmentation remain unimplemented. See
[technical notes](technical_notes.md) and the [MERCURY6 manual](../README_MERCURY6.md)
for these method limitations.
