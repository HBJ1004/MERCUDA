# Testing MERCUDA

The validation suite checks the behavior promised by the user guides. It checks
CPU and CUDA calculations against independent references, and it checks that
unsupported combinations stop with an error. It does not prove that every
possible orbit or user-written force is correct.

## Results for v1.0.0

The full CPU/CUDA campaign was run on the release candidate, revision
[`8564487`](https://github.com/HBJ1004/MERCUDA/commit/8564487e0ef38042c240e1bf73ab3f9da64fb7ab).
It passed **27,555 cases** with no failures and no incomplete checks. These
included 1,591 `close6` checks, 1,168 scientific checks and 324 capacity checks
(up to 100,000 particles and 257 massive bodies). All 20 NVIDIA Compute
Sanitizer checks passed for the five algorithms. They found no memory-access
errors, uninitialized reads, synchronization errors or shared-memory races.
Counts include inputs that are expected to be rejected.

Two small postprocessing changes were made after that run: `element6` row
selection for output intervals that are inexact in binary, and the reading of
`close.in` answers and selection lines. Both are covered by the regression
suites, which pass all 70 tests in optimized and debug CUDA builds with no skips.

Hardware and tools: NVIDIA GeForce RTX 4070 (12 GB), driver 591.86, CUDA
toolkit 12.6.85 and GNU Fortran 13.3. Passing on one GPU does not certify other
drivers or architectures. See [results and limits](#results-and-limits).

## Routine checks

From the repository directory:

```sh
make test
make test-debug
```

These run the regression tests using the chosen build. A CPU-only build skips
the GPU-specific checks. GitHub Actions runs the CPU regression suite in
optimized and debug builds.

## Full validation

The full campaign builds its own optimized and debug CPU and CUDA programs in
`build/validation/`. It does not overwrite the programs or simulation files in
the repository directory. It needs a few Python packages:

```sh
python3 -m venv build/validation/venv
build/validation/venv/bin/pip install -r tests/validation/requirements.txt
make VALIDATION_PYTHON=build/validation/venv/bin/python test-full
```

Complete GPU validation needs a working NVIDIA GPU, the CUDA toolkit and NVIDIA
Compute Sanitizer. Set `COMPUTE_SANITIZER=/path/to/compute-sanitizer` if it is
not on your path. Without GPU hardware, use:

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

This is a Windows setting, and the validation runner does not change it.

The full campaign can take hours. A nonzero exit status means the campaign
failed or is incomplete. A CPU-only pass covers only the CPU programs. Debug
builds enable Fortran bounds checks and traps for invalid arithmetic, division
by zero and overflow.

## What is checked

| Area | Checks |
| --- | --- |
| Build and use | Separate build profiles, the sample-input workflow, postprocessing and cleanup commands |
| Forces | All 512 on/off combinations of PN, PR, Yarkovsky, cometary A1/A2/A3 and J2/J4/J6 |
| Algorithms | BS, BS2, RADAU, MVS and HYBRID, including rejection of unsupported force combinations |
| Accuracy | Analytic Kepler orbits and independent IAS15 trajectories, in both time directions and at several epochs |
| Convergence | Three tolerances or timesteps, with reference calculations at two IAS15 tolerances |
| Epochs | Initial epochs before and after the start time, mixed small-body epochs, Julian-date epochs and output intervals |
| Restarts | MERCUDA dumps, dumps from original MERCURY6, incompatible dumps, backend changes, and which settings come from inputs or dumps |
| Inputs and outputs | Cartesian, Asteroidal and Cometary inputs, postprocessing frames, precision, time units and body selection |
| Events | Close encounters, stopping on an encounter, central impacts, ejection and mergers |
| Capacity | Empty systems, thread and block boundaries, the auto-backend threshold, 100,000 particles and 257 massive bodies |
| Errors | Missing files, invalid settings, nonfinite values, invalid orbital elements and undefined force directions |
| Custom forces | Representative time-dependent and velocity-dependent CPU forces, and their rejection on CUDA |
| Memory | Device lifecycle and encounter-buffer growth, host AddressSanitizer/UBSan, and NVIDIA Compute Sanitizer on a lifecycle test program and the real integration programs |

The list of checked claims is in [tests/validation/catalog.json](../tests/validation/catalog.json).
The exact fixtures and settings are recorded in the machine-readable report.
The force reference is coded separately from MERCUDA and evaluated with
80-digit arithmetic. It differentiates a Legendre potential for oblateness.
The trajectory reference uses [REBOUND IAS15](https://rebound.hanno-rein.de/integrators/ias15/)
with separately written callbacks for the extra forces. In `param.in`, J2/J4/J6
are dimensionless, and MERCUDA multiplies them by the central radius to the
appropriate power before evaluating accelerations.

The acceptance bound for stable orbits is `1e-8`, for position divided by the
characteristic distance and for velocity divided by the corresponding Kepler
speed. The bound for encounters is `1e-6`. Force errors must be at most `1e-12`
times the sum of the absolute contributing terms, plus `1e-30` in AU/day². The
reference uncertainty must be below one tenth of the trajectory bound. Refining
a setting must improve the error by at least a factor of two, unless the error
is already within ten times the reference or roundoff floor. These bounds are
fixed in the catalog. They are test acceptance criteria, not a universal
accuracy guarantee.

## Close6 validation

The `close6` checks compare the actual numbers in `.clo` files, not just whether
the program finishes. The reference uses an independent integer base-224 decoder
and 80-digit decimal orbital invariants. A small file written by pinned original
MERCURY6 checks compatibility with its writer. The reference starts from the
compressed state, so compression error is kept separate from postprocessor
error. Encounter states use four base-224 digits whatever the integration output
precision setting.

Coverage includes:

- Circular, eccentric, near-parabolic, hyperbolic, radial and zero-velocity
  states, different inclinations and masses, and 256 fixed-seed random states.
- Exact parabolic and radial states tested directly through the element helper,
  including finite radial semimajor axes and undefined inclination.
- All four time formats, different input-file origins, negative and large dates,
  leap years and the Julian/Gregorian transition.
- Selections, including duplicate and absent ones, names, filename collisions,
  existing outputs, line endings and complete records without a final newline.
- 1, 2 and 50 input files with changing populations and remapped codes, 255,
  256, 257 and 513 output files, the original capacity boundaries, and 100,000
  input bodies with a small selected subset. Allocating the full encoding limit
  is not claimed.
- Invalid settings and records, stale body codes, and 256 fixed-seed mutations
  per build profile. Each must stop with a controlled error before any output is
  created.
- Files written by all five algorithms, in both integration directions and with
  all three precision settings. These include force-enabled BS and RADAU, HYBRID
  restarts, mergers, central impacts and ejections. CPU and CUDA encounter states
  are compared separately.
- Numerical samples from the 4,100-encounter regression, with all row counts,
  identities and order checked. Host AddressSanitizer/UBSan instruments the
  actual reader, the reporting helper and the support code.

For ordinary numeric output, the acceptance bound is half the last printed unit
plus 64 floating-point units of the reference value. Scientific notation uses
its own printed resolution. Direct checks of the element helper use
`1e-12 * max(1,e)` for eccentricity, `1e-10` degrees for defined inclinations,
and `1e-12` for inverse semimajor axis times radius. Inclination is marked
undefined when all angular-momentum components lie within a roundoff bound of
16 machine epsilons times the sum of their absolute products. A small angular
momentum that is resolved stays defined. Near zero energy, the inverse semimajor
axis is compared instead of the enormous semimajor axis itself. The report
records errors with their units. These checks cannot recover digits that were
discarded by compression.

`close6` can only decode encounters that the integrator recorded. MVS cannot
resolve close encounters, and HYBRID records only the encounters handled by its
Bulirsch-Stoer subsystem. See the
[usage guide](../README_MERCUDA.md#reading-close-encounters) and the
[technical notes](technical_notes.md#close-encounter-output) for output behavior.

## Results and limits

Reports, CSV tables and Python plots are written to `results/validation/full/`
or `results/validation/cpu/`. Sanitizer-only results go to
`results/validation/sanitize/`. Results and build products are ignored by Git.
Simulation directories are temporary and are removed after each test. The report
records source hashes, the revision, tool versions, hardware, case settings,
measured errors and any checks that could not run. Figure hashes and rendering
details are in `rendering.json`. The convergence figure shows the worst error at
each refinement level, and coarse settings can exceed the final acceptance
bound. Partial runs are always marked incomplete and cannot replace a full
campaign.

Passing on one GPU does not certify every driver or GPU architecture, and a
passing campaign does not rule out undiscovered defects. Arbitrary custom forces,
pathological singular orbits and long chaotic integrations cannot be covered
exhaustively. MVS cannot resolve close encounters, and the symplectic methods
need an adequate timestep for eccentric orbits. The PR tests check MERCUDA's
radiation-pressure and PR expression, including its component-wise velocity
term. They do not show equivalence to any other PR model. Binary-star drivers
and fragmentation are not implemented. See the
[technical notes](technical_notes.md) and the [MERCURY6 manual](../README_MERCURY6.md)
for these method limitations.
