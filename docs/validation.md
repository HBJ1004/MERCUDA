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

On Windows/WSL, Compute Sanitizer may require enabling NVIDIA's Windows debugger
interface. A blocked sanitizer check is recorded as **incomplete**, never as a
successful check. See [NVIDIA's documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html).

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

## Results and limits

Reports, CSV tables and Python plots are written to `results/validation/full/`
or `results/validation/cpu/`. Sanitizer-only results go to
`results/validation/sanitize/`. Results and build products are gitignored.
Simulation directories are temporary and removed after each test. The report
records source hashes, revision, tool versions, hardware, case settings,
measured errors and unavailable checks. Development subsets are always marked
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
