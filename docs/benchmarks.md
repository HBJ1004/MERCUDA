# Benchmarks

[MERCUDA overview](../README.md) · [Usage guide](../README_MERCUDA.md)

These measurements compare **identical Newtonian physics**, with solar 1PN
relativity (PN), radiation pressure and PR drag, A1/A2/A3 and oblateness
disabled. The example below uses eight planets and 100,000 massless particles
for 365 days. Timings include startup and final output.

![Runtime and endpoint accuracy](images/speed_and_accuracy.png)

| Algorithm | MERCURY6 CPU | MERCUDA CPU | MERCUDA CUDA | MERCURY6 / CUDA | MERCUDA CPU / CUDA | MERCURY6 error (AU) | CUDA error (AU) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BS | 63.7 s | 12.6 s | 3.83 s | 16.63× | 3.28× | 8.27e-11 | 8.22e-11 |
| BS2 | 59.9 s | 8.89 s | 3.57 s | 16.77× | 2.49× | 7.04e-12 | 6.6e-12 |
| RADAU | 71.2 s | 20.1 s | 4.33 s | 16.45× | 4.65× | 5.2e-12 | 5.18e-12 |
| MVS | 121 s | 70.8 s | 3.84 s | 31.58× | 18.42× | 1.31e-10 | 1.31e-10 |
| HYBRID | 63.1 s | 8.25 s | 3.89 s | 16.19× | 2.12× | 5.41e-06 | 5.41e-06 |

Errors are the largest absolute Cartesian endpoint differences from a converged
REBOUND IAS15 reference, over eight planets and up to 64 sampled particles.
CPU and CUDA were also compared over all particles. The plotted cases show
similar accuracy, which is not a guarantee for every orbit or encounter. BS2's
corrected error norm can change the timesteps even at the same requested
tolerance.

The speedup over MERCURY6 includes faster CPU initialization as well as CUDA
acceleration. The MERCUDA CPU / CUDA column separates the two backends more
closely. Small or short runs can be faster on CPU, and the
[particle-count and duration plots](#particle-count-and-duration) show how this
depends on the workload.

For a fair comparison with extra forces, a separate **MERCURY6 + matched forces**
baseline keeps the original integrators and step controllers but uses the same
PN, PR and Yarkovsky equations and input switches as MERCUDA. Original MERCURY6's
PN and PR routines are empty placeholders, so this baseline is needed. With solar
1PN, particle beta=1e-4 and a Yarkovsky coefficient of 1e-12 AU/day² in the same
100,000-particle, 365-day case:

| Algorithm | MERCURY6 + matched forces | MERCUDA CPU | MERCUDA CUDA | Baseline / CUDA | Baseline / CUDA error (AU) |
| --- | ---: | ---: | ---: | ---: | ---: |
| BS | 78.8 s | 24.8 s | 5.47 s | 14.41× | 8.23e-11 / 8.23e-11 |
| RADAU | 90.9 s | 33.4 s | 6.04 s | 15.06× | 5.21e-12 / 5.21e-12 |

The reference for this case uses independent IAS15 stepping with MERCUDA's
validated heliocentric force code. It tests integration accuracy under the same
force model rather than giving an independent implementation of that model. PN,
PR and Yarkovsky are enabled together, so these timings do not isolate the cost
of PN.

**Measurement setup:** RTX 4070 and one pinned i5-13600KF CPU thread, Linux/WSL,
gfortran 13.3.0 and CUDA 12.6. Each value is the median of three fresh runs after
one warmup. Adaptive tolerance 1e-11, fixed timestep one day, high output
precision, output at the final epoch only. The eight-planet system is synthetic.
Particles start at perihelion with random azimuths, a=3.2–3.8 AU and e=0.01–0.05
(seed 1729). The scaling sweeps cover 0–100,000 particles and 32–3,650 days.
All orbits are stable.

MERCUDA was measured on a pre-release build, revision
[7e0084d](https://github.com/HBJ1004/MERCUDA/commit/7e0084de7af16fbaaa1b7d7d4c1cfc6717e668d3).
In that build the Yarkovsky input was named `A2`, which is why the extra-force
figure uses that label. It is the same term as `yar` in this release. The
original [MERCURY6 source](https://github.com/smirik/mercury/tree/aee9e0f6b8e4d359a9ed3607ee12e34a2e2dafac)
was changed only to raise NMAX from 2,000 to 100,010 for the Newtonian runs. The
matched-force baseline also changes MFO_PN, MFO_PR, MFO_NGF and MIO_IN, for the
force equations, PN input and combined Yarkovsky/PR activation. IAS15 tolerances
of 1e-13 and 1e-15 agreed within 2.57e-13 AU in every case. All 156
configurations completed, and repeated runs of each configuration gave identical
endpoints. The largest CPU/CUDA position difference over all bodies was
3.79e-11 AU. Only the final figures are published. The benchmark scripts,
reference builds and simulation files are not part of the package.

## Particle count and duration

The figures below show how total runtime changes with particle count and
duration. Every implementation starts from the same initial states in each
configuration.

![Runtime by particle count](images/runtime_vs_particles.png)

Eight planets and 0–100,000 massless particles, integrated for 32 days.

![Runtime by integration duration](images/runtime_vs_duration.png)

Eight planets and 4,096 massless particles, integrated for 32–3,650 days.

![Runtime and accuracy with identical additional forces](images/matched_forces.png)

100,000 massless particles plus eight planets for 365 days, with the same solar
1PN, beta=1e-4 and Yarkovsky coefficient of 1e-12 AU/day² (labelled `A2` in the
figure). The patched MERCURY6 baseline keeps its original step controllers. The
IAS15 reference shares MERCUDA's validated heliocentric force code, so this
checks integration accuracy rather than deriving the forces independently.

Speed depends on hardware, algorithm, population, encounters and output cadence.
Startup can dominate short runs. These stable-orbit measurements do not show
that the accuracy is equivalent in every case. Exploratory pre-release
benchmarks found MVS accuracy differences in some other configurations, so check
convergence for your own problem.
