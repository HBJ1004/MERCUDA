# MERCUDA

A derivative of John E. Chambers' **MERCURY6** with GPU acceleration for BS,
BS2, RADAU, MVS and HYBRID, solar 1PN corrections, preserved radiation/PR forces,
and Yarkovsky drift. CPU mode also includes performance and integration fixes.

The workflow stays familiar: `make`, settings in `.in` files, and the same
`mercury6`, `element6` and `close6` executables. CUDA requires an NVIDIA GPU and
CUDA toolkit; select `execution backend = cuda` in `param.in` to enable it.
CPU remains the default.

- **[MERCUDA guide](README_MERCUDA.md)** — a first run, GPU setup, force settings and restarts.
- **[Original MERCURY6 README](README_MERCURY6.md)** and **[manual](mercury6.man)** — standard inputs, outputs, postprocessing and restarts.
- **[Technical notes](docs/technical_notes.md)** — equations, integration changes and CUDA implementation.
- **[Benchmarks](docs/benchmarks.md)** — accuracy, runtime, particle-count and integration-time scaling.
- **[References](docs/references.md)** — integrator and force-model literature.

## Performance

![Runtime and endpoint accuracy](docs/images/speed_and_accuracy.png)

In this stable-orbit example, MERCUDA CUDA is **16–32× faster than original
MERCURY6**, with similar endpoint accuracy: eight planets, 100,000 massless
particles, 365 days, extra forces off, RTX 4070 versus one i5-13600KF CPU thread.
The speedup includes CPU improvements; small or short runs can favor CPU.
[Full results, matched-force comparisons and methodology](docs/benchmarks.md).

## Citation

Please cite [Chambers (1999)](https://doi.org/10.1046/j.1365-8711.1999.02379.x)
and record the MERCUDA revision and force settings used.
[Additional references](docs/references.md) · [Original MERCURY6 repository](https://github.com/smirik/mercury).
