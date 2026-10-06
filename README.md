# MERCUDA

A derivative of John E. Chambers' **MERCURY6** with GPU acceleration for BS,
BS2, RADAU, MVS and HYBRID, solar 1PN relativity, radiation pressure and
Poynting–Robertson (PR) drag, and Yarkovsky drift. CPU mode also includes
integration fixes and faster setup for large particle counts.

## Quick start for MERCURY6 users

Keep your usual `.in` files and run the same programs without new command line
options. The additional choices are:

1. **Build:** `make` includes GPU support if it finds NVIDIA's CUDA toolkit
   (`nvcc`). Otherwise it builds a CPU-only version. `make cpu` always builds
   CPU only. Run `make clean-build` before changing build settings. It keeps
   your simulation files.
2. **Enable GPU:** append this line **after all existing settings** in `param.in`:

   ```text
    execution backend = cuda
   ```

   CPU is the default if the line is missing. `auto` chooses a backend from the
   particle count and GPU availability. CUDA needs an NVIDIA GPU, its driver and
   the CUDA toolkit.
3. **Check force settings:** relativity (PN) uses the existing switch in
   `param.in`. Yarkovsky drift uses `yar=<value>` on the body's parameter line in
   `big.in` or `small.in`, in AU/day² at 1 AU (default zero). PR uses `b=<beta>`
   for massless bodies. **Use BS or RADAU for PN, PR or Yarkovsky.** `A1`, `A2`
   and `A3` keep their original cometary meaning.
4. **Run:** `./mercury6`, then `./element6` or `./close6` as usual. Check
   `info.out` to confirm whether CPU or CUDA ran. Custom `mfo_user` forces need CPU.
5. **Restart or rerun:** a restart reads the dynamics from the dump files, and
   only the backend can be changed in `param.in`. To rerun from the inputs, run
   `make rm-gen`, which deletes outputs and dumps, so save your results first.
   **`make clean` also deletes `.in` files.**

## Documentation

- **[MERCUDA guide](README_MERCUDA.md):** a first run, GPU setup, force settings and restarts.
- **[MERCURY6 manual](README_MERCURY6.md):** standard inputs, outputs, postprocessing and restarts, adapted for this package. The unmodified original is in [mercury6.man](mercury6.man), but its build instructions do not apply.
- **[Technical notes](docs/technical_notes.md):** equations, integration changes and the CUDA implementation.
- **[Benchmarks](docs/benchmarks.md):** accuracy, runtime, particle-count and integration-time scaling.
- **[Validation](docs/validation.md):** regression tests, independent accuracy checks and coverage limits.
- **[References](docs/references.md):** integrator and force-model literature.

## Performance

![Runtime and endpoint accuracy](docs/images/speed_and_accuracy.png)

In this stable-orbit example, MERCUDA CUDA is **16–32× faster than original
MERCURY6**, with similar endpoint accuracy. The test used eight planets, 100,000
massless particles and 365 days with extra forces off, on an RTX 4070 compared
with one i5-13600KF CPU thread. The speedup includes CPU improvements, and small
or short runs can be faster on CPU.
[Full results, matched-force comparisons and methodology](docs/benchmarks.md).

## Citation

Please cite [Chambers (1999)](https://doi.org/10.1046/j.1365-8711.1999.02379.x)
and record the MERCUDA version (for example `v1.0.0`) and the force settings you
used. [Additional references](docs/references.md) · [Original MERCURY6 repository](https://github.com/smirik/mercury).

MERCUDA is distributed under [GNU GPL version 3](LICENSE), following the
[upstream MERCURY6 license](https://github.com/smirik/mercury/blob/aee9e0f6b8e4d359a9ed3607ee12e34a2e2dafac/LICENSE).
See [NOTICE](NOTICE) for attribution and modification dates.
