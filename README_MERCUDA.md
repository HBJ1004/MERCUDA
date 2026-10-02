# Using MERCUDA

MERCUDA is based on John E. Chambers' MERCURY6. It adds GPU support, relativity,
Poynting–Robertson (PR) drag and Yarkovsky drift, while keeping the usual input
files and programs: `mercury6`, `element6` and `close6`.

This is a user guide. For the standard file formats and settings, read the
[original MERCURY6 README](README_MERCURY6.md) or [manual](mercury6.man).
For equations, numerical changes and implementation details, read the
[technical notes](docs/technical_notes.md).

## What changes for existing MERCURY6 users?

You still build with `make`, edit `.in` files, and run `./mercury6` without command
line options. CPU is the default. To use the GPU, build with CUDA support and add
`execution backend = cuda` at the end of `param.in`.

MERCUDA also changes CPU runs: it improves speed, removes the fixed 2,000-body
limit, and corrects several integration problems, including backward integration.
It is therefore not identical to the original MERCURY6, even on CPU.

**A2 now means Yarkovsky drift, not cometary transverse outgassing.** Review old
A2 values before using existing inputs. Old restart files with A2 or relativity
enabled cannot be continued under the new model; start from initial conditions.
[Details of these changes](docs/technical_notes.md#changes-from-mercury6).

## Your first run: step by step

This example uses the supplied planets and asteroids for a short CPU run.
Linux or WSL is assumed. You need Git, GNU Make, gfortran and a C++ compiler such
as g++. A GPU is optional. On Ubuntu, you can install these tools with
`sudo apt install git make gfortran g++`.

### 1. Download the package

Open a terminal and enter:

```sh
git clone https://github.com/HBJ1004/MERCUDA.git
cd MERCUDA
```

If you already have the package, open a terminal in that folder instead.
Run the following commands from the package folder.

### 2. Build the programs and create input files

```sh
make cpu
make gen-in
```

The first command builds the three programs without requiring CUDA. The second
copies the sample inputs into `.in` files. It keeps any input files already present;
use a fresh folder for this example.

| File | What it contains |
| --- | --- |
| `big.in` | Planets and other big bodies. |
| `small.in` | Asteroids, comets and other small bodies. |
| `param.in` | Integration method, times, accuracy and other settings. |
| `files.in` | Names of the integration's input and output files. |
| `element.in` | Settings for producing orbital element tables. |
| `close.in` | Settings for reading close encounters. |
| `message.in` | Messages used by the programs; leave this file as supplied. |

Keep the supplied `big.in` and `small.in` for this example. For your own system,
follow [section 3 of the original manual](mercury6.man).

### 3. Set a short integration

Open `param.in` in a text editor. Replace the existing values of these settings:

```text
 algorithm (MVS, BS, BS2, RADAU, HYBRID etc) = bs
 start time (days) = 2458400.5d0
 stop time (days) = 2458765.5d0
 output interval (days) = 30.d0
 timestep (days) = 1.d0
 accuracy parameter = 1.d-11
```

This runs for 365 days and requests output every 30 days. Keep relativity and
user-defined forces set to `no`, and leave the other sample settings unchanged.
Do not remove the file's first line or change the order of the existing settings.
The sample bodies have different starting dates; the program brings them to the
integration's start date before beginning this run.

### 4. Run the integration

```sh
./mercury6
```

When it finishes, read `info.out` for any problems. It should report
`Execution backend: CPU`. The orbital data are in `xv.out`, close-encounter data
in `ce.out`, and restart information in the `.dmp` files.

### 5. Produce readable orbital elements

Open `element.in` and change `minimum interval between outputs (days)` to `30.d0`.
Then run:

```sh
./element6
```

This produces `.aei` files for the selected bodies. Open one in a text editor to
see its orbital elements. To examine close encounters, run `./close6`; encounters
recorded during the run are written to `.clo` files. See
[manual sections 4–5](mercury6.man) for selecting bodies and output columns.

## Using the GPU

You need an NVIDIA GPU, its driver, and a CUDA toolkit containing `nvcc`.
A GPU or driver alone is not enough. To rebuild the CPU example with GPU support:

```sh
make clean-build
make
```

`make clean-build` keeps your inputs and simulation results. `make` builds CUDA
support when it finds the toolkit; otherwise it builds CPU support. If your
toolkit is not found, see [build options](docs/technical_notes.md#build-options).

Append this line after all existing settings in `param.in`:

```text
 execution backend = cuda
```

Run `./mercury6` as usual and check `info.out` for `Execution backend: CUDA`.
Existing dumps continue the previous run; to repeat the first-run example from
its inputs, save any results you need and run `make rm-gen` before `./mercury6`.

| Setting | What it does |
| --- | --- |
| `cpu` | Uses CPU. This is the default if the setting is omitted. |
| `cuda` | Uses GPU. Stops with an error if CUDA or the chosen configuration is unavailable. |
| `auto` | Chooses GPU for supported runs with at least 4,096 small bodies; otherwise uses CPU. It can fall back to CPU if GPU setup fails. |

Small systems and short runs may be faster on CPU. `auto` is a convenient rule,
not a guarantee of the fastest choice. See the [benchmarks](docs/benchmarks.md).

## Choosing an integration method

Set the method in the existing algorithm line of `param.in`.
All five methods below can run on CPU or GPU.

| Method | Supported forces and limits |
| --- | --- |
| BS | Supports all built-in forces, including velocity-dependent and dissipative terms. |
| RADAU | Supports all built-in forces, including velocity-dependent and dissipative terms. |
| BS2 | Use for gravity, including central-body oblateness. PN, PR and A1/A2/A3 are not supported. |
| MVS | PN, PR and Yarkovsky are not supported. Small bodies must have zero mass. |
| HYBRID | PN, PR and Yarkovsky are not supported. |

For the purpose and usual settings of each method, see the
[original manual](mercury6.man). Custom `mfo_user` forces require CPU. The close-
and wide-binary methods have no working drivers in this distribution.
[Full support details](docs/technical_notes.md#algorithm-support).

## Adding forces

**Use BS or RADAU for PN, PR or Yarkovsky drift.**

| Force | How to enable it |
| --- | --- |
| Relativity (PN) | Change the existing `include relativity in integration` setting in `param.in` to `yes`. |
| Radiation pressure / PR | Add `b=<beta>` on the body's name/parameter line. It applies only to bodies with zero mass. The supplied prescription is preserved, based on Burns et al. (1979), Liou et al. (1995) and Klačka et al. (2012); see the [references](docs/references.md). |
| Yarkovsky drift | Add `A2=<value>` on the body's name/parameter line in `big.in` or `small.in`. It defaults to zero. |

For example, a massive asteroid's name/parameter line can be:

```text
ASTEROID m=1.0d-15 r=1.0d0 d=2.5d0 A2=-3.0d-14
```

Keep its following position/elements, velocity and spin lines in the usual format.
A2 is measured in AU/day² at 1 AU. Positive A2 acts along orbital motion;
negative A2 acts against it. It applies to both massive and massless bodies.
A2 does not also supply cometary transverse outgassing; A1 and A3 keep their
original cometary meaning. Beta is dimensionless and defaults to zero.

These are simplified force models. The [technical notes](docs/technical_notes.md#forces-and-body-input)
give their equations and limits, and the [references](docs/references.md) identify
the literature they are based on.

## Backward runs, restarts and cleanup

For backward integration, set the stop time earlier than the start time.
With PR or Yarkovsky, reversing velocities is not a substitute for integrating
backward. MVS and HYBRID write output on their integration timestep grid;
a smaller output interval does not create intermediate states.

Continue a run using its dump files as described in
[manual sections 6–7](mercury6.man). Changing ordinary `.in` files does not change
the saved run's dynamics. You can change `execution backend` in `param.in` to
switch between CPU and GPU when continuing a MERCUDA run. For old dumps, see the
[restart compatibility notes](docs/technical_notes.md#backward-integration-and-restarts).

| Command | What it removes |
| --- | --- |
| `make clean-build` | Compiled files and programs; keeps inputs and simulation results. |
| `make rm-gen` | Results and restart files: `.aei`, `.clo`, `.out`, `.dmp`, `.tmp`; keeps inputs and programs. |
| `make clean` | Compiled files, programs, simulation results, restart files **and `.in` inputs**. |

Save any needed results and restart files before deleting them.

## More information

- [Original MERCURY6 README](README_MERCURY6.md) and [manual](mercury6.man): standard input formats, settings and output instructions.
- [Technical notes](docs/technical_notes.md): equations, numerical changes, GPU implementation and tests.
- [Benchmarks](docs/benchmarks.md): speed and accuracy comparisons.
- [References](docs/references.md): papers to consult and cite.
