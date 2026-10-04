# Original MERCURY6 encounter fixture

`upstream.ce` was written by unmodified MERCURY6 at
[smirik/mercury aee9e0f](https://github.com/smirik/mercury/tree/aee9e0f6b8e4d359a9ed3607ee12e34a2e2dafac),
built with gfortran 13.3 (`-O3 -ffixed-line-length-none -fallow-argument-mismatch`).
It contains a Newtonian BS flyby: central mass 1 solar mass, PLANET mass 1e-12,
PLANET at (1,0,0) AU with circular velocity and r=10000 Hill radii; PARTICLE
at (1.001,-0.005,0) AU with velocity (0,0.1,0) AU/day. Epoch/start 0 days,
stop 0.1 days, initial step 0.001 days, tolerance 1e-12, high output precision,
central radius 0.005 AU, ejection distance 100 AU, collisions and extra forces off.

This checks compatibility with the original binary writer. Expected orbital
values are calculated independently from the stored bytes; the original
postprocessor is not the numerical oracle. The source and raw simulation
artifacts are not retained in the package.
