# Benchmarks for emg3d using airspeed velocity

[![asv](http://img.shields.io/badge/benchmarked%20by-asv-blue.svg?style=flat)](http://empymod.github.io/emg3d-asv/)

Currently there are benchmarks for:

   - `solver`: `solver`, `smoothing`, `residual`.
   - `njitted`: None;
   - `utils`: None.

Not too many benchmarks so far. But enough to detect any sever regression. More
importantly, everything is set up and in place. So before trying to improve
speed or memory in any function, write a benchmark for it.

Make sure to set
```
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
OMP_NUM_THREADS=1
```
as outlined in the
[asv-docs](https://asv.readthedocs.io/en/stable/tuning.html#library-settings).

The results which are shown on
[empymod.github.io/emg3d-asv](http://empymod.github.io/emg3d-asv/) are stored in the
[github.com/empymod/emg3d-bench](http://github.com/empymod/emg3d-bench)-repo.
