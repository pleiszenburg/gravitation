
# Infrastructure

* Support for `conda`
* Add tests (pytest is desired)

# Benchmark dimensions

* half, single and double precision floating point (where possible)
* Python interpreters for benchmark workers (where applicable) - CPython 3.x, pypy, different compiler versions, different compile-time optimizations
* C- / Fortan-compilers - clang/LLWM, GCC >= 4.x, Intel, ...

# Kernels

See "Desired Kernels" in [README.md](https://github.com/pleiszenburg/gravitation/blob/master/README.md) for further ideas. Prepared but not yet integrated code:

* C with SSE2 intrinsics single-thread
* Plain C multi-thread openMP
* Visual Basic / VBA

# Kernel Design & Implementation Issues

* make `gravitation worker` pypy-compatible (again ...)
* np4: improve joblib worker initialization
* oc4: improve joblib worker initialization
* C code: translate comments into English
* pc3: figure out what goes wrong (bodies keep "disappearing")
* gravitation.lib.simulation: translate comments into English
* gravitation.lib.simulation.create_galaxy: test against original C implementation
* np3, np4 and other parallel kernels: code for computing batch size is redundant and broken for small numbers of bodies ("line index tuples for evenly sized batches")
