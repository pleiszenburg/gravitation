:github_url:

.. _wishlist:

Wishlist
========

**Contributions are highly welcome**.

Desired / Planned Kernels
-------------------------

- Faster "pure" Python implementation(s) - "pure" as in "standard-library only"
- Faster "pure" ``numpy`` implementation(s) - "pure" as in "standard-library plus ``numpy`` only"
- Balanced / optimized combinations of ``numpy``, ``numba`` and ``numexpr`` (for individually both, smaller and larger numbers of bodies)
- ``numpy``-implementation via custom ``ufunc``, see `numpy docs`_ and `ufunclab`_, implemented in C
- `JAX`_
- `pythran`_
- `Nuitka`_
- `mypyc`_
- **Rust**, also see `struct.simd`_
- **Go**
- Swift backend(s) - if this is at all possible
- C backend(s) with CUDA (without ``PyCUDA``)
- C backend(s) called through trough different interfaces (instead of ``ctypes``): `cffi`_, etc.
- C++ backend(s) called through different interfaces: ``swig``, ``sip``, ``cython``, etc.
- Faster CUDA backend(s) in general, with or without ``PyCUDA``
- **openCL** backend(s), any language
- ROCr / **ROCm** backend(s), any language
- **Fortran** backend(s)
- **Julia** backend(s)
- **TensorFlow** backend(s), for both CPU and GPU - (theoretically) possible
- JavaScript on ``nodejs``
- Faster JavaScript in general
- Parallel JavaScript with workers
- **Matlab** on original `Matlab`_ interpreter (not Octave)
- Lisp backend(s)
- Parallel backend(s) based on MPI via plain C or `mpi4py`_
- Parallel backend(s) based on `Dask`_
- `Mojo`_, once released

.. _numpy docs: https://numpy.org/doc/stable/user/c-info.ufunc-tutorial.html
.. _ufunclab: https://github.com/WarrenWeckesser/ufunclab
.. _JAX: https://jax.readthedocs.io/en/latest/index.html
.. _pythran: https://github.com/serge-sans-paille/pythran
.. _mypyc: https://github.com/mypyc/mypyc
.. _struct.simd: https://doc.rust-lang.org/std/simd/struct.Simd.html
.. _Dask: https://www.dask.org/
.. _mpi4py: https://mpi4py.readthedocs.io/en/stable/
.. _Matlab: https://www.mathworks.com/help/matlab/matlab-engine-for-python.html
.. _cffi: https://cffi.readthedocs.io/en/latest/
.. _Mojo: https://docs.modular.com/mojo/
.. _Nuitka: https://github.com/Nuitka/Nuitka

Interpreter level
-----------------

- Different versions of ``CPython``
- Different versions of `pypy`_
- `codon`_ - (theoretically) possible

.. _pypy: https://www.pypy.org/
.. _codon: https://github.com/exaloop/codon
