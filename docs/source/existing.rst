:github_url:

.. _existing:

Existing Kernels
================

All of *gravitation*'s kernels reside in the `kernel sub-module`_.

.. _kernel sub-module: https://github.com/pleiszenburg/gravitation/blob/develop/src/gravitation/kernel

- py*: Pure Python, single-thread (compatible with pypy)
- np*: Both single-thread and parallel, based on numpy
- cp*: numpy-compatible implementations using cupy (GPU/CUDA)
- torch*: almost numpy-compatible implementations using torch (GPU/CUDA)
- nb*: accelerated by numba, both CPU and GPU (CUDA), single-thread
- ne*: accelerated by numexpr, single-thread
- pc*: PyCUDA kernels
- cN*: C backends, both single-thread and parallel, both plain C and SIMD (SSE2) intrinsics
- cy*: Cython backends, both plain Python (compiled) and isolated Cython, both single-thread and parallel
- js*: JavaScript backends, currently single-thread and based on py_mini_racer (V8)
- oc*: Octave backends, very likely Matlab-compatible (not yet tested), based on oct2py, both single-thread and parallel
