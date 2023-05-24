:github_url:

.. _sysreq:

System Requirements
===================

*gravitation* was developed for and tested on Linux x86 64bit (AMD64). However, there is no reason why it should not work on other operating systems (Windows, Mac OS, BSD, Solaris, etc.) or other platforms (ARM, Power, etc.). The basic benchmark infrastructure *should* be platform independent. Certain kernels might require a few tweaks (e.g. alternatives to using ``/dev/shm`` for "on-disk" caching or inter-process communication via files). Kernels depending on certain x86-specific features will of cause not work on other platforms. A CUDA-compatible accelerator is highly recommended, although without it only kernels depending on CUDA will not work. Other kernels are not affected. There are no pre-compiled binaries at this point (although this may change in future).

Installation is supported through ``pip``. Support for ``conda`` is likely going to be added.

**TODO** pip vs conda setup

Hardware Prerequisites
----------------------

- `AMD64`_-compatible CPU, ideally with SSE4 and AVX2, see `SIMD`_
- `CUDA`_-compatible accelerator card

.. _AMD64: https://en.wikipedia.org/wiki/X86-64
.. _SIMD: https://en.wikipedia.org/wiki/SIMD
.. _CUDA: https://en.wikipedia.org/wiki/CUDA

Operating System & Infrastructure Prerequisites
-----------------------------------------------

- `Linux x86 64bit`_
- `CUDA Toolkit`_
- C-compiler: `GCC`_ or `Clang/LLVM`_ (consult your Linux distribution's documentation for details)
- `openMP`_ headers (consult your Linux distribution's as well as your compiler's documentation for details)
- `gnuplot`_
- `Octave`_

.. _Linux x86 64bit: https://distrochooser.de/en/
.. _CUDA Toolkit: https://developer.nvidia.com/cuda-downloads?target_os=Linux
.. _GCC: https://en.wikipedia.org/wiki/GNU_Compiler_Collection
.. _Clang/LLVM: https://en.wikipedia.org/wiki/Clang
.. _openMP: https://en.wikipedia.org/wiki/OpenMP
.. _gnuplot: http://www.gnuplot.info/
.. _Octave: https://www.gnu.org/software/octave/download.html

Python Prerequisites
--------------------

- `CPython`_ 3.9 or later - likely part of your Linux distribution (consult its documentation for details). Older versions are supported on a best-effort basis.
- `PyCUDA`_
- `PyTorch`_
- `CuPy`_

.. _CPython: https://www.python.org/downloads/
.. _PyCUDA: https://wiki.tiker.net/PyCuda/Installation/Linux
.. _PyTorch: https://pytorch.org/get-started/locally/
.. _CuPy: http://docs-cupy.chainer.org/en/stable/install.html

Installing all of the above into a `virtual environment`_ is highly recommended.

.. _virtual environment: https://docs.python.org/3/library/venv.html
