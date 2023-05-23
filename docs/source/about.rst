:github_url:

.. _about:

About ``gravitation``
=====================

.. _synopsis:

Synopsis
--------

In science and engineering, it is a prominent scenario to use Python as a high level or "glue" language on top of code written C, Fortran or other "fast" languages. In some cases, a Python project starts out as a Python wrapper around other, possibly older non-Python code. In other cases, functionality is rapidly prototyped in Python. Eventually, performance-critical code-paths are identified, isolated and optimized - or even re-implemented in a second, "faster" language. Either way, there is a great diversity of possible approaches and tools for accelerating Python code and/or combining it with other languages. Depending on a project's requirements, it can be hard to choose the right one(s). Virtually all are based on at least one more programming language other than Python, in one way or the other - hence the "two language problem". *gravitation* aims to demonstrate a broad selection of approaches and tools. Both their performance and level of implementation complexity are (cautiously) compared based on a (single) **typical computation-bound problem: n-body simulations**. n-body simulations scale up easily, which in return gives invaluable insights into the scaling behaviors of all demonstrated approaches. They are also fairly trivial targets for parallelization, allowing to study their behaviors on hardware platforms with multiple CPU cores (both heterogeneous and homogeneous), multiple CPU sockets and GPUs or comparable numerical accelerators.

.. _implementation:

Implementation
--------------

The design idea is simple: A number of n-body simulation implementations, referred to as **kernels**, are derived from a single class, :class:`gravitation.UniverseBase`, written in pure Python. It provides the `fundamental Python infrastructure`_ while it can not (and should not) perform a simulation on its own.

.. _fundamental Python infrastructure: https://github.com/pleiszenburg/gravitation/blob/develop/src/gravitation/kernel/_base.py

*gravitation* kernels divide each simulation time step into two major stages, updating certain parameters of all bodies:

1) Acceleration, i.e. the forces between all pairs of bodies: :math:`O(N^2)`
2) Velocity and position: :math:`O(N)`

Because of its :math:`O(N^2)` complexity, **stage 1 is the prime target for optimizations and/or re-implementations**. Therefore, the base class offers a default implementation for stage 2 only. A kernel class derived from the base class must (at least) implement stage 1 on its own. Overloading stage 2 is possible but not required. The `py1 kernel`_ serves as a minimal working reference kernel, implementing stage 1 in pure Python without dependencies.

.. _py1 kernel: https://github.com/pleiszenburg/gravitation/blob/develop/src/gravitation/kernel/py1.py

Kernels must not change the "interface for frond-end code" exposed by the base class. "Python users" or other pieces of Python software must be able to run all kernels transparently without knowing about any of the intricate details of their implementations. Both position and velocity data of every body (also referred to as "point mass") must be accessible after every time step through iterable objects (e.g. Python lists, arrays or numpy arrays). For reference, have a look at the :class:`gravitation.PointMass` class. Kernels must be located in `src/gravitation/kernel`_ for being auto-detected.

.. _PointMass class: https://github.com/pleiszenburg/gravitation/blob/develop/src/gravitation/kernel/_base.py
.. _src/gravitation/kernel: https://github.com/pleiszenburg/gravitation/blob/develop/src/gravitation/kernel

Certain kernels allow to switch between single precision floating point numbers and double precision floating point numbers, others do not (due to language- or instruction-level restrictions for instance).

While kernels compute time steps, Python's `garbage collector`_ remains switched off. This allows clean results not affected by "randomly occurring" garbage collections. Directly before and after every time step, a garbage collection is triggered "manually". The time required for collecting garbage after a time step has been computed is also (separately) measured and recorded.

.. _garbage collector: https://docs.python.org/3/library/gc.html
