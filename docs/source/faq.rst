:github_url:

.. _faq:

FAQ
===

Does actual physics matter?
---------------------------

No. *gravitation* is certainly not about physics.

Does numerical accuracy matter?
-------------------------------

Yes and no. In certain applications, numerical accuracy is more desirable than speed. In other cases, it is the opposite or somewhere in between. Studying the impact of various trade-offs with respect to both speed and accuracy is therefore highly interesting.

What about optimizations such as e.g. tree methods, for instance `Barnes–Hut`_?
-------------------------------------------------------------------------------

This is not what *gravitation* is about. *gravitation* is intentionally written as a direct n-body simulation where forces are computed for all pairs of bodies (in time steps of equal length).

Why is there even a JavaScript kernel?
--------------------------------------

At first, it seemed like a crazy experiment. But after some initial tests with `V8`_ and `SpiderMonkey`_, it became obvious that JavaScript engines had come a long way. The results were simply impressive. Why should one use it? Well, the basic argument is that `JavaScript currently is the most widely used programming language in existence`_, for better or for worse. JavaScript development skills are therefore relatively easy to get hold of. There are even books about how to use it for research projects including numerical computations, e.g. `JavaScript for Data Science`_ aka. "JavaScript versus Data Science".

Why is ``cffi`` on the :ref:`wishlist <wishlist>` if there are already kernels using ``ctypes``?
------------------------------------------------------------------------------------------------

From a functional point of view, ``cffi`` and ``ctypes`` do not differ much. However, they differ in both code complexity and performance. The differences when scaling up are highly interesting.

What about different compilers and compiler versions?
-----------------------------------------------------

This is yet another interesting dimension that is intended to be added to the benchmark infrastructure. The project's C code already shows significant differences in performance if compiled with e.g. GCC 4 or 6 or clang/LLVM.

Why are the ``numpy`` implementations so (relatively) slow?
-----------------------------------------------------------

All implementations are optimized for ``O(N)`` memory usage which implies at least one rather slow Python ``for``-loop and relatively small arrays for ``numpy`` to perform its operations on.

What about Intel Compilers, MKL and MKL-enabled ``numpy``?
----------------------------------------------------------

As far as testing went, there is no significant difference between "regular" and ``MKL``-enabled ``numpy``. *gravitation* is about plain number crunching and does not use any type of "higher" algebra that has been optimized in ``MKL``. The Intel C compiler on its own does seem to make a difference, however.

Are contributions limited to what is listed on the :ref:`wishlist <wishlist>`?
------------------------------------------------------------------------------

No, not at all. Anything that works and adds a new facet to this project is truly welcome.

What about scaling up on computer clusters / super computers?
-------------------------------------------------------------

Although this has yet not been within the scope of this project, contributions are nevertheless welcome.

.. _Barnes–Hut: https://doi.org/10.1038%2F324446a0
.. _V8: https://en.wikipedia.org/wiki/V8_(JavaScript_engine)
.. _SpiderMonkey: https://en.wikipedia.org/wiki/SpiderMonkey
.. _JavaScript currently is the most widely used programming language in existence: https://insights.stackoverflow.com/survey/2021#most-popular-technologies-language-prof
.. _JavaScript for Data Science: https://third-bit.com/js4ds/
