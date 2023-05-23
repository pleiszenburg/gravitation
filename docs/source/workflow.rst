:github_url:

.. _workflow:

Workflow
========

Fundamentals
------------

A typical *gravitation* workflow is command-line oriented. However, it is also possible to import *gravitation* from other Python scripts - see chapter on APIs for details.

Test an individual kernel with :ref:`gravitation view <cli>`. It visualizes a simulation in "real-time", i.e. as fast as time steps are computed.

Run a benchmark across multiple kernels with :ref:`gravitation benchmark <cli>`. It will run :ref:`gravitation worker <cli>` for all possible permutations of its given input parameters. Its results will be stored into a log file.

Transform the log file into a well structured JSON file with :ref:`gravitation analyze <cli>` for further analysis with your favorite tools.

Plot one or more JSON files with :ref:`gravitation plot <cli>` for quick exploration.

Available kernels and the maximum number of available threads will be auto-detected.

The default scenario for benchmarks is "galaxy" (a single, galaxy-like constellation of "stars" with a central "heavy body" loosely resembling a back hole). If you call the benchmark worker script ``gravitation worker`` directly e.g. for testing alternative Python interpreters, the number of point masses can be tuned as follows: ``--len 2000``. Alternatively to ``gravitation worker``, you can also start a worker with ``python -c "from gravitation.cli import cli; cli()" worker``.

Example
-------

.. |benchmark| image:: https://raw.githubusercontent.com/pleiszenburg/gravitation/develop/demo/benchmark.png
    :target: https://gravitation.pleiszenburg.de/
    :alt: screenshot of interactive benchmark plot, click on the image to see interactive version

|benchmark|

The plot above was generated running the commands below on an Intel *i5-4570* CPU with a Nvidia *GeForce GTX 1050 Ti* graphics card. CPython 3.6.7, GCC 8.2.0, Linux 4.15.0 x86_64, Octave 4.2.2, CUDA Toolkit 9.1, Nvidia driver 390.77. CPython, GCC and Linux Kernel are original distribution binaries of Linux Mint 19.1.

.. code:: bash

    # Faster kernels, can work on larger numbers of bodies
    gravitation benchmark -p 4 -i 3 -t 5 -k c1a -k c4a -k cp1 -k cp2 -k cy2 -k cy4 -k js1 -k nb1 -k nb2 -k np1 -k np2 -k np3 -k np4 -k oc1 -k oc4 -k pc1 -k pc2 -k torch1 -k ne1 -b 4 14 -l benchmark_long.log

    # Slower kernels, should work on smaller numbers of bodies
    gravitation benchmark -p 4 -i 3 -t 5 -k py1 -k py2 -k cy1 -b 4 12 -l benchmark_short.log

    # Transform logs
    gravitation analyze -l benchmark_long.log -o benchmark_long.json
    gravitation analyze -l benchmark_short.log -o benchmark_short.json

    # Plot data and display in browser
    gravitation plot -l benchmark_long.json -l benchmark_short.json -o benchmark.html
