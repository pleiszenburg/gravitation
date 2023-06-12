# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/cli/benchmark.py: benchmark command

    Copyright (C) 2019-2023 Sebastian M. Ernst <ernst@pleiszenburg.de>

<LICENSE_BLOCK>
The contents of this file are subject to the GNU General Public License
Version 2 ("GPL" or "License"). You may not use this file except in
compliance with the License. You may obtain a copy of the License at
https://www.gnu.org/licenses/old-licenses/gpl-2.0.txt
https://github.com/pleiszenburg/gravitation/blob/master/LICENSE

Software distributed under the License is distributed on an "AS IS" basis,
WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License for the
specific language governing rights and limitations under the License.
</LICENSE_BLOCK>

"""

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# IMPORT
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import click

from ..lib.benchmark import Benchmark
from ..lib.const import Display
from ..lib.kernel import KERNELS

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ROUTINES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


@click.command(short_help="run a benchmark across kernels")
@click.option(
    "--logfile",
    "-l",
    default="benchmark.log",
    type=str,
    show_default=True,
    help="name of log file",
)
@click.option(
    "--datafile",
    "-d",
    default="data.h5",
    type=str,
    show_default=True,
    help="name of data file",
)
@click.option(
    "--common_initial_state",
    "-c",
    default=False,
    is_flag=True,
    show_default=True,
    help="use common initial state per length for all kernels",
)
@click.option(
    "--all_kernels",
    "-a",
    is_flag=True,
    default=False,
    show_default=True,
    help="run all kernels",
)
@click.option(
    "--len_range",
    "-b",
    default=(2, 16),
    type=(int, int),
    show_default=True,
    help="2^x bodies in simulation, for x from lower to upper boundary",
)
@click.option(
    "--save_after_iteration",
    "-s",
    type=int,
    show_default=True,
    multiple=True,
    help="save model universe into file iteration x, -1 if nothing should be saved",
)
@click.option(
    "--min_iterations",
    "-i",
    default=10,
    type=int,
    show_default=True,
    help="minimum number of simulation steps",
)
@click.option(
    "--min_total_runtime",
    "-t",
    default=10,
    type=int,
    show_default=True,
    help="minimal total runtime of (all) steps, in seconds",
)
@click.option(
    "--display",
    default="plot",
    type=click.Choice([item.name for item in Display]),
    show_default=True,
    help="what to show during benchmark",
)
@click.argument(
    "kernel",
    type=click.Choice(sorted(list(KERNELS.keys()))),
    nargs=-1,
)
def benchmark(
    logfile,
    datafile,
    common_initial_state,
    all_kernels,
    len_range,
    save_after_iteration,
    min_iterations,
    min_total_runtime,
    display,
    kernel,
):
    """run a benchmark across kernels"""

    Benchmark.run(
        logfile = logfile,
        datafile = datafile,
        common_initial_state = common_initial_state,
        kernels = sorted(list(KERNELS.keys())) if all_kernels else list(kernel),
        sq_range_start = len_range[0],
        sq_range_stop = len_range[1],
        save_after_iteration = save_after_iteration,
        min_iterations = min_iterations,
        min_total_runtime = min_total_runtime,
        display = Display[display],
    )
