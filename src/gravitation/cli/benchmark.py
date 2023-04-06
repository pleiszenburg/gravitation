# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

	src/gravitation/cli/benchmark.py: benchmark command

	Copyright (C) 2019 Sebastian M. Ernst <ernst@pleiszenburg.de>

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

import atexit
import json
import math
import shutil

import termplotlib as tpl
import click
import psutil

from ..lib import proc
from ..lib.load import inventory
from .worker import worker_command

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CONST
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

MAX_TREADS = psutil.cpu_count(logical=True)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ROUTINES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def _process_data(kernel, threads, bodies, results_dict, outputlines_list, fh, display):
    """factory, returning function for reading a worker log in realtime"""

    def callback(stream_id, msg_line):
        fh.write(msg_line + "\n")
        if display == "log":
            print(msg_line)
        outputlines_list.append(msg_line)
        try:
            msg = json.loads(msg_line)
        except:
            return
        results_kernel_dict = results_dict[kernel][threads]
        if msg["log"] == "BEST_TIME":
            if bodies not in results_kernel_dict.keys():
                results_kernel_dict[bodies] = msg["value"]
            elif results_kernel_dict[bodies] != msg["value"]:
                results_kernel_dict[bodies] = msg["value"]
            else:
                return
        else:
            return
        if display != "plot":
            return
        x = sorted(list(results_kernel_dict.keys()))
        y = [results_kernel_dict[n] for n in x]
        t = shutil.get_terminal_size((80, 20))
        fig = tpl.figure()
        fig.plot(
            x,
            y,
            label="{kernel}@{threads}".format(kernel=kernel, threads=threads),
            width=t.columns,
            height=t.lines,
            extra_gnuplot_arguments=[
                "set logscale x 2",
                'set format y "10^{%L}"',
                "set logscale y 10",
            ],
        )
        fig.show()

    return callback


def _range(start, end):
    """special range generator, going from 2^start to 2^end with some interpolation"""
    l = [2**i for i in range(start, end + 1)]
    l_intp = [2**start]
    if end > start:
        l_intp.append(((2**start) + (2 ** (start + 1))) // 2)
    l_intp.extend(
        [x for y in zip(l[1:], [i + j for i, j in zip(l[1:], l[:-1])]) for x in y][:-1]
    )
    for n in l_intp:
        yield n


@click.command(short_help="run a benchmark across kernels")
@click.option(
    "--logfile",
    "-l",
    default="benchmark.log",
    type=str,
    show_default=True,
    help="name of output log file",
)
@click.option(
    "--data_out_file",
    "-o",
    default="data.h5",
    type=str,
    show_default=True,
    help="name of output data file",
)
@click.option(
    "--interpreter",
    "-i",
    default="python3",
    type=str,
    show_default=True,
    help="python interpreter command",
)
@click.option(
    "--kernel",
    "-k",
    type=click.Choice(sorted(list(inventory.keys()))),
    multiple=True,
    help="name of kernel module, can be specified multiple times",
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
    "--n_body_power_boundaries",
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
    "-d",
    default="plot",
    type=click.Choice(["plot", "log", "none"]),
    show_default=True,
    help="what to show during benchmark",
)
@click.option(
    "--threads",
    "-p",
    type=click.Choice([str(i) for i in range(1, MAX_TREADS + 1)]),
    multiple=True,
    help=(
        "number of threads/processes for parallel implementations, "
        "can be specified multiple times, defaults to maximum number of available threads"
    ),
)
def benchmark(
    logfile,
    data_out_file,
    interpreter,
    kernel,
    all_kernels,
    n_body_power_boundaries,
    save_after_iteration,
    min_iterations,
    min_total_runtime,
    display,
    threads,
):
    """run a benchmark across kernels"""

    if all_kernels:
        kernels = sorted(list(inventory.keys()))
    else:
        kernels = list(kernel)

    threads = [MAX_TREADS] if len(threads) == 0 else sorted([int(n) for n in threads])

    results_dict = {
        kernel_name: {threads_num: dict() for threads_num in range(1, MAX_TREADS + 1)}
        for kernel_name in kernels
    }
    outputlines_list = []

    fh = open(logfile, "w")

    def shutdown():
        fh.close()

    atexit.register(shutdown)

    for kernel_name in kernels:
        inventory[kernel_name].load_meta()
        parallel = inventory[kernel_name]["parallel"]
        parallel = parallel if isinstance(parallel, bool) else False
        threads_iterator = threads if parallel else [1]
        for threads_num in threads_iterator:
            for bodies in _range(*n_body_power_boundaries):
                proc.run_command(
                    worker_command(
                        data_out_file,
                        interpreter,
                        kernel_name,
                        "galaxy",
                        {"stars_len": bodies},
                        save_after_iteration,
                        min_iterations,
                        min_total_runtime,
                        threads_num,
                    ),
                    unbuffer=True,
                    processing=_process_data(
                        kernel_name,
                        threads_num,
                        bodies,
                        results_dict,
                        outputlines_list,
                        fh,
                        display,
                    ),
                )
                fh.flush()
