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

import atexit
from io import TextIOWrapper
import json
import shutil
from typing import Generator, List

import termplotlib as tpl
import click
import psutil

from .worker import worker_command
from ..kernel._base import UniverseBase
from ..lib.debug import typechecked
from ..lib.load import inventory
from ..lib.proc import run_command

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CONST
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

MAX_TREADS = psutil.cpu_count(logical=True)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ROUTINES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


@typechecked
class _Processing:
    "reading a worker log in realtime"

    def __init__(
        self,
        kernel: str,
        threads: int,
        bodies: int,
        results: dict,
        outputs: List[str],
        fh: TextIOWrapper,
        display: str,
    ):

        self._kernel = kernel
        self._threads = threads
        self._bodies = bodies
        self._results = results
        self._outputs = outputs
        self._fh = fh
        self._display = display

        self._counter = 0

        self._error_state = False
        self._error = None

    def __call__(
        self,
        id: int,  # 1 STDOUT, 2 STDERR
        line: str,
    ):

        self._fh.write(f"{line:s}\n")
        if self._display == "log":
            print(line)
        self._outputs.append(line)

        if not self._error_state:
            try:
                msg = json.loads(line)
            except Exception as e:
                print(line)
                self._error_state = True
                self._error = e
                return
        else:
            if line.startswith(' '):
                print(line)
                return
            print(line)
            raise self._error

        if msg["log"] == "STEP":
            self._counter = msg["counter"]

        bests = self._results[self._kernel][self._threads]
        if msg["log"] == "BEST_TIME":
            if self._bodies not in bests.keys():
                bests[self._bodies] = msg["value"]
            elif bests[self._bodies] > msg["value"]:
                bests[self._bodies] = msg["value"]
            else:
                return
        else:
            return

        if self._display != "plot":
            return

        x = sorted(list(bests.keys()))
        y = [bests[n] for n in x]
        t = shutil.get_terminal_size((80, 20))
        fig = tpl.figure()
        fig.plot(
            x,
            y,
            label=f"{self._kernel:s} / t={self._threads:d} / n={self._bodies:d} / i={self._counter:d} / b={y[-1]:.02e}s",
            width=t.columns,
            height=t.lines,
            extra_gnuplot_arguments=[
                "set logscale x 2",
                'set format y "10^{%L}"',
                "set logscale y 10",
            ],
        )
        fig.show()


@typechecked
def _range(start: int, end: int) -> Generator:
    """special range generator, going from 2^start to 2^end with some interpolation"""
    assert start <= end
    state = start
    while True:
        value = 2**state
        yield value
        if state == end:
            break
        yield value + value // 2
        state += 1


@typechecked
class _UniverseZero(UniverseBase):
    "Generating common start universe"

    def step_stage1(self):
        "not required here"


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
    default="data_out.h5",
    type=str,
    show_default=True,
    help="name of output data file",
)
@click.option(
    "--data_in_file",
    "-j",
    default="data_in.h5",
    type=str,
    show_default=True,
    help="name of input data file",
)
@click.option(
    "--data_in_group",
    "-g",
    default="state",
    type=str,
    show_default=True,
    help="name of input data group",
)
@click.option(
    "--read_initial_state",
    "-r",
    default=False,
    is_flag=True,
    show_default=True,
    help="name of input data file",
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
    data_in_file,
    data_in_group,
    read_initial_state,
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
        names = sorted(list(inventory.keys()))
    else:
        names = list(kernel)

    threads = [MAX_TREADS] if len(threads) == 0 else sorted([int(n) for n in threads])

    results = {
        name: {threads_num: dict() for threads_num in range(1, MAX_TREADS + 1)}
        for name in names
    }
    outputs = []

    fh = open(logfile, "w", encoding="utf-8")

    def shutdown():
        fh.close()

    atexit.register(shutdown)

    if read_initial_state:
        for bodies in _range(*n_body_power_boundaries):
            print(f'Creating initial state for {bodies:d} masses (max {2**n_body_power_boundaries[1]:d}) ...')
            initial_state = _UniverseZero.from_galaxy(stars_len=bodies)
            initial_state.to_hdf5(fn = data_in_file, gn = f'{data_in_group:s}_{bodies:d}')

    for name in names:
        inventory[name].load_meta()
        parallel = inventory[name]["parallel"]
        parallel = parallel if isinstance(parallel, bool) else False
        threads_iterator = threads if parallel else [1]

        for threads_num in threads_iterator:
            for bodies in _range(*n_body_power_boundaries):
                run_command(
                    worker_command(
                        data_out_file = data_out_file,
                        interpreter = interpreter,
                        kernel = name,
                        len = bodies,
                        save_after_iteration = save_after_iteration,
                        data_in_file = data_in_file,
                        data_in_group = f'{data_in_group:s}_{bodies:d}',
                        read_initial_state = read_initial_state,
                        min_iterations = min_iterations,
                        min_total_runtime = min_total_runtime,
                        threads = threads_num,
                    ),
                    unbuffer=True,
                    processing=_Processing(
                        kernel = name,
                        threads = threads_num,
                        bodies = bodies,
                        results = results,
                        outputs = outputs,
                        fh = fh,
                        display = display,
                    ),
                )
                fh.flush()
