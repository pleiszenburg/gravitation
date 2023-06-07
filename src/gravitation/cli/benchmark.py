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
from ..lib.baseuniverse import BaseUniverse
from ..lib.const import Stream
from ..lib.debug import typechecked
from ..lib.kernel import KERNELS
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
        length: int,
        results: dict,
        outputs: List[str],
        fh: TextIOWrapper,
        display: str,
    ):
        self._kernel = kernel
        self._length = length
        self._results = results
        self._outputs = outputs
        self._fh = fh
        self._display = display

        self._counter = 0

        self._error_state = False
        self._error = None

    def __call__(
        self,
        stream_id: Stream,
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
            print(line)
            if line.startswith(" "):
                return
            raise self._error

        if msg["log"] == "STEP":
            self._counter = msg["counter"]

        bests = self._results[self._kernel]
        if msg["log"] == "BEST_TIME":
            if self._length not in bests.keys():
                bests[self._length] = msg["value"]
            elif bests[self._length] > msg["value"]:
                bests[self._length] = msg["value"]
            else:
                return
        else:
            return

        if self._display != "plot":
            return

        x = sorted(list(bests.keys()))
        y = [bests[n]*1e-9 for n in x]
        t = shutil.get_terminal_size((80, 20))
        fig = tpl.figure()
        fig.plot(
            x,
            y,
            label=f"{self._kernel:s} / n={self._length:d} / i={self._counter:d} / b={y[-1]*1e-9:.02e}s",
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
class _UniverseZero(BaseUniverse):
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
    "--kernel",
    "-k",
    type=click.Choice(sorted(list(KERNELS.keys()))),
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
    type=click.Choice(["plot", "log", "none"]),
    show_default=True,
    help="what to show during benchmark",
)
def benchmark(
    logfile,
    datafile,
    common_initial_state,
    kernel,
    all_kernels,
    len_range,
    save_after_iteration,
    min_iterations,
    min_total_runtime,
    display,
):
    """run a benchmark across kernels"""

    if all_kernels:
        names = sorted(list(KERNELS.keys()))
    else:
        names = list(kernel)

    results = {
        name: {}
        for name in names
    }
    outputs = []

    fh = open(logfile, "w", encoding="utf-8")

    def shutdown():
        fh.close()

    atexit.register(shutdown)

    if common_initial_state:
        for length in _range(*len_range):
            print(
                f"Creating initial state for {length:d} masses (max {2**len_range[1]:d}) ..."
            )
            initial_state = _UniverseZero.from_galaxy(length=length)
            initial_state.to_hdf5(fn=datafile, gn=_UniverseZero.export_name_group(kernel = "zero", length = length, steps = 0))

    for name in names:
        KERNELS[name].load_meta()
        parallel = KERNELS[name]["parallel"]
        parallel = parallel if isinstance(parallel, bool) else False

        for length in _range(*len_range):
            run_command(
                worker_command(
                    datafile=datafile,
                    kernel=name,
                    length=length,
                    save_after_iteration=save_after_iteration,
                    read_initial_state=common_initial_state,
                    min_iterations=min_iterations,
                    min_total_runtime=min_total_runtime,
                ),
                unbuffer=True,
                processing=_Processing(
                    kernel=name,
                    length=length,
                    results=results,
                    outputs=outputs,
                    fh=fh,
                    display=display,
                ),
            )
            fh.flush()
