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
import sys
from typing import Generator

import termplotlib as tpl
import click

from .worker import worker_command

from ..lib.baseuniverse import BaseUniverse
from ..lib.const import Stream
from ..lib.debug import typechecked
from ..lib.kernel import KERNELS
from ..lib.proc import run_command
from ..lib.variation import Variation

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ROUTINES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


@typechecked
class _Processing:
    "reading a worker log in realtime"

    def __init__(
        self,
        kernel: str,
        variation: Variation,
        results: dict,
        length: int,
        fh: TextIOWrapper,
        display: str,
    ):
        self._kernel = kernel
        self._length = length
        self._fh = fh
        self._display = display
        self._results = results

        self._variation = variation
        variation = variation.to_dict()
        self._variation_str = ' / '.join([f'{key:s}={variation[key]:s}' for key in sorted(variation.keys())])

        self._counter = 0

        self._error_state = False
        self._error = None

    def __call__(
        self,
        stream_id: Stream,
        line: str,
    ):

        if stream_id is Stream.stderr:
            line = json.dumps({"log": "stderr", "value": line})

        self._fh.write(f"{line:s}\n")
        self._fh.flush()

        if self._display == "log":
            print(line)
        if stream_id is Stream.stderr:
            return

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

        if msg["log"] == "BEST_TIME":
            if self._length not in self._results.keys():
                self._results[self._length] = msg["value"]
            elif self._results[self._length] > msg["value"]:
                self._results[self._length] = msg["value"]
        else:
            return

        if self._display != "plot" or len(self._results) == 0:
            return

        x = sorted(list(self._results.keys()))
        y = [self._results[n]*1e-9 for n in x]
        t = shutil.get_terminal_size((80, 20))

        label = " / ".join([
            f"kernel={self._kernel:s}",
            self._variation_str,
            f"implementation={sys.implementation.name:s}",
            f"len={self._length:d}",
            f"iteration={self._counter:d}",
            f"best={y[-1]*1e-9:.02e}s",
        ])

        fig = tpl.figure()
        fig.plot(
            x,
            y,
            label=label,
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
def _range(start: int, stop: int) -> Generator:
    """special range generator, going from 2^start to 2^stop with some interpolation"""
    if start > stop:
        raise ValueError()
    for value in range(start, stop):
        yield 2 ** value
        yield round(2 ** (value + 0.5))
    yield 2 ** stop


@typechecked
def _common_initial_states(start: int, stop: int, datafile: int):
    """create common initial states for benchmarks for later evaluation of results"""
    for length in _range(start, stop):
        print(
            f"Creating initial state for {length:d} masses (max {2**stop:d}) ..."
        )
        initial_state = _UniverseZero.from_galaxy(length=length)
        initial_state.to_hdf5(fn=datafile, gn=_UniverseZero.export_name_group(kernel = "zero", length = length, steps = 0))


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

    names = sorted(list(KERNELS.keys())) if all_kernels else list(kernel)

    if common_initial_state:
        _common_initial_states(*len_range, datafile)

    fh = open(logfile, "w", encoding="utf-8")

    def shutdown():
        fh.close()

    atexit.register(shutdown)

    for name in names:

        KERNELS[name].load_meta()

        for variation in KERNELS[name].variations:

            if variation['threads'].name.startswith('t'):
                continue

            results = {}

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
                        **variation.to_dict(),
                    ),
                    unbuffer=True,
                    processing=_Processing(
                        kernel=name,
                        variation=variation,
                        results=results,
                        length=length,
                        fh=fh,
                        display=display,
                    ),
                )
